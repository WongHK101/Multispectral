"""Reference-only ROI materialization; never crop or load Gaussian models."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from pyproj import Transformer

from .contracts import read_json, record_hash, safe_member, sha256, verify_sha
from .proxy_common_score import CORE_METHODS, common_masks, mask_digest, unique_rows
from .score_native_gcp import write_json


SCENES = {'road': ('road_01_20260602_1648_40m', 12), 'five_k': ('gcp_5000_20260602', 13)}
PROTOCOL = 'umgs_core_reference_first_hit_roi_proxy_v1'
MANIFEST = 'umgs_core_reference_first_hit_roi_manifest_v1'
PIXELS = 'openmvs_rasterizer_px_plus_half_v1'


def bound_json(record):
    verify_sha(record['path'], record['sha256'])
    return read_json(record['path'])


def check_record_root(value):
    if record_hash({k: v for k, v in value.items() if k != 'records_root_sha256'}) != value['records_root_sha256']:
        raise ValueError('Record root mismatch')


def camera_record(image, camera):
    from utils.read_write_model import qvec2rotmat
    if camera.model != 'PINHOLE' or camera.width <= 0 or camera.height <= 0:
        raise ValueError('Expected frozen undistorted PINHOLE camera')
    scale = min(1., 1200. / camera.width)
    k = np.asarray(camera.params, dtype=np.float64) * scale
    r = qvec2rotmat(image.qvec)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3], c2w[:3, 3] = r.T, -r.T @ image.tvec
    result = dict(image_name=image.name, image_id=int(image.id), camera_id=int(image.camera_id),
        width=int(round(camera.width * scale)), height=int(round(camera.height * scale)),
        fx=float(k[0]), fy=float(k[1]), cx=float(k[2]), cy=float(k[3]),
        source_c2w_opencv=c2w.tolist(), pixel_sampling=PIXELS, construction='colmap_float64',
        resize='openmvs_max_width_1200_shared_scale_python_round_no_crop_v1')
    result['record_sha256'] = record_hash(result)
    return result


def fingerprint_camera_record(record, image_name):
    from tools.depth_reference_geometry_v2.render_openmvs_canonical_camera import load_umgs_canonical_camera
    # Reuse the exact camera of the existing canonical reference, not an
    # independently resized COLMAP camera with merely the same image shape.
    c = load_umgs_canonical_camera(record['path'], expected_file_sha256=record['sha256'],
        expected_payload_sha256=record['payload_sha256'], expected_target=image_name)
    result = dict(image_name=image_name, image_id=c.view.image_id,
        camera_id=None, width=c.width, height=c.height, fx=c.fx, fy=c.fy, cx=c.cx, cy=c.cy,
        source_c2w_opencv=c.camera_to_world.tolist(), pixel_sampling=PIXELS,
        construction='frozen_canonical_fingerprint_float32', fingerprint=record,
        resize='existing_canonical_reference_no_resize')
    result['record_sha256'] = record_hash(result)
    return result


def validate_camera(camera):
    if record_hash({k: v for k, v in camera.items() if k != 'record_sha256'}) != camera['record_sha256']:
        raise ValueError('Reference camera hash mismatch')
    if camera['pixel_sampling'] != PIXELS:
        raise ValueError('Unsupported pixel sampling')
    if any(type(camera[k]) is not int or camera[k] <= 0 for k in ('width', 'height')):
        raise ValueError('Invalid camera dimensions')
    k = np.array([camera[x] for x in ('fx', 'fy', 'cx', 'cy')], dtype=np.float64)
    p = np.asarray(camera['source_c2w_opencv'], dtype=np.float64)
    if not np.isfinite(k).all() or np.any(k[:2] <= 0) or p.shape != (4, 4) or not np.isfinite(p).all():
        raise ValueError('Invalid reference intrinsics/pose')
    if not np.allclose(p[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ValueError('Invalid affine pose')
    tolerances = {'colmap_float64': 1e-10, 'frozen_canonical_fingerprint_float32': 1e-6}
    if camera['construction'] not in tolerances:
        raise ValueError('Unknown reference camera construction')
    if not np.allclose(p[:3, :3].T @ p[:3, :3], np.eye(3), rtol=0, atol=tolerances[camera['construction']]) or np.linalg.det(p[:3, :3]) <= 0:
        raise ValueError('Reference pose must be proper rigid rotation')
    return k, p


def validate_reference(packet, camera):
    validate_camera(camera)
    if set(packet) != {'depth', 'valid', 'triangle_id', 'barycentric'}:
        raise ValueError('Reference keyset mismatch')
    shape = (camera['height'], camera['width'])
    if any(packet[k].shape != shape for k in ('depth', 'valid', 'triangle_id')) or packet['barycentric'].shape != (*shape, 3):
        raise ValueError('Reference/camera grid mismatch; resizing prohibited')
    d, v, tid = packet['depth'], packet['valid'], packet['triangle_id']
    if not np.isin(v, [0, 1]).all() or not np.issubdtype(tid.dtype, np.integer):
        raise ValueError('Malformed reference valid/triangle_id')
    valid = v.astype(bool)
    if not np.array_equal(valid, np.isfinite(d) & (tid >= 0)) or not valid.any() or not (d[valid] > 0).all():
        raise ValueError('Invalid reference first-hit depths')
    bary = packet['barycentric'][valid]
    if not np.isfinite(bary).all() or not np.allclose(bary.sum(axis=1), 1., rtol=0, atol=1e-5):
        raise ValueError('Malformed reference barycentric data')
    return valid


def roi_mask(depth, valid, camera, transform, bounds, xy_transform):
    """Retain full-mesh first hits, including occlusion by outside-ROI objects."""
    k, pose = validate_camera(camera)
    if depth.shape != (camera['height'], camera['width']) or valid.shape != depth.shape:
        raise ValueError('ROI camera grid mismatch')
    b = np.asarray(bounds, dtype=np.float64)
    r, t = np.asarray(transform['rotation']), np.asarray(transform['translation'])
    s = float(transform['scale'])
    if (b.shape != (4,) or not np.isfinite(b).all() or b[0] >= b[2] or b[1] >= b[3]
            or r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all()
            or not np.isfinite(t).all() or not np.isfinite(s) or s <= 0
            or not np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-10) or np.linalg.det(r) <= 0):
        raise ValueError('Invalid frozen transform/ROI')
    if transform['method_specific_fit'] or transform['lidar_used_in_fit']:
        raise ValueError('Method/LiDAR fit prohibited')
    y, x = np.nonzero(valid)
    z = np.asarray(depth[y, x], dtype=np.float64)
    if not (np.isfinite(z) & (z > 0)).all():
        raise ValueError('Nonfinite/nonpositive first-hit depth')
    camera_xyz = np.c_[(x + .5 - k[2]) / k[0] * z, (y + .5 - k[3]) / k[1] * z, z]
    source = camera_xyz @ pose[:3, :3].T + pose[:3, 3]
    survey = s * (source @ r.T) + t
    e, n = xy_transform(survey[:, 0], survey[:, 1])
    e, n = np.asarray(e), np.asarray(n)
    if not np.isfinite(e).all() or not np.isfinite(n).all():
        raise ValueError('CRS conversion produced nonfinite coordinates')
    inside = (e > b[0]) & (e < b[2]) & (n > b[1]) & (n < b[3])
    mask = np.zeros(depth.shape, dtype=bool)
    mask[y[inside], x[inside]] = True
    ix = np.minimum(3, ((e[inside] - b[0]) / (b[2] - b[0]) * 4).astype(int))
    iy = np.minimum(3, ((n[inside] - b[1]) / (b[3] - b[1]) * 4).astype(int))
    regions = np.full(depth.shape, 255, dtype=np.uint8)
    regions[y[inside], x[inside]] = iy * 4 + ix
    return mask, regions


def roi_common_masks(reference, valid, packets, methods, reference_roi):
    roi = np.asarray(reference_roi)
    if roi.shape != reference.shape or not np.isin(roi, [0, 1]).all() or np.any(roi.astype(bool) & ~valid.astype(bool)):
        raise ValueError('ROI mask not a subset of reference support')
    # No predicted world-point ROI test: a wrong but finite depth remains an error.
    masks = common_masks(reference, valid & roi.astype(bool), packets, methods)
    denominator = int(roi.sum())
    coverage = {name: dict(reference_roi_pixels=denominator, valid_pixels=int(mask.sum()),
        missing_pixels=denominator - int(mask.sum()),
        coverage=float(mask.sum() / denominator) if denominator else None) for name, mask in masks.items()}
    return masks, coverage


def qualify_reference(audit, gates, view_reports, original):
    # Reuse all old engineering decisions without editing or reinterpreting old files.
    from tools.depth_reference_geometry_v2.openmvs_campaign_core import evaluate_audit_against_gates
    replay = evaluate_audit_against_gates(audit, gates)
    if any(original.get(k) != v for k, v in replay.items()):
        raise ValueError('Original qualification replay mismatch')
    failed = [f for f in replay['failed_gates'] if f != 'mesh_sparse_bbox_diag_ratio_out_of_range']
    for row in view_reports:
        if row['reference_roi_pixels'] == 0:
            failed.append('empty_reference_roi:' + row['image_name'])
        if row['positive_depth_p98_over_p02'] > float(gates['depth_range']['depth_p98_over_p02_max']):
            failed.append('reference_depth_range:' + row['image_name'])
    thresholds = gates['heldout_triangle_render']
    coverages = [v['full_mesh_pixels'] / (v['camera']['width'] * v['camera']['height']) for v in view_reports]
    if min(coverages) < thresholds['minimum_per_target_coverage_pass']:
        failed.append('actual_minimum_per_target_triangle_render_coverage')
    if np.median(coverages) < thresholds['minimum_median_coverage_pass']:
        failed.append('actual_minimum_median_triangle_render_coverage')
    usable = float(np.mean(np.asarray(coverages) >= thresholds['usable_target_coverage_threshold']))
    if usable < thresholds['minimum_usable_heldout_target_fraction']:
        failed.append('actual_minimum_usable_heldout_target_fraction')
    return dict(status='BLOCKED' if failed else 'ENGINEERING_PROXY_SCORABLE_ACCURACY_NOT_ESTABLISHED',
        scorable=not failed, failed_gates=failed, caution_gates=replay['caution_gates'],
        original_qualification=replay, global_bbox_role='diagnostic_only_in_new_protocol',
        roi_bbox_used_as_gate=False, all_method_metrics_complete=False,
        actual_usable_fraction=usable, usable_coverage_threshold=thresholds['usable_target_coverage_threshold'],
        claim='Source-image-only engineering proxy, not independent geometry truth')


def materialize(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_sha(args.job, args.job_sha256)
    job = read_json(args.job)
    if job['schema'] != 'umgs_core_roi_reference_job_v1' or job['protocol'] != PROTOCOL or job['scene'] not in SCENES:
        raise ValueError('Unknown ROI job/protocol/scene')
    scene_id, count = SCENES[job['scene']]
    geometry, binding = bound_json(job['geometry']), bound_json(job['binding'])
    check_record_root(geometry); check_record_root(binding)
    if geometry['scene'] != scene_id:
        raise ValueError('Shared transform scene mismatch')
    scene = next(s for s in binding['scenes'] if s['scene'] == scene_id)
    if scene['boundary'] != 'strict_inside_contains_xy' or scene['method_horizontal_conversion'] != 'EPSG:4545 to EPSG:32649 always_xy':
        raise ValueError('Frozen ROI/CRS convention mismatch')
    audit, gates, original = [bound_json(job[k]) for k in ('audit', 'gates', 'qualification')]
    if (audit['scene'] != scene_id or audit['track'] not in ('source_image_only', 'source_image_only_OpenMVS_mesh_under_benchmark_poses')
            or audit['render_max_width'] != 1200 or audit['heldout_image_count'] != count):
        raise ValueError('Reference audit scope mismatch')
    leakage = bound_json(job['source_only_leakage'])
    if leakage['pass'] is not True or leakage['leakage_issue_count'] != 0:
        raise ValueError('Source-only reconstruction leakage')
    verify_sha(audit['mesh_meta']['path'], audit['mesh_meta']['sha256'])
    for key in ('cameras', 'images', 'test_list'):
        verify_sha(job[key]['path'], job[key]['sha256'])
    if job['test_list']['sha256'] != audit['split']['test_hash']:
        raise ValueError('Audit target population mismatch')
    from utils.read_write_model import read_cameras_binary, read_images_binary
    cameras = read_cameras_binary(job['cameras']['path'])
    images = read_images_binary(job['images']['path'])
    image_map = {im.name: im for im in images.values()}
    if len(image_map) != len(images):
        raise ValueError('Duplicate source image name')
    names = Path(job['test_list']['path']).read_text().splitlines()
    if len(names) != count or len(set(names)) != count:
        raise ValueError('Frozen target count mismatch')
    refs = unique_rows(job['references'], 'image_name', names)
    transformer = Transformer.from_crs(4545, 32649, always_xy=True)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'masks').mkdir()
    rows = []
    for index, name in enumerate(names):
        image = image_map[name]
        camera = camera_record(image, cameras[image.camera_id])
        record = refs[name]['reference']
        if 'fingerprint' in refs[name]:
            camera = fingerprint_camera_record(refs[name]['fingerprint'], name)
            provenance = bound_json(refs[name]['reference_provenance'])
            if (provenance['packet_sha256'] != record['sha256'] or provenance['target'] != name
                    or provenance['mesh_ply_sha256'] != audit['mesh_meta']['sha256']
                    or provenance['fingerprint_payload_sha256'] != refs[name]['fingerprint']['payload_sha256']
                    or provenance['method_independent'] is not True or provenance['method_specific_alignment_used']):
                raise ValueError('Canonical reference provenance mismatch')
        verify_sha(record['path'], record['sha256'])
        with np.load(record['path'], allow_pickle=False) as archive:
            packet = {key: archive[key] for key in archive.files}
        valid = validate_reference(packet, camera)
        mask, regions = roi_mask(packet['depth'], valid, camera, geometry['transform'], scene['roi_bounds_xy_m'], transformer.transform)
        path = args.output / 'masks' / f'{index:03d}.npy'
        with path.open('xb') as f:
            np.save(f, mask.astype(np.uint8), allow_pickle=False)
        region_path = path.with_name(f'{index:03d}_subregions.npy')
        with region_path.open('xb') as f:
            np.save(f, regions, allow_pickle=False)
        lo, hi = np.percentile(packet['depth'][valid], [2, 98])
        rows.append(dict(image_name=name, camera=camera, reference=record,
            reference_provenance=refs[name].get('reference_provenance'),
            mask=dict(path=path.relative_to(args.output).as_posix(), sha256=sha256(path), matrix_sha256=mask_digest(mask)),
            subregions=dict(path=region_path.relative_to(args.output).as_posix(), sha256=sha256(region_path)),
            reference_roi_pixels=int(mask.sum()), full_mesh_pixels=int(valid.sum()),
            positive_depth_p98_over_p02=float(hi / lo),
            first_hit_pixel_counts_by_roi_subregion=np.bincount(regions[mask], minlength=16).reshape(4, 4).tolist()))
    eligibility = qualify_reference(audit, gates, rows, original)
    result = dict(schema=MANIFEST, protocol=PROTOCOL, scene=job['scene'], scene_id=scene_id,
        methods=list(CORE_METHODS), target_names=names, targets=rows, roi_bounds_xy_m=scene['roi_bounds_xy_m'],
        inputs={k: job[k] for k in ('geometry', 'binding', 'audit', 'gates', 'qualification', 'source_only_leakage', 'cameras', 'images', 'test_list')},
        job_sha256=args.job_sha256, qualification=eligibility, gpu_used=False, method_values_read=False,
        original_assets_modified=False, full_mesh_occlusion_preserved=True,
        coordinate_chain='source_SfM -> frozen_control_only_Sim3 -> EPSG4545 -> EPSG32649; normal_height_unchanged',
        all_method_metrics_complete=False, author_script_sha256=sha256(__file__))
    result['records_root_sha256'] = record_hash(result)
    write_json(args.output / 'roi_manifest.json', result)
    print({'scene': job['scene'], 'targets': len(rows), 'qualification': eligibility['status']}, flush=True)


def load_roi_manifest(path, expected_sha):
    verify_sha(path, expected_sha)
    manifest = read_json(path)
    check_record_root(manifest)
    if manifest['schema'] != MANIFEST or manifest['protocol'] != PROTOCOL or manifest['scene'] not in SCENES:
        raise ValueError('Unsupported ROI manifest')
    if manifest['methods'] != list(CORE_METHODS) or not manifest['full_mesh_occlusion_preserved']:
        raise ValueError('ROI comparison set/visibility mismatch')
    if not manifest['qualification']['scorable']:
        raise ValueError('ROI reference not scorable')
    for item in manifest['inputs'].values():
        verify_sha(item['path'], item['sha256'])
    names = manifest['target_names']
    if len(names) != SCENES[manifest['scene']][1] or len(set(names)) != len(names):
        raise ValueError('ROI target count mismatch')
    rows = unique_rows(manifest['targets'], 'image_name', names)
    for name in names:
        row = rows[name]
        validate_camera(row['camera'])
        if row['camera']['image_name'] != name:
            raise ValueError('ROI camera image mismatch')
        verify_sha(row['reference']['path'], row['reference']['sha256'])
        mask = safe_member(Path(path).parent, row['mask']['path'])
        verify_sha(mask, row['mask']['sha256'])
        with mask.open('rb') as f:
            array = np.load(f, allow_pickle=False)
        if (mask_digest(array) != row['mask']['matrix_sha256'] or int(array.sum()) != row['reference_roi_pixels']
                or array.shape != (row['camera']['height'], row['camera']['width'])):
            raise ValueError('ROI mask count/shape/hash mismatch')
        region_path = safe_member(Path(path).parent, row['subregions']['path'])
        verify_sha(region_path, row['subregions']['sha256'])
        with region_path.open('rb') as f:
            regions = np.load(f, allow_pickle=False)
        if regions.shape != array.shape or not np.array_equal(regions < 16, array.astype(bool)):
            raise ValueError('ROI subregion labels mismatch')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--job-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    materialize(p.parse_args())


if __name__ == '__main__':
    main()
