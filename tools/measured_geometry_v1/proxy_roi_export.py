"""Read-only full-model export on hash-bound reference ROI camera grids."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from .contracts import read_json, record_hash, sha256, verify_sha, verify_source_snapshot
from .ms_checkpoint_export import write_json, json_summary
from .native_moments import packet_from_reference
from .proxy_checkpoint_export import BINDING_SHA, FORWARD_SHA, WRAPPER_SHA, source_moments_with_legacy_parity
from .proxy_roi import PROTOCOL, load_roi_manifest, validate_camera
from .umgs_checkpoint_export import support_identity


UNITS = 'source_common_SfM_reconstruction_units_not_metres'
SCHEMA = 'umgs_core_roi_common_camera_export_v1'
PROJECTION_TOLERANCE = 5e-5  # Existing canonical-reference float32 projection bound.


def projection_matrix(camera):
    validate_camera(camera)
    w, h = camera['width'], camera['height']
    near, far = .01, 100.
    p = np.zeros((4, 4), dtype=np.float64)
    p[0, 0], p[1, 1] = 2 * camera['fx'] / w, 2 * camera['fy'] / h
    # Rasterizer ndcToPix yields integer-centered array coordinates. The mesh
    # ray at array (x,y) uses corner-origin (x+.5,y+.5), including off-center K.
    p[2, 0], p[2, 1] = 2 * camera['cx'] / w - 1, 2 * camera['cy'] / h - 1
    p[2, 2], p[2, 3], p[3, 2] = far / (far - near), 1, -far * near / (far - near)
    return p


def projection_audit(camera, matrix, tolerance):
    validate_camera(camera)
    w, h = camera['width'], camera['height']
    pixels = np.array([[.5, .5], [w-.5, .5], [.5, h-.5], [w-.5, h-.5],
                       [w/2, h/2], [camera['cx'], camera['cy']], [137.25, 209.75]])
    z = np.array([1., 2., 3., 4., 5., 6., 7.])
    rays = (pixels - [camera['cx'], camera['cy']]) / [camera['fx'], camera['fy']]
    xyz = np.c_[rays * z[:, None], z, np.ones(len(z))]
    clip = xyz @ np.asarray(matrix, dtype=np.float64)
    ndc = clip[:, :2] / clip[:, 3:4]
    index_pixels = ((ndc + 1) * [w, h] - 1) / 2
    error = float(np.max(np.abs(index_pixels + .5 - pixels)))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError('Reference/rasterizer pixel-center projection mismatch')
    return dict(max_pixel_error=error, tolerance_px=tolerance, test_corner_origin_pixels=pixels.tolist(),
                rasterizer_index_pixels=index_pixels.tolist(), half_pixel_conversion='corner = array_index + 0.5')


def normalization_record(value):
    transform = np.asarray(value['transform'], dtype=np.float64)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, [0, 0, 0, 1]])
    scale = float(value['scale'])
    if (transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.isfinite(scale) or scale <= 0
            or not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-12)
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), rtol=0, atol=1e-6)
            or np.linalg.det(transform[:3, :3]) <= 0):
        raise ValueError('Invalid saved training normalization; fitting prohibited')
    return transform, scale


def model_pose(camera, normalization):
    _, source = validate_camera(camera)
    transform, scale = normalization_record(normalization)
    result = transform @ source
    result[:3, 3] *= scale
    recovered = result.copy()
    recovered[:3, 3] /= scale
    recovered = np.linalg.inv(transform) @ recovered
    error = float(np.max(np.abs(recovered - source)))
    if error > 1e-7:
        raise ValueError('Source/dataparser pose roundtrip mismatch')
    return result, error


def renderer_camera(record, normalization):
    import torch
    from utils.graphics_utils import getWorld2View2
    pose, error = model_pose(record, normalization)
    inverse = np.linalg.inv(pose)
    r, t = inverse[:3, :3].T, inverse[:3, 3]
    world = torch.tensor(getWorld2View2(r, t), dtype=torch.float32, device='cuda').transpose(0, 1)
    p = projection_matrix(record)
    exact = projection_audit(record, p, 1e-9)
    projection = torch.tensor(p, dtype=torch.float32, device='cuda')
    actual = projection_audit(record, projection.cpu().numpy(), PROJECTION_TOLERANCE)
    actual_view = world.cpu().numpy().T
    if not np.allclose(actual_view, inverse, rtol=0, atol=5e-6):
        raise ValueError('Float32 renderer world-to-camera differs from canonical construction')
    full = world.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)
    w, h = record['width'], record['height']
    camera = SimpleNamespace(image_name=record['image_name'], colmap_id=record['image_id'], uid=record['image_id'],
        image_width=w, image_height=h, FoVx=2*math.atan(w/(2*record['fx'])), FoVy=2*math.atan(h/(2*record['fy'])),
        znear=.01, zfar=100., R=r, T=t, world_view_transform=world, projection_matrix=projection,
        full_proj_transform=full, camera_center=world.inverse()[3, :3], c2w_opencv=pose)
    audit = dict(reference_camera_sha256=record['record_sha256'], source_to_model_pose_roundtrip_max_abs=error,
        model_c2w_opencv=pose.tolist(), model_w2c_float32=actual_view.tolist(), projection_row_float32=projection.cpu().tolist(),
        world_view_transform=world.cpu().tolist(), full_proj_transform=full.cpu().tolist(),
        exact_projection=exact, float32_projection=actual, model_specific_fit=False)
    return camera, audit


def authenticated_model(job, parent, modules):
    import torch
    checkpoint = parent['checkpoint']
    verify_sha(checkpoint, parent['checkpoint_sha256'])
    if parent['method'] != job['method_id'] or parent['scene'] != 'gcp_5000_20260602':
        raise ValueError('Native export model/scene identity mismatch')
    if job['method_id'] == 'umgs':
        model_record = parent['models']['D']
        if checkpoint != model_record['checkpoint'] or parent['checkpoint_sha256'] != model_record['checkpoint_sha256']:
            raise ValueError('UMGS anchor identity mismatch')
        verify_sha(model_record['ply'], model_record['ply_sha256'])
        captured, iteration = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if iteration != 30000 or iteration != model_record['iteration'] or len(captured) != 12 or captured[0] != 3:
            raise ValueError('UMGS endpoint schema/iteration mismatch')
        arrays = {key: captured[i].detach().cpu().numpy() for key, i in [('xyz', 1), ('scaling', 4), ('rotation', 5), ('opacity', 6)]}
        geometry = support_identity(arrays)
        if geometry != model_record['support'] or not parent['rgb_anchor_geometry_shared_only_after_bitwise_support_check']:
            raise ValueError('UMGS/RGB ordered support identity mismatch')
        if any(r['support'] != geometry for r in parent['models'].values()):
            raise ValueError('Historical band support alias mismatch')
        normalization = parent['normalization']
        matrix, scale = normalization_record(normalization)
        if not np.array_equal(matrix, np.eye(4)) or scale != 1:
            raise ValueError('UMGS must retain source world identity')
    else:
        dp = job['dataparser']
        verify_sha(dp['path'], dp['sha256'])
        normalization = read_json(dp['path'])
        if (normalization != parent['normalization'] or dp['sha256'] != parent['normalization_sha256']):
            raise ValueError('Saved normalization differs from accepted native export')
        arrays, geometry = modules.e3.load_mss_checkpoint_geometry_e3(Path(checkpoint))
        if geometry['gaussian_count'] != parent['gaussian_count']:
            raise ValueError('Native/common exporter Gaussian count mismatch')
    normalization_record(normalization)
    return modules.e1.make_geometry_model(arrays), geometry, normalization


def export(args):
    started = time.monotonic()
    verify_sha(args.job, args.job_sha256)
    job = read_json(args.job)
    if job['schema'] != 'umgs_core_roi_common_camera_export_job_v1' or job['method_id'] not in ['umgs','jo','ms_splatting_neural','sig_mechanism']:
        raise ValueError('Unsupported read-only export job')
    if args.output.exists():
        raise FileExistsError(args.output)
    roi = load_roi_manifest(job['roi_manifest']['path'], job['roi_manifest']['sha256'])
    if roi['scene'] != 'five_k' or len(roi['targets']) != 13:
        raise ValueError('This admission is limited to the 13 frozen 5K targets')
    verify_sha(job['native_export']['path'], job['native_export']['sha256'])
    parent = read_json(job['native_export']['path'])
    verify_sha(job['wrapper'], WRAPPER_SHA)
    spec = importlib.util.spec_from_file_location('frozen_self5_roi_export', job['wrapper'])
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    modules, exporter = wrapper.audited_modules(), wrapper.audited_exporter()
    identity = verify_source_snapshot(job['reference_root'], job['reference_manifest_sha256'])
    from .preflight import _load_reference_module
    reference = _load_reference_module(Path(job['reference_root']), 'metric_depth_packet')
    import torch
    import gaussian_renderer
    import diff_gaussian_rasterization as rasterizer
    runtime = wrapper.audited_runtime_root()
    renderer = runtime / 'gaussian_renderer/__init__.py'
    if Path(gaussian_renderer.__file__).resolve() != renderer.resolve():
        raise ValueError('Wrong frozen renderer import')
    verify_sha(renderer, wrapper.AUDITED_RUNTIME_SHA256['gaussian_renderer/__init__.py'])
    verify_sha(runtime / 'submodules/diff-gaussian-rasterization/cuda_rasterizer/forward.cu', FORWARD_SHA)
    verify_sha(rasterizer.__file__, BINDING_SHA)
    extension = Path(rasterizer._C.__file__).resolve()
    verify_sha(extension, job['extension_sha256'])
    model, geometry, normalization = authenticated_model(job, parent, modules)
    scale = float(normalization['scale'])
    count = geometry['gaussian_count']
    nan_rows = np.asarray(geometry.get('raw_log_scaling_complete_nan_row_indices', []), dtype=np.int64)
    background = torch.zeros(3, dtype=torch.float32, device='cuda')
    color = torch.full((count, 3), .5, dtype=torch.float32, device='cuda')
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'legacy').mkdir()
    (args.output / 'v2').mkdir()
    (args.output / 'progress').mkdir()
    rows = []
    with torch.no_grad():
        for target in roi['targets']:
            camera, audit = renderer_camera(target['camera'], normalization)
            result = gaussian_renderer.render(camera, model, pipeline, background, override_color=color,
                use_trained_exp=False, separate_sh=False, return_expected_camera_z_packet=True,
                opacity_epsilon=1e-6, variance_clamp_tolerance=1e-6)
            if nan_rows.size and not bool((result['radii'][torch.as_tensor(nan_rows, device='cuda')] == 0).all()):
                raise ValueError('Historical NaN-scale sentinel contributed to render')
            legacy_dp = exporter.packet_tensor_to_named_arrays(result['expected_camera_z_packet'],
                opacity_epsilon=1e-6, variance_clamp_tolerance=1e-6)
            legacy = wrapper.convert_packet_to_source_units(modules, legacy_dp, scale)
            gate = wrapper.m1_a_gate(modules, legacy, (camera.image_height, camera.image_width))
            raw = source_moments_with_legacy_parity(result['expected_camera_z_packet'].cpu().numpy(),
                result['depth'].cpu().numpy(), legacy_dp, legacy, scale)
            packet, consistency = packet_from_reference(raw, reference)
            if consistency['passed'] is not True:
                raise ValueError('Packet/reference contract failed')
            stem = Path(target['image_name']).stem
            lp, vp = args.output/'legacy'/(stem+'.npz'), args.output/'v2'/(stem+'.npz')
            modules.packets.deterministic_npz(lp, legacy)
            with vp.open('xb') as f:
                np.savez_compressed(f, **packet)
            rows.append(dict(image_name=target['image_name'], reference_camera=target['camera'], actual_camera=audit,
                legacy=dict(path=str(lp), sha256=sha256(lp)), v2=dict(path=str(vp), sha256=sha256(vp)),
                legacy_validation=gate, packet_ref=consistency, packet_ref_passed=True,
                legacy_raw_accumulator_bitwise_parity=True))
            write_json(args.output/'progress'/f'{len(rows):02d}.json', dict(method=job['method_id'], completed=len(rows), total=13, last_view=rows[-1]))
            print(json.dumps(dict(method=job['method_id'], completed=len(rows), total=13, image_name=target['image_name'])), flush=True)
            del result, legacy_dp, legacy, raw, packet, camera
    verify_sha(parent['checkpoint'], parent['checkpoint_sha256'])
    manifest = dict(schema=SCHEMA, protocol=PROTOCOL, scene=roi['scene'], method_id=job['method_id'],
        roi_manifest_sha256=job['roi_manifest']['sha256'], formula='M1/A', depth_units=UNITS,
        method_specific_alignment=False, resized_depth=False, native_export=job['native_export'],
        checkpoint=parent['checkpoint'], checkpoint_sha256=parent['checkpoint_sha256'], normalization=normalization,
        geometry=geometry, target_count=len(rows), packets=rows, wrapper_sha256=WRAPPER_SHA,
        reference_identity=identity, renderer_sha256=sha256(renderer), binding_sha256=BINDING_SHA,
        forward_sha256=FORWARD_SHA, extension_sha256=job['extension_sha256'],
        generator_source_sha256=sha256(__file__), orchestrator_commit=job['orchestrator_commit'], job_sha256=args.job_sha256,
        full_model_no_roi_crop=True, same_render_call_real_H=True, native_gsplat_track=False,
        legacy_six_arrays_and_numeric_valid_unchanged=True, v2_valid_does_not_replace_legacy_proxy_mask=True,
        rasterizer_early_termination='test_T < 1e-4', wall_seconds=time.monotonic()-started)
    manifest = json_summary(manifest)
    manifest['records_root_sha256'] = record_hash(manifest)
    write_json(args.output/'export_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--job-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    export(parser.parse_args())


if __name__ == '__main__':
    main()
