"""Score an explicitly bound five-method core ROI without altering old results."""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
from pathlib import Path

import numpy as np

from .contracts import read_json, sha256, verify_sha
from .proxy_common_score import CORE_METHODS, METRICS, complete_mean, load_packet, mask_digest, unique_rows
from .proxy_checkpoint_export import LEGACY_NAMES, WRAPPER_SHA, REGISTRY_SHA
from .proxy_roi import PROTOCOL, check_record_root, load_roi_manifest, roi_common_masks, validate_reference
from .score_native_gcp import write_json


def validate_packet_index(index, roi):
    check_record_root(index)
    if (index['schema'] != 'umgs_core_roi_proxy_packet_index_v1' or index['protocol'] != PROTOCOL
            or index['scene'] != roi['scene'] or index['methods'] != list(CORE_METHODS)
            or index['target_names'] != roi['target_names'] or index['depth_units'] != 'source_common_SfM_reconstruction_units_not_metres'
            or index['resized_depth'] or index['method_specific_alignment']):
        raise ValueError('Unsupported packet index contract')
    if not index['same_reference_camera_verified'] or not index['all_five_methods_bound']:
        raise ValueError('Packet index camera/producer proof incomplete')
    rows = unique_rows(index['targets'], 'image_name', roi['target_names'])
    for target in roi['targets']:
        row = rows[target['image_name']]
        if row['reference_camera_sha256'] != target['camera']['record_sha256'] or set(row['packets']) != set(CORE_METHODS):
            raise ValueError('Method/reference camera or method set mismatch')
        # RGB and UMGS geometry are a verified common checkpoint alias, not
        # two independent trained models in this experiment.
        if row['packets']['rgb_anchor'] != row['packets']['umgs']:
            raise ValueError('RGB/UMGS geometry alias identity mismatch')
    return rows


def verify_producers(index, roi, rows):
    """Rebind packet entries to real parent manifests, not stored PASS flags."""
    evidence = index['producer_evidence']
    for record in evidence:
        verify_sha(record['path'], record['sha256'])
    if index['producer_kind'] == 'reviewed_legacy_road_common_proxy_summary':
        if roi['scene'] != 'road' or len(evidence) != 1:
            raise ValueError('Historical producer only supports Road')
        parent = read_json(evidence[0]['path'])
        if (parent['schema'] != 'umgs_multi_method_common_proxy_scoring_v1'
                or parent['scene'] != 'road' or parent['method_set'] != list(CORE_METHODS)
                or parent['wrapper_sha256'] != WRAPPER_SHA or parent['registry_sha256'] != REGISTRY_SHA
                or parent['target_count'] != len(roi['targets'])):
            raise ValueError('Historical producer contract mismatch')
        lookup = unique_rows(parent['targets'], 'target', roi['target_names'])
        for target in roi['targets']:
            old = lookup[target['image_name']]
            if (old['fingerprint_sha256'] != target['camera']['fingerprint']['payload_sha256']
                    or old['reference']['sha256'] != target['reference']['sha256']
                    or old['packets'] != rows[target['image_name']]['packets']):
                raise ValueError('Historical packet/camera/reference association mismatch')
    elif index['producer_kind'] == 'fresh_core_roi_common_camera_exports_v1':
        if len(evidence) != 4:
            raise ValueError('Require four physical model exports, including shared RGB anchor')
        producers = unique_rows([read_json(r['path']) for r in evidence], 'method_id', CORE_METHODS[:-1])
        for method, producer in producers.items():
            if (producer['schema'] != 'umgs_core_roi_common_camera_export_v1'
                    or producer['scene'] != roi['scene'] or producer['roi_manifest_sha256'] != index['roi_manifest_sha256']
                    or producer['formula'] != 'M1/A' or producer['depth_units'] != index['depth_units']
                    or producer['method_specific_alignment'] or producer['resized_depth']):
                raise ValueError('Fresh producer contract mismatch')
            packets = unique_rows(producer['packets'], 'image_name', roi['target_names'])
            for target in roi['targets']:
                packet = packets[target['image_name']]
                if (packet['reference_camera'] != target['camera'] or packet['packet_ref_passed'] is not True
                        or packet['legacy'] != rows[target['image_name']]['packets'][method]):
                    raise ValueError('Fresh packet/camera association mismatch')
    else:
        raise ValueError('Unknown packet producer; native-depth resizing forbidden')


def score(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    verify_sha(args.job, args.job_sha256)
    job = read_json(args.job)
    if job['schema'] != 'umgs_core_roi_proxy_score_job_v1':
        raise ValueError('Unknown ROI scoring job')
    roi_path = Path(job['roi_manifest']['path'])
    roi = load_roi_manifest(roi_path, job['roi_manifest']['sha256'])
    index_record = job['packet_index']
    verify_sha(index_record['path'], index_record['sha256'])
    index = read_json(index_record['path'])
    if index['roi_manifest_sha256'] != job['roi_manifest']['sha256']:
        raise ValueError('Packet index bound to a different ROI manifest')
    methods = validate_packet_index(index, roi)
    verify_producers(index, roi, methods)
    verify_sha(job['frozen_metric_wrapper'], WRAPPER_SHA)
    spec = importlib.util.spec_from_file_location('frozen_roi_metrics', job['frozen_metric_wrapper'])
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    modules = wrapper.audited_modules()
    gradient = importlib.import_module('tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'masks').mkdir()
    details, rows = [], []
    for ti, target in enumerate(roi['targets']):
        name = target['image_name']
        ref = load_packet(target['reference'], {'depth', 'valid', 'triangle_id', 'barycentric'})
        valid = validate_reference(ref, target['camera'])
        with (roi_path.parent / target['mask']['path']).open('rb') as f:
            reference_roi = np.load(f, allow_pickle=False).astype(bool)
        with (roi_path.parent / target['subregions']['path']).open('rb') as f:
            regions = np.load(f, allow_pickle=False)
        packets = {m: load_packet(methods[name]['packets'][m], LEGACY_NAMES) for m in CORE_METHODS}
        gates = {m: wrapper.m1_a_gate(modules, p, ref['depth'].shape) for m, p in packets.items()}
        masks, coverage = roi_common_masks(ref['depth'], valid, packets, CORE_METHODS, reference_roi)
        detail = dict(image_name=name, packet_gates=gates, masks={})
        for mask_name, mask in masks.items():
            domain = gradient.reference_high_gradient_domain(ref['depth'], mask)
            artifacts = {}
            for label, value in [('mask', mask), ('gradient', domain.high_mask)]:
                path = args.output / 'masks' / f'{ti:03d}_{mask_name}_{label}.npy'
                with path.open('xb') as f:
                    np.save(f, value.astype(np.uint8), allow_pickle=False)
                artifacts[label] = dict(path=path.relative_to(args.output).as_posix(), sha256=sha256(path), matrix_sha256=mask_digest(value))
            counts = dict(**coverage[mask_name], support_coverage=float(mask.mean()),
                reference_roi_fraction_frame=float(reference_roi.mean()),
                reference_roi_fraction_mesh=float(reference_roi.sum() / valid.sum()))
            detail['masks'][mask_name] = dict(**counts, artifacts=artifacts,
                gradient_threshold=domain.threshold, methods={}, subregions=[dict(
                    region=i, reference_roi_pixels=int(np.count_nonzero(regions == i)),
                    supported_pixels=int(np.count_nonzero(mask & (regions == i))),
                    coverage=float(np.count_nonzero(mask & (regions == i)) / np.count_nonzero(regions == i)) if np.any(regions == i) else None
                ) for i in range(16)])
            for method in CORE_METHODS:
                metric = modules.packets._candidate_metrics(ref['depth'], packets[method]['expected_camera_z'], mask, domain) if mask.any() else {}
                values = {k: float(metric[k]) if metric.get(k) is not None and np.isfinite(metric[k]) else None for k in METRICS}
                detail['masks'][mask_name]['methods'][method] = values
                rows.append(dict(scene=roi['scene'], image_name=name, method=method, mask=mask_name, **counts, **values))
        details.append(detail)
    aggregate = {mask: {method: {metric: complete_mean([r[metric] for r in rows if r['method'] == method and r['mask'] == mask], len(roi['targets']))
        for metric in METRICS} for method in CORE_METHODS} for mask in masks}
    with (args.output / 'per_target_metrics.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(schema='umgs_core_roi_proxy_scores_v1', protocol=PROTOCOL, scene=roi['scene'],
        method_set=list(CORE_METHODS), target_count=len(details), targets=details, aggregate=aggregate,
        all_declared_metric_values_complete=all(v['complete'] for a in aggregate.values() for b in a.values() for v in b.values()),
        roi_manifest=job['roi_manifest'], packet_index=index_record, job_sha256=args.job_sha256,
        gpu_used=False, method_specific_alignment=False, depth_units=index['depth_units'],
        scene_set=['road', 'five_k'], mixed_six_scene_macro_allowed=False,
        old_results_modified=False, stage_review='PENDING',
        claim_boundary='Engineering proxy on reported reference-visible ROI support; not ground truth')
    write_json(args.output / 'summary.json', result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--job-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    score(p.parse_args())


if __name__ == '__main__':
    main()
