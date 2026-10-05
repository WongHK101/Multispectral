"""CPU scoring on one predeclared multi-method common proxy domain.

Reuses the frozen five-scene Graphdeco/OpenMVS formulas, not native gsplat
GCP/LiDAR packets. No alignment, depth rescaling, threshold fitting or ranking
on independently paired denominators is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np

from .contracts import sha256, verify_sha
from .proxy_checkpoint_export import LEGACY_NAMES, REGISTRY_SHA, WRAPPER_SHA


PRIMARY_METHODS = ("umgs", "jo", "ms_splatting_neural")
CORE_METHODS = (*PRIMARY_METHODS, "sig_mechanism", "rgb_anchor")
METRICS = ("mean_absrel", "median_absrel", "p90_absrel", "rmse", "spearman", "high_gradient_cosine")


def common_masks(reference, valid, packets, methods):
    if not methods or len(set(methods)) != len(methods) or set(packets) != set(methods):
        raise ValueError("Predeclared method set mismatch")
    reference, valid = np.asarray(reference), np.asarray(valid)
    if reference.ndim != 2 or reference.shape != valid.shape:
        raise ValueError("Reference shape mismatch")
    primary = valid.astype(bool) & np.isfinite(reference) & (reference > 0)
    opacity = np.ones(reference.shape, dtype=bool)
    variance = np.ones(reference.shape, dtype=bool)
    for method in methods:
        p = packets[method]
        if set(p) != LEGACY_NAMES or any(np.asarray(v).shape != reference.shape for v in p.values()):
            raise ValueError("Legacy proxy shape/keyset mismatch")
        z = np.asarray(p['expected_camera_z'])
        primary &= np.asarray(p['numeric_valid']).astype(bool) & np.isfinite(z) & (z > 0)
        opacity &= np.asarray(p['accumulated_opacity']) >= .5
        v = np.asarray(p['camera_z_variance'])
        variance &= np.isfinite(v) & (v >= 0)
    return dict(common_primary=primary, common_opacity_ge_0_5=primary & opacity,
                common_variance_valid=primary & variance)


def mask_digest(mask):
    a = np.asarray(mask, dtype=np.uint8, order='C')
    if a.ndim != 2 or not np.isin(a, [0, 1]).all():
        raise ValueError("Expected two-dimensional binary mask")
    payload = b'UMGS_PROXY_COMMON_MASK_V1\0' + struct.pack('<II', *a.shape) + a.tobytes(order='C')
    return hashlib.sha256(payload).hexdigest()


def complete_mean(values, expected):
    if len(values) != expected:
        raise ValueError("Target count mismatch during aggregation")
    count = sum(v is not None and np.isfinite(v) for v in values)
    return dict(value=float(np.mean(values)) if count == expected else None,
                valid_target_count=int(count), expected_target_count=expected,
                complete=count == expected)


def load_packet(record, keys):
    p = Path(record['path'])
    verify_sha(p, record['sha256'])
    with np.load(p, allow_pickle=False) as archive:
        if set(archive.files) != set(keys):
            raise ValueError("Packet keyset mismatch: " + str(p))
        return {name: archive[name] for name in archive.files}


def unique_rows(rows, key, expected):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows) or set(result) != set(expected):
        raise ValueError("Missing, extra or duplicate target")
    return result


def score(args):
    if args.output.exists():
        raise ValueError("Output exists")
    verify_sha(args.job, args.job_sha256)
    job = json.loads(args.job.read_text(encoding='utf-8'))
    if job['schema'] != 'umgs_proxy_common_scoring_job_v1':
        raise ValueError("Unknown scoring job")
    methods = CORE_METHODS if job['scene'] == 'road' else PRIMARY_METHODS
    if tuple(job['methods']) != methods:
        raise ValueError("Cannot shrink or reorder predeclared comparison set")
    wrapper_path, registry_path = Path(job['wrapper']), Path(job['registry'])
    verify_sha(wrapper_path, WRAPPER_SHA)
    verify_sha(registry_path, REGISTRY_SHA)
    spec = importlib.util.spec_from_file_location('frozen_proxy_scoring_wrapper', wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    modules = wrapper.audited_modules()
    gradient = importlib.import_module('tools.depth_reference_geometry_v2.openmvs_da3_overlap_corrected')
    scene = wrapper.scene_record(wrapper.load_registry(registry_path), job['scene'])
    targets = scene['targets']
    names = [t['image_name'] for t in targets]
    if len(set(names)) != scene['target_count'] or names != job['target_names']:
        raise ValueError("Frozen target identity mismatch")
    manifest_records = {}
    maps = {}
    for method in methods:
        if method in ('umgs', 'rgb_anchor'):
            continue
        record = job['method_manifests'][method]
        path = Path(record['path']); verify_sha(path, record['sha256'])
        m = json.loads(path.read_text())
        if m['scene'] != job['scene'] or m['target_count'] != len(names):
            raise ValueError("Export scene/count mismatch")
        if method == 'jo':
            if (m['schema'] != 'mmsplat_common_sfm_self5_method_scene_manifest_v1'
                    or m['geometry']['checkpoint_step'] != 119999 or not m['source_unchanged']
                    or m['method_specific_alignment_used'] or m['scale_shift_fit_used'] or m['sim3_used']):
                raise ValueError("Historical JO contract mismatch")
            maps[method] = unique_rows(m['packets'], 'target', names)
        else:
            if (m['schema'] != 'umgs_common_graphdeco_proxy_fresh_v2_v1' or m['method_id'] != method
                    or m['registry_sha256'] != REGISTRY_SHA or not m['same_render_call_real_H']
                    or m['method_specific_alignment'] or not m['legacy_six_arrays_and_numeric_valid_unchanged']):
                raise ValueError("Fresh proxy export contract mismatch")
            maps[method] = unique_rows(m['packets'], 'image_name', names)
        manifest_records[method] = dict(path=str(path), sha256=record['sha256'])
    args.output.mkdir(parents=True)
    (args.output / 'masks').mkdir()
    details, rows = [], []
    for target in targets:
        shape = (target['height'], target['width'])
        ref = load_packet(target['reference_packet'], {'depth', 'valid', 'triangle_id', 'barycentric'})
        depth, valid = ref['depth'], ref['valid'].astype(bool)
        if depth.shape != shape or valid.shape != shape or not valid.any() or not (np.isfinite(depth[valid]) & (depth[valid] > 0)).all():
            raise ValueError("Invalid frozen reference")
        records = {'umgs': target['umgs_full_packet']}
        packet_manifest = Path(records['umgs']['path']).parent / 'PACKET_MANIFEST.json'
        if 'rgb_anchor' in methods:
            verify_sha(packet_manifest, job['rgb_anchor_manifests'][target['image_name']])
            anchor = json.loads(packet_manifest.read_text())
            if (anchor['method_kind'] != 'rgb-anchor' or anchor['model_identity']['kind'] != 'rgb_anchor_checkpoint'
                    or anchor['target'] != target['image_name'] or anchor['packet']['sha256'] != records['umgs']['sha256']
                    or anchor['camera_contract']['payload_sha256'] != target['fingerprint_sha256']):
                raise ValueError("RGB/UMGS common anchor artifact identity not proven")
            records['rgb_anchor'] = records['umgs']
        for method, mapping in maps.items():
            p = mapping[target['image_name']]
            if p['camera']['fingerprint_sha256'] != target['fingerprint_sha256'] or (p['camera']['height'], p['camera']['width']) != shape:
                raise ValueError("Common renderer camera fingerprint mismatch")
            parent = Path(manifest_records[method]['path']).parent
            if method == 'jo':
                records[method] = dict(path=str(parent / p['packet_file']), sha256=p['packet_sha256'])
            else:
                if not p['packet_ref']['passed'] or not p['legacy_raw_accumulator_bitwise_parity']:
                    raise ValueError("Same-call packet/ref gate failed")
                records[method] = dict(path=str(parent / p['legacy']['path']), sha256=p['legacy']['sha256'])
        packets = {method: load_packet(records[method], LEGACY_NAMES) for method in methods}
        gates = {method: wrapper.m1_a_gate(modules, packets[method], shape) for method in methods}
        masks = common_masks(depth, valid, packets, methods)
        detail = dict(target=target['image_name'], reference=target['reference_packet'], packets=records,
                      packet_gates=gates, fingerprint_sha256=target['fingerprint_sha256'], masks={})
        for mask_name, mask in masks.items():
            domain = gradient.reference_high_gradient_domain(depth, mask)
            stem = target['normalized_stem'] + '__' + mask_name
            artifacts = {}
            for suffix, value in [('mask', mask), ('reference_gradient_domain', domain.high_mask)]:
                p = args.output / 'masks' / (stem + '__' + suffix + '.npy')
                with p.open('xb') as f: np.save(f, value.astype(np.uint8), allow_pickle=False)
                artifacts[suffix] = dict(path=p.relative_to(args.output).as_posix(), sha256=sha256(p), matrix_sha256=mask_digest(value))
            common = dict(support_pixels=int(mask.sum()), support_coverage=float(mask.mean()), mask_sha256=mask_digest(mask),
                          reference_gradient_mask_sha256=mask_digest(domain.high_mask), reference_gradient_threshold=domain.threshold)
            results = {}
            for method in methods:
                metric = modules.packets._candidate_metrics(depth, packets[method]['expected_camera_z'], mask, domain) if mask.any() else {k: None for k in METRICS}
                for key, value in list(metric.items()):
                    if isinstance(value, (float, np.floating)) and not np.isfinite(value): metric[key] = None
                results[method] = metric
                rows.append(dict(scene=job['scene'], target=target['image_name'], method=method, mask=mask_name, **common,
                                 **{k: metric.get(k) for k in METRICS}))
            detail['masks'][mask_name] = dict(**common, artifacts=artifacts, methods=results)
        details.append(detail)
        print(json.dumps(dict(completed=len(details), total=len(targets), target=target['image_name'])), flush=True)
    aggregates = {}
    for mask_name in ('common_primary', 'common_opacity_ge_0_5', 'common_variance_valid'):
        aggregates[mask_name] = {}
        for method in methods:
            selected = [r for r in rows if r['mask'] == mask_name and r['method'] == method]
            aggregates[mask_name][method] = {k: complete_mean([r[k] for r in selected], len(targets)) for k in METRICS}
    with (args.output / 'per_target_metrics.csv').open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(schema='umgs_multi_method_common_proxy_scoring_v1', scene=job['scene'], method_set=list(methods),
                  target_count=len(targets), targets=details, aggregate=aggregates, gpu_used=False,
                  job_sha256=args.job_sha256, wrapper_sha256=WRAPPER_SHA, registry_sha256=REGISTRY_SHA,
                  method_manifests=manifest_records, script_sha256=sha256(Path(__file__)),
                  packet_contract='frozen_legacy_six_array_expected_camera_z_M1_over_A',
                  new_packet_v2_is_separate_not_a_mask_override=True, method_specific_alignment=False,
                  target_weighting='equal_no_successful_subset_mean', scene_weighting='equal_within_declared_scene_set',
                  rgb_anchor_geometry_shared='same_registered_RGB_checkpoint_artifact' if 'rgb_anchor' in methods else None,
                  depth_units='source_common_SfM_reconstruction_units_not_metres',
                  claim_boundary='OpenMVS engineering proxy, not absolute geometry ground truth',
                  whole_matrix_row_complete=False, stage_review='PENDING')
    result['all_declared_metric_values_complete'] = all(v['complete'] for a in aggregates.values() for b in a.values() for v in b.values())
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--job-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    score(p.parse_args())


if __name__ == '__main__':
    main()
