"""CPU-only campaign preparation, result accounting and closeout decisions.

This module never launches a child, opens a server connection or powers off a
machine. Decisions are consumed by the supervised execution workflow; they do
not replace live GPU/protocol qualification or external review.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from .contracts import read_json, record_hash, safe_member, verify_sha

SCENES = (
    "road_01_20260602_1648_40m", "gcp_5000_20260602",
    "eucalyptus_01_20260526_1053_pruned", "maize_02_20260526_1658",
    "cassava_01_20260526_1603", "papaya_01_20251217",
)
CORE = SCENES[:2]
PAIRED_METHODS = ("umgs", "jo", "ms_splatting_neural")
CORE_METHODS = ("rgb_anchor", "sig_mechanism")
METRICS = {
    "gcp": ("rmse_h_m", "rmse_z_m", "rmse_3d_m", "checkpoint_coverage", "observation_coverage"),
    "lidar": ("precision_5cm", "recall_5cm", "fscore_5cm", "precision_10cm", "recall_10cm", "fscore_10cm",
              "precision_20cm", "recall_20cm", "fscore_20cm"),
    "proxy": ("abs_rel", "rmse", "coverage"),
    "appearance": ("rgb_psnr", "rgb_ssim", "rgb_lpips", "spectral_sam"),
    "resources": ("training_seconds", "gpu_hours", "peak_vram_bytes", "model_bytes"),
}
REQUIRED_GATES = (
    "input_identity", "split_and_shared_sfm", "method_recipe", "environment",
    "camera_ray_and_normalization", "metric_packet_and_reference", "gcp_lidar_protocol",
)
RECEIPT_STATUSES = {"COMPLETE", "PARTIAL", "FAILED", "BLOCKED"}


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", value):
        raise ValueError("Invalid campaign identity")
    return value


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Explicit lowercase SHA-256 required")
    return value


def expected_rows():
    rows = []
    for scene in SCENES:
        for method in (*PAIRED_METHODS, *(CORE_METHODS if scene in CORE else ())):
            rows.append({"row_id": f"{scene}/{method}", "scene": scene, "method": method,
                         "track": "triple" if scene in CORE else "proxy",
                         "evaluation_groups": [*(["gcp", "lidar"] if scene in CORE else []),
                                               "proxy", "appearance", "resources"]})
    return rows


def make_plan(campaign_id):
    _id(campaign_id)
    pipelines = [{"scene": CORE[1], "method": "umgs"}, {"scene": CORE[1], "method": "jo"}]
    pipelines.extend({"scene": s, "method": "sig_mechanism"} for s in CORE)
    pipelines.extend({"scene": s, "method": "ms_splatting_neural"} for s in SCENES)
    for p in pipelines:
        p["pipeline_id"] = f"{p['scene']}/{p['method']}"
        p["command_binding"] = "PENDING_RECIPE_AND_RUNTIME_QUALIFICATION"
    plan = {"schema": "umgs_tgrs_campaign_preparation_v1", "campaign_id": campaign_id,
            "authorization_stage": "PREPARATION_ONLY", "gpu_user_notification_received": False,
            "new_training_pipelines": pipelines, "result_rows": expected_rows(),
            "required_gates": list(REQUIRED_GATES),
            "gate_status": {g: "PENDING" for g in REQUIRED_GATES},
            "execution_order": "existing_3k_evaluation_qualification_then_approved_dependencies",
            "scene_aggregate_policy": "six_scene_pairs_separate_from_two_scene_mechanism_results",
            "shutdown_policy": {"server": "901", "on_batch_complete": True,
                                "on_review_failed": True, "on_review_tool_error": True,
                                "preparation_only_shutdown": False, "foreign_tasks_must_be_absent": True,
                                "saved_evidence_and_quiescence_required": True},
            "storage_policy": "capacity_user_managed_but_never_write_until_disk_full",
            "training_ready": False, "external_review_status": "NOT_REQUESTED",
            "boundaries": ["no_gpu_until_explicit_user_notification", "no_other_project_mutations",
                           "no_scientific_protocol_changes", "no_result_dependent_tuning",
                           "no_overwrite", "no_power_operation_during_preparation"]}
    return {"plan": plan, "plan_sha256": record_hash(plan)}


def validate_plan(envelope):
    plan = envelope["plan"]
    if record_hash(plan) != _hash(envelope["plan_sha256"]):
        raise ValueError("Campaign plan hash mismatch")
    expected = make_plan(plan["campaign_id"])["plan"]
    if plan != expected:
        raise ValueError("Preparation contract was modified; runtime approval is a separate record")
    return plan


def gpu_start_decision(envelope, authorization, gate_evidence, resource_snapshot, *, now):
    plan = validate_plan(envelope)
    reasons = []
    if (authorization.get("campaign_id") != plan["campaign_id"]
            or authorization.get("plan_sha256") != envelope["plan_sha256"]
            or authorization.get("explicit_user_gpu_available") is not True):
        reasons.append("explicit_matching_user_gpu_notification_missing")
    try:
        _hash(authorization.get("user_message_evidence_sha256"))
    except ValueError:
        reasons.append("user_notification_evidence_missing")
    for gate in REQUIRED_GATES:
        evidence = gate_evidence.get(gate, {})
        if evidence.get("status") != "PASS":
            reasons.append(f"gate_not_passed:{gate}")
        try:
            _hash(evidence.get("verified_report_sha256"))
        except ValueError:
            reasons.append(f"gate_report_unbound:{gate}")
    reasons.extend(_resource_reasons(resource_snapshot, now, gpu_start=True))
    return {"allowed": not reasons, "reasons": reasons, "child_launched": False}


def _resource_reasons(snapshot, now, *, gpu_start):
    reasons = []
    stamp = snapshot.get("observed_at_unix")
    if (type(stamp) not in (int, float) or not math.isfinite(stamp)
            or not 0 <= now - stamp <= 30):
        reasons.append("live_snapshot_missing_stale_or_future")
    if snapshot.get("server") != "901":
        reasons.append("wrong_server")
    if snapshot.get("process_ownership_verified") is not True:
        reasons.append("process_ownership_unknown")
    for field in ("foreign_jobs", "unknown_jobs", "own_active_jobs", "active_transfers"):
        if type(snapshot.get(field)) is not int or snapshot[field] != 0:
            reasons.append(f"not_quiescent:{field}")
    if gpu_start and snapshot.get("three_idle_samples_passed") is not True:
        reasons.append("gpu_idle_gate_missing")
    return reasons


def closeout_decision(envelope, batch, snapshot, *, now):
    plan = validate_plan(envelope)
    reasons = []
    if (batch.get("campaign_id") != plan["campaign_id"]
            or batch.get("plan_sha256") != envelope["plan_sha256"]):
        reasons.append("batch_identity_mismatch")
    try:
        _hash(batch.get("user_shutdown_authorization_evidence_sha256"))
    except ValueError:
        reasons.append("shutdown_authorization_evidence_missing")
    if batch.get("experiments_started_after_user_notification") is not True:
        reasons.append("preparation_only_or_unapproved_start")
    if batch.get("outcome") not in {"COMPLETED", "REVIEW_FAILED", "REVIEW_TOOL_ERROR"}:
        reasons.append("batch_not_at_authorized_shutdown_boundary")
    for field in ("all_owned_children_stopped", "logs_flushed", "evidence_saved_off_server",
                  "artifact_inventory_verified", "explicit_batch_shutdown_authorization"):
        if batch.get(field) is not True:
            reasons.append(f"closeout_incomplete:{field}")
    reasons.extend(_resource_reasons(snapshot, now, gpu_start=False))
    return {"decision": "SHUTDOWN_ELIGIBLE_RECHECK_BEFORE_COMMAND" if not reasons else "DO_NOT_SHUTDOWN",
            "reasons": reasons, "review_failure_does_not_become_pass": True,
            "power_command_executed": False}


def _artifact(root, spec):
    path = safe_member(root, spec["path"])
    verify_sha(path, _hash(spec["sha256"]))
    if type(spec["size_bytes"]) is not int or path.stat().st_size != spec["size_bytes"]:
        raise ValueError("Evidence artifact size mismatch")
    return path


def validate_receipt(receipt, expected, root, campaign_id):
    if receipt.get("schema") != "umgs_tgrs_result_receipt_v1" or receipt.get("campaign_id") != campaign_id:
        raise ValueError("Result receipt schema or campaign mismatch")
    if any(receipt.get(k) != expected[k] for k in ("row_id", "scene", "method", "track")):
        raise ValueError("Result identity mismatch")
    if receipt.get("status") not in RECEIPT_STATUSES:
        raise ValueError("Unknown result status")
    if set(receipt.get("groups", {})) != set(expected["evaluation_groups"]):
        raise ValueError("Missing or unexpected evaluation group")
    for group, value in receipt["groups"].items():
        if value not in {"PASS", "PARTIAL", "FAILED", "NOT_RUN"}:
            raise ValueError(f"Unknown group status: {group}")
    metrics = receipt.get("metrics", {})
    allowed = {f"{g}.{m}" for g in expected["evaluation_groups"] for m in METRICS[g]}
    if not set(metrics) <= allowed:
        raise ValueError("Unknown metric or unexpected evaluation track")
    for key, value in metrics.items():
        if (value is not None and (type(value) not in (int, float) or not math.isfinite(value))):
            raise ValueError("Numeric metric must be finite, not a boolean/string")
        if value is not None and receipt["groups"][key.split(".")[0]] not in {"PASS", "PARTIAL"}:
            raise ValueError("Failed/unrun group cannot publish a numeric metric")
        if expected["method"] == "rgb_anchor" and key == "appearance.spectral_sam" and value is not None:
            raise ValueError("RGB-only anchor has no spectral result")
    if not receipt.get("reason") and receipt["status"] != "COMPLETE":
        raise ValueError("Non-complete result requires a reason")
    if not receipt.get("artifacts"):
        raise ValueError("Result must bind evidence, including failures")
    paths = []
    for spec in receipt["artifacts"]:
        paths.append(str(_artifact(root, spec)))
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate result evidence")
    if receipt["status"] == "COMPLETE":
        if any(s != "PASS" for s in receipt["groups"].values()):
            raise ValueError("Complete result has incomplete groups")
        required_metrics = allowed - ({"appearance.spectral_sam"} if expected["method"] == "rgb_anchor" else set())
        if any(metrics.get(key) is None for key in required_metrics):
            raise ValueError("Complete result is missing required summary metrics")
        identity = receipt.get("identity", {})
        for key in ("input_manifest_sha256", "method_recipe_sha256", "scoring_protocol_sha256", "checkpoint_sha256"):
            _hash(identity.get(key))
        audit_path = _artifact(root, receipt["independent_audit"])
        if str(audit_path) not in paths:
            raise ValueError("Independent audit must be in result evidence")
        audit = read_json(audit_path)
        expected_metric_hash = record_hash({"identity": identity, "groups": receipt["groups"], "metrics": metrics,
                                           "coverage_status": receipt.get("coverage_status")})
        if (audit.get("status") != "PASS" or audit.get("row_id") != expected["row_id"]
                or audit.get("campaign_id") != campaign_id or audit.get("metric_record_sha256") != expected_metric_hash):
            raise ValueError("Independent audit does not bind these exact metrics and identities")
        # Coverage completeness is an explicit audited outcome, not inferred from RMSE.
        if receipt.get("coverage_status") not in {"COMPLETE", "INCOMPLETE_VALID_RESULT"}:
            raise ValueError("Explicit scientific coverage disposition required")
    return receipt


def collect_results(envelope, index, root):
    plan = validate_plan(envelope)
    if (index.get("schema") != "umgs_tgrs_result_index_v1" or index.get("campaign_id") != plan["campaign_id"]
            or index.get("plan_sha256") != envelope["plan_sha256"]):
        raise ValueError("Result index does not match campaign")
    expected = {r["row_id"]: r for r in plan["result_rows"]}
    receipts = {}
    for entry in index["receipts"]:
        row_id = entry["row_id"]
        if row_id not in expected or row_id in receipts:
            raise ValueError("Unknown/duplicate result row")
        path = _artifact(root, entry)
        receipt = read_json(path)
        receipts[row_id] = validate_receipt(receipt, expected[row_id], root, plan["campaign_id"])
    rows = []
    for exp in expected.values():
        row = {**exp, "status": "PENDING", "coverage_status": "NOT_EVALUATED",
               "reason": "no_bound_result_receipt", "metrics": {}}
        if exp["row_id"] in receipts:
            row.update(receipts[exp["row_id"]])
            row["reason"] = receipts[exp["row_id"]].get("reason", "")
        rows.append(row)
    return {"schema": "umgs_tgrs_total_results_v1", "campaign_id": plan["campaign_id"],
            "plan_sha256": envelope["plan_sha256"], "row_count": len(rows), "rows": rows,
            "external_review_status": "NOT_ASSERTED_BY_COLLECTOR",
            "numeric_results_present": any(v is not None for r in rows for v in r["metrics"].values()),
            "all_required_rows_complete": all(r["status"] == "COMPLETE" for r in rows),
            "aggregation": "not_performed_no_mixing_two_scene_and_six_scene_means"}


def write_json_exclusive(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_table(path, summary):
    columns = ["scene", "method", "track", "status", "coverage_status", "reason"]
    columns.extend(f"{g}.{m}" for g in METRICS for m in METRICS[g])
    with Path(path).open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\r\n")
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow({**{c: row[c] for c in columns[:6]}, **row["metrics"]})


def main():
    from .cpu_guard import install_cpu_guard
    install_cpu_guard()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--campaign_id", required=True)
    prepare.add_argument("--output_dir", type=Path, required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--index", type=Path, required=True)
    collect.add_argument("--evidence_root", type=Path, required=True)
    collect.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        plan = make_plan(args.campaign_id)
        index = {"schema": "umgs_tgrs_result_index_v1", "campaign_id": args.campaign_id,
                 "plan_sha256": plan["plan_sha256"], "receipts": []}
        summary = collect_results(plan, index, args.output_dir)
    else:
        plan, index = read_json(args.plan), read_json(args.index)
        summary = collect_results(plan, index, args.evidence_root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(args.output_dir / "campaign_plan.json", plan)
    write_json_exclusive(args.output_dir / "result_index.json", index)
    write_json_exclusive(args.output_dir / "total_results.json", summary)
    write_table(args.output_dir / "total_results.csv", summary)
    print(json.dumps({"output_dir": str(args.output_dir), "row_count": summary["row_count"],
                      "all_required_rows_complete": summary["all_required_rows_complete"],
                      "gpu_started": False, "power_command_executed": False}))


if __name__ == "__main__":
    raise SystemExit(main())
