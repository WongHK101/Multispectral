"""Campaign bookkeeping tests, using synthetic receipts only."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from .campaign import (CORE, METRICS, REQUIRED_GATES, closeout_decision, collect_results,
                       expected_rows, gpu_start_decision, macro_eligibility, make_plan,
                       metric_record, ranking_record, validate_plan, write_table,
                       QUALIFICATION_CPU_GATES, qualification_start_decision)
from .contracts import canonical_bytes, record_hash, sha256


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.plan = make_plan("umgs_tgrs_test")

    def snapshot(self):
        return {"server": "901", "observed_at_unix": 100, "process_ownership_verified": True,
                "foreign_jobs": 0, "unknown_jobs": 0, "own_active_jobs": 0,
                "active_transfers": 0, "three_idle_samples_passed": True}

    def test_plan_is_deterministic(self):
        self.assertEqual(canonical_bytes(self.plan), canonical_bytes(make_plan("umgs_tgrs_test")))

    def test_exact_ten_new_pipelines(self):
        rows = self.plan["plan"]["new_training_pipelines"]
        self.assertEqual(len(rows), 10)
        self.assertEqual(len({r["pipeline_id"] for r in rows}), 10)
        self.assertEqual(sum(r["method"] == "ms_splatting_neural" for r in rows), 6)
        self.assertEqual(sum(r["method"] == "sig_mechanism" for r in rows), 2)

    def test_result_matrix_22_with_separate_sig_scope(self):
        rows = expected_rows()
        self.assertEqual(len(rows), 22)
        self.assertEqual({r["scene"] for r in rows if r["method"] == "sig_mechanism"}, set(CORE))
        self.assertEqual(sum(r["track"] == "triple" for r in rows), 10)

    def test_no_external_aerial_scenes(self):
        self.assertFalse(any("aerial" in r["scene"] for r in expected_rows()))

    def test_unknown_campaign_name(self):
        with self.assertRaises(ValueError):
            make_plan("../other")

    def test_tampered_plan_rejected(self):
        self.plan["plan"]["training_ready"] = True
        with self.assertRaises(ValueError):
            validate_plan(self.plan)

    def test_rehashed_changed_scope_still_rejected(self):
        self.plan["plan"]["new_training_pipelines"].pop()
        self.plan["plan_sha256"] = record_hash(self.plan["plan"])
        with self.assertRaises(ValueError):
            validate_plan(self.plan)

    def authorization(self):
        return {"campaign_id": "umgs_tgrs_test", "plan_sha256": self.plan["plan_sha256"],
                "explicit_user_gpu_available": True, "user_message_evidence_sha256": "a" * 64}

    def gates(self):
        return {g: {"status": "PASS", "verified_report_sha256": "b" * 64} for g in REQUIRED_GATES}

    def test_gpu_waits_for_explicit_user(self):
        self.assertFalse(gpu_start_decision(self.plan, {}, self.gates(), self.snapshot(), now=100)["allowed"])

    def test_preparation_never_implicitly_approves_gates(self):
        self.assertFalse(gpu_start_decision(self.plan, self.authorization(), {}, self.snapshot(), now=100)["allowed"])

    def test_all_bound_gates_and_live_user_notice_decision_only(self):
        out = gpu_start_decision(self.plan, self.authorization(), self.gates(), self.snapshot(), now=100)
        self.assertTrue(out["allowed"])
        self.assertFalse(out["child_launched"])

    def test_foreign_gpu_or_cpu_job_prevents_start(self):
        s = self.snapshot(); s["foreign_jobs"] = 1
        self.assertFalse(gpu_start_decision(self.plan, self.authorization(), self.gates(), s, now=100)["allowed"])

    def test_each_missing_gate_blocks(self):
        for key in REQUIRED_GATES:
            with self.subTest(key=key):
                g = self.gates(); g.pop(key)
                self.assertFalse(gpu_start_decision(self.plan, self.authorization(), g, self.snapshot(), now=100)["allowed"])

    def test_unknown_or_stale_resources_block(self):
        for stamp in (None, -1, 101, float("nan")):
            with self.subTest(stamp=stamp):
                s = self.snapshot(); s["observed_at_unix"] = stamp
                self.assertFalse(gpu_start_decision(self.plan, self.authorization(), self.gates(), s, now=100)["allowed"])

    def qualification_request(self):
        return {"scene": CORE[0], "method": "ms_splatting_neural", "stage": "GPU_QUALIFICATION",
                "operation": "kernel_packet_parity", "output_class": "nonformal_qualification_only",
                "max_gpu_seconds": 900, "max_iterations": 0, "gpu_budget_seconds_remaining": 86400}

    def qualification_cpu(self):
        return {g: {"status": "PASS", "verified_report_sha256": "b" * 64} for g in QUALIFICATION_CPU_GATES}

    def test_short_qualification_not_circular_or_full_run_permission(self):
        out = qualification_start_decision(self.plan, self.authorization(), self.qualification_cpu(),
                                            self.snapshot(), self.qualification_request(), now=100)
        self.assertTrue(out["allowed"])
        self.assertFalse(out["full_experiment_authorized"])
        self.assertFalse(out["gpu_packet_qualification_passed"])
        self.assertFalse(gpu_start_decision(self.plan, self.authorization(), self.qualification_cpu(),
                                           self.snapshot(), now=100)["allowed"])

    def test_qualification_scope_budget_and_limits_rejected(self):
        for key, value in (("scene", CORE[1]), ("operation", "formal_training"),
                           ("output_class", "formal"), ("max_iterations", 120000),
                           ("max_gpu_seconds", 1801), ("gpu_budget_seconds_remaining", 0),
                           ("gpu_budget_seconds_remaining", float("nan"))):
            req = {**self.qualification_request(), key: value}
            with self.subTest(key=key):
                self.assertFalse(qualification_start_decision(self.plan, self.authorization(),
                    self.qualification_cpu(), self.snapshot(), req, now=100)["allowed"])

    def test_qualification_waits_for_user_and_fresh_cpu_evidence(self):
        for auth, gates in (({}, self.qualification_cpu()), (self.authorization(), {})):
            self.assertFalse(qualification_start_decision(self.plan, auth, gates, self.snapshot(),
                self.qualification_request(), now=100)["allowed"])
        snapshot = self.snapshot(); snapshot["foreign_jobs"] = 1
        self.assertFalse(qualification_start_decision(self.plan, self.authorization(), self.qualification_cpu(),
            snapshot, self.qualification_request(), now=100)["allowed"])

    def batch(self, outcome="COMPLETED"):
        return {"campaign_id": "umgs_tgrs_test", "plan_sha256": self.plan["plan_sha256"],
                "user_shutdown_authorization_evidence_sha256": "d" * 64,
                "stage": "EXPERIMENT_BATCH", "gpu_authorization": self.authorization(),
                "experiments_started_after_user_notification": True,
                "reason": "" if outcome == "COMPLETED" else "synthetic_review_failure",
                "outcome": outcome, "all_owned_children_stopped": True, "logs_flushed": True,
                "evidence_saved_off_server": True, "artifact_inventory_verified": True,
                "explicit_batch_shutdown_authorization": True}

    def test_completed_or_review_failure_all_allow_safe_closeout(self):
        for outcome in ("COMPLETED", "BLOCKED", "REVIEW_FAILED", "REVIEW_TOOL_ERROR"):
            with self.subTest(outcome=outcome):
                d = closeout_decision(self.plan, self.batch(outcome), self.snapshot(), now=100)
                self.assertEqual(d["decision"], "SHUTDOWN_ELIGIBLE_RECHECK_BEFORE_COMMAND")
                self.assertFalse(d["power_command_executed"])
                self.assertTrue(d["review_failure_does_not_become_pass"])

    def test_preparation_must_not_shutdown(self):
        b = self.batch(); b.update(stage="CPU_PREPARATION", experiments_started_after_user_notification=False)
        self.assertEqual(closeout_decision(self.plan, b, self.snapshot(), now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_user_enabled_gpu_blocked_before_training_can_closeout(self):
        b = self.batch("BLOCKED")
        b.update(stage="GPU_QUALIFICATION", experiments_started_after_user_notification=False,
                 reason="kernel_qualification_failed")
        out = closeout_decision(self.plan, b, self.snapshot(), now=100)
        self.assertEqual(out["decision"], "SHUTDOWN_ELIGIBLE_RECHECK_BEFORE_COMMAND")
        self.assertFalse(out["power_command_executed"])

    def test_qualification_does_not_claim_success_or_training(self):
        for outcome, started in (("COMPLETED", False), ("BLOCKED", True)):
            b = self.batch(outcome)
            b.update(stage="GPU_QUALIFICATION", experiments_started_after_user_notification=started)
            self.assertEqual(closeout_decision(self.plan, b, self.snapshot(), now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_closeout_does_not_infer_user_gpu_authorization(self):
        for auth in ({}, {**self.authorization(), "explicit_user_gpu_available": False},
                     {**self.authorization(), "plan_sha256": "f" * 64},
                     {**self.authorization(), "user_message_evidence_sha256": None}):
            b = self.batch(); b["gpu_authorization"] = auth
            self.assertEqual(closeout_decision(self.plan, b, self.snapshot(), now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_blocked_closeout_requires_real_reason(self):
        b = self.batch("BLOCKED"); b["reason"] = ""
        self.assertEqual(closeout_decision(self.plan, b, self.snapshot(), now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_shutdown_authorization_must_bind_campaign_and_message(self):
        for field, value in (("plan_sha256", "e" * 64),
                             ("user_shutdown_authorization_evidence_sha256", None)):
            with self.subTest(field=field):
                b = self.batch(); b[field] = value
                self.assertEqual(closeout_decision(self.plan, b, self.snapshot(), now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_foreign_tasks_block_shutdown_even_after_review_error(self):
        s = self.snapshot(); s["foreign_jobs"] = 1
        self.assertEqual(closeout_decision(self.plan, self.batch("REVIEW_TOOL_ERROR"), s, now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_missing_backup_blocks_shutdown(self):
        b = self.batch(); b["evidence_saved_off_server"] = False
        self.assertEqual(closeout_decision(self.plan, b, self.snapshot(), now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_each_incomplete_closeout_gate_blocks(self):
        for key in ("all_owned_children_stopped", "logs_flushed", "artifact_inventory_verified", "explicit_batch_shutdown_authorization"):
            with self.subTest(key=key):
                b = self.batch(); b[key] = False
                self.assertEqual(closeout_decision(self.plan, b, self.snapshot(), now=100)["decision"], "DO_NOT_SHUTDOWN")

    def test_wrong_server_unknown_processes_and_transfer_block_shutdown(self):
        for key, value in (("server", "740"), ("unknown_jobs", 1), ("active_transfers", 1), ("process_ownership_verified", False)):
            with self.subTest(key=key):
                s = self.snapshot(); s[key] = value
                self.assertEqual(closeout_decision(self.plan, self.batch(), s, now=100)["decision"], "DO_NOT_SHUTDOWN")


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = make_plan("umgs_tgrs_test")
        self.index = {"schema": "umgs_tgrs_result_index_v1", "campaign_id": "umgs_tgrs_test",
                      "plan_sha256": self.plan["plan_sha256"], "receipts": []}
        self.exp = expected_rows()[0]

    def save(self, name, value):
        path = self.root / name
        path.write_bytes(canonical_bytes(value))
        return {"path": name, "sha256": sha256(path), "size_bytes": path.stat().st_size}

    def receipt(self):
        r = {"schema": "umgs_tgrs_result_receipt_v1", "campaign_id": "umgs_tgrs_test", **self.exp,
             "status": "COMPLETE", "coverage_status": "COMPLETE", "reason": "",
             "identity": {k: "c"*64 for k in ("input_manifest_sha256", "method_recipe_sha256", "scoring_protocol_sha256", "checkpoint_sha256")},
             "groups": {g: "PASS" for g in self.exp["evaluation_groups"]},
             "metrics": {f"{g}.{m}": 1.0 for g in self.exp["evaluation_groups"] for m in METRICS[g]}}
        audit = {"status": "PASS", "row_id": r["row_id"], "campaign_id": r["campaign_id"],
                 "metric_record_sha256": record_hash({k: r[k] for k in ("identity", "groups", "metrics", "coverage_status")})}
        spec = self.save("audit.json", audit)
        r["artifacts"] = [spec]; r["independent_audit"] = spec
        return r

    def collect(self, r):
        spec = self.save("receipt.json", r)
        self.index["receipts"] = [{**spec, "row_id": self.exp["row_id"]}]
        return collect_results(self.plan, self.index, self.root)

    def reaudit(self, r):
        audit = {"status": "PASS", "row_id": r["row_id"], "campaign_id": r["campaign_id"],
                 "metric_record_sha256": record_hash(metric_record(r))}
        spec = self.save("audit.json", audit)
        r["artifacts"] = [spec]; r["independent_audit"] = spec
        return r

    def v2_receipt(self):
        r = self.receipt()
        r["schema"] = "umgs_tgrs_result_receipt_v2"
        r["ranking"] = {g: ranking_record(4, 4, "f" * 64) for g in r["groups"]}
        return self.reaudit(r)

    def test_legacy_complete_is_not_automatically_ranked(self):
        row = self.collect(self.receipt())["rows"][0]
        self.assertEqual(row["status"], "COMPLETE")
        self.assertFalse(any(row["metric_ranking_eligible"].values()))

    def test_v2_complete_and_audited_ranking(self):
        row = self.collect(self.v2_receipt())["rows"][0]
        self.assertTrue(row["metric_ranking_eligible"]["gcp.rmse_3d_m"])

    def test_complete_delivery_incomplete_gcp_stays_unranked(self):
        r = self.v2_receipt()
        r["coverage_status"] = "INCOMPLETE_VALID_RESULT"
        r["ranking"]["gcp"] = ranking_record(4, 3, "f" * 64, reason="formal_checkpoint_coverage_incomplete")
        row = self.collect(self.reaudit(r))["rows"][0]
        self.assertEqual(row["status"], "COMPLETE")
        self.assertEqual(row["ranking"]["gcp"]["status"], "INCOMPLETE_UNRANKED")
        self.assertFalse(row["metric_ranking_eligible"]["gcp.rmse_3d_m"])
        self.assertTrue(row["metric_ranking_eligible"]["appearance.rgb_psnr"])

    def test_missing_resources_do_not_remove_verified_gcp_eligibility(self):
        r = self.v2_receipt()
        r.update(status="PARTIAL", reason="historical_resource_measurements_missing")
        r["groups"]["resources"] = "PARTIAL"
        r["ranking"]["resources"] = ranking_record(4, 0, "f" * 64, reason=r["reason"])
        r["metrics"] = {k: v for k, v in r["metrics"].items() if not k.startswith("resources.")}
        row = self.collect(self.reaudit(r))["rows"][0]
        self.assertTrue(row["metric_ranking_eligible"]["gcp.rmse_3d_m"])
        self.assertFalse(row["metric_ranking_eligible"]["resources.training_seconds"])

    def test_ranking_tamper_and_legacy_rank_injection_rejected(self):
        r = self.v2_receipt()
        r["ranking"]["gcp"] = ranking_record(4, 3, "f" * 64, reason="incomplete")
        with self.assertRaises(ValueError):
            self.collect(r)
        r = self.receipt(); r["ranking"] = {}
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_false_complete_rank_rejected_even_if_reaudited(self):
        r = self.v2_receipt()
        r["ranking"]["gcp"]["population_passed"] = 3
        with self.assertRaises(ValueError):
            self.collect(self.reaudit(r))

    def test_partial_v2_metrics_still_need_reaudit(self):
        r = self.v2_receipt()
        r.update(status="PARTIAL", reason="partial_delivery")
        r["metrics"]["gcp.rmse_3d_m"] = 0.1
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_blocked_before_checkpoint_does_not_require_fake_identity(self):
        r = self.v2_receipt()
        r.update(status="BLOCKED", reason="input_binding_failed", identity={}, metrics={},
                 coverage_status="NOT_EVALUATED", groups={g: "NOT_RUN" for g in r["groups"]})
        r["ranking"] = {g: ranking_record(4, 0, "f" * 64, reason="not_evaluated") for g in r["groups"]}
        row = self.collect(self.reaudit(r))["rows"][0]
        self.assertFalse(any(row["metric_ranking_eligible"].values()))

    def test_not_evaluated_cannot_smuggle_ranked_numbers(self):
        r = self.v2_receipt(); r["coverage_status"] = "NOT_EVALUATED"
        with self.assertRaises(ValueError):
            self.collect(self.reaudit(r))

    def test_invalid_population_rejected(self):
        for total, passed in ((0, 0), (4, 5), (4, -1), (True, 1), (4, 3.0)):
            with self.subTest(total=total, passed=passed), self.assertRaises(ValueError):
                ranking_record(total, passed, "f" * 64, reason="invalid")

    def test_primary_macro_never_uses_surviving_scenes_only(self):
        out = self.collect(self.v2_receipt())
        self.assertFalse(macro_eligibility(out, "umgs", "gcp.rmse_3d_m")["eligible"])
        self.assertEqual(len(macro_eligibility(out, "umgs", "appearance.rgb_psnr")["required_scenes"]), 6)
        self.assertEqual(len(macro_eligibility(out, "sig_mechanism", "appearance.rgb_psnr")["required_scenes"]), 2)

    def test_macro_needs_each_frozen_scene_and_does_not_compute_mean(self):
        out = self.collect(self.v2_receipt())
        metric = "appearance.rgb_psnr"
        for row in out["rows"]:
            if row["method"] == "umgs":
                row["metric_ranking_eligible"][metric] = True
        decision = macro_eligibility(out, "umgs", metric)
        self.assertTrue(decision["eligible"])
        self.assertFalse(decision["mean_computed"])

    def test_empty_table_preserves_22_pending_not_zero_metrics(self):
        out = collect_results(self.plan, self.index, self.root)
        self.assertEqual(out["row_count"], 22)
        self.assertTrue(all(r["status"] == "PENDING" and not r["metrics"] for r in out["rows"]))
        self.assertFalse(out["all_required_rows_complete"])

    def test_valid_single_receipt_does_not_complete_campaign(self):
        out = self.collect(self.receipt())
        self.assertEqual(out["rows"][0]["status"], "COMPLETE")
        self.assertEqual(out["rows"][0]["reason"], "")
        self.assertFalse(out["all_required_rows_complete"])
        self.assertEqual(out["external_review_status"], "NOT_ASSERTED_BY_COLLECTOR")

    def test_duplicate_receipt_rejected(self):
        self.collect(self.receipt())
        self.index["receipts"].append(self.index["receipts"][0])
        with self.assertRaises(ValueError):
            collect_results(self.plan, self.index, self.root)

    def test_unknown_scene_cannot_enter_table(self):
        r = self.receipt(); r["scene"] = "aerial_golf"
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_joint_cannot_be_relabeled_as_neural(self):
        r = self.receipt(); r["method"] = "ms_splatting_neural"
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_nan_inf_boolean_metric_rejected(self):
        for value in (float("nan"), float("inf"), True, "1.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                r = self.receipt(); r["metrics"]["gcp.rmse_3d_m"] = value
                self.collect(r)

    def test_tampered_evidence_sha_rejected(self):
        self.collect(self.receipt())
        (self.root / "audit.json").write_text("{}")
        with self.assertRaises(ValueError):
            collect_results(self.plan, self.index, self.root)

    def test_changed_metrics_without_independent_reaudit_rejected(self):
        r = self.receipt(); r["metrics"]["gcp.rmse_3d_m"] = 0.123
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_changed_coverage_requires_independent_reaudit(self):
        r = self.receipt(); r["coverage_status"] = "INCOMPLETE_VALID_RESULT"
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_null_metrics_are_not_numeric_results(self):
        r = self.receipt()
        r.update(status="PARTIAL", reason="not_evaluated", metrics={"gcp.rmse_3d_m": None})
        self.assertFalse(self.collect(r)["numeric_results_present"])

    def test_rgb_anchor_cannot_publish_spectral_metric(self):
        self.exp = next(r for r in expected_rows() if r["method"] == "rgb_anchor")
        r = self.receipt()
        with self.assertRaises(ValueError):
            self.collect(r)
        r["metrics"].pop("appearance.spectral_sam")
        audit = {"status": "PASS", "row_id": r["row_id"], "campaign_id": r["campaign_id"],
                 "metric_record_sha256": record_hash({k: r[k] for k in ("identity", "groups", "metrics", "coverage_status")})}
        spec = self.save("audit.json", audit)
        r["artifacts"] = [spec]; r["independent_audit"] = spec
        out = self.collect(r)
        self.assertEqual(next(row for row in out["rows"] if row["row_id"] == r["row_id"])["status"], "COMPLETE")

    def test_complete_cannot_have_empty_metrics(self):
        r = self.receipt(); r["metrics"] = {}
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_group_failure_cannot_masquerade_as_complete(self):
        r = self.receipt(); r["groups"]["lidar"] = "FAILED"
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_failure_needs_bound_evidence_and_reason(self):
        r = self.receipt(); r.update(status="FAILED", metrics={}, reason="")
        with self.assertRaises(ValueError):
            self.collect(r)
        r["reason"] = "synthetic_failure"
        out = self.collect(r)
        self.assertEqual(out["rows"][0]["status"], "FAILED")

    def test_path_escape_rejected(self):
        r = self.receipt(); r["artifacts"][0]["path"] = "../audit.json"
        with self.assertRaises(ValueError):
            self.collect(r)

    def test_pending_csv_no_fabricated_zero(self):
        out = collect_results(self.plan, self.index, self.root)
        p = self.root / "table.csv"
        write_table(p, out)
        rows = p.read_text(encoding="utf-8-sig").splitlines()
        self.assertEqual(len(rows), 23)
        self.assertNotIn(",0.0,", rows[1])
        with self.assertRaises(FileExistsError):
            write_table(p, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
