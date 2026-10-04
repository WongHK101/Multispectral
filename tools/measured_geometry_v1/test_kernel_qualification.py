import json
import platform
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from . import kernel_qualification as kernel
from .campaign import CORE, make_plan
from .contracts import sha256


class QualificationExecutorTests(unittest.TestCase):
    def test_idle_samples_no_torch_and_no_kill(self):
        with patch.object(kernel.subprocess, "check_output", side_effect=["GPU-abc,0,0,90000\n", ""]*3), \
                patch.object(kernel.time, "sleep"), patch.object(kernel.os, "killpg", create=True) as kill:
            self.assertEqual(len(kernel.idle_samples()), 3)
            kill.assert_not_called()

    def test_foreign_compute_or_busy_rejected_not_killed(self):
        for utilization, memory, apps in ((99, 0, ""), (0, 2048, ""), (0, 0, "99,GPU-abc")):
            with patch.object(kernel.subprocess, "check_output", side_effect=[f"GPU-abc,{utilization},{memory},90000\n", apps]), \
                    patch.object(kernel.os, "killpg", create=True) as kill:
                with self.assertRaises(ValueError):
                    kernel.idle_samples()
                kill.assert_not_called()

    def test_deadline_kills_only_created_group_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as d:
            child = MagicMock(pid=4713)
            child.wait.side_effect = [subprocess.TimeoutExpired(["synthetic"], .01), 143]
            child.poll.return_value = 143
            with patch.object(kernel, "os", SimpleNamespace(name="posix", killpg=MagicMock())) as mocked_os, \
                    patch.object(kernel.subprocess, "Popen", return_value=child) as popen, \
                    patch.object(kernel.time, "monotonic", side_effect=[100., 100.02]):
                result = kernel.bounded_child(["synthetic"], cwd=d, env={}, seconds=.01, output=Path(d))
                self.assertTrue(result["timeout"])
                self.assertEqual(result["exit_code"], 143)
                self.assertEqual(mocked_os.killpg.call_args[0][0], 4713)
                self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertTrue((Path(d)/"console.log").is_file())

    def config(self):
        return {"schema": "umgs_gsplat_native_kernel_qualification_v1",
            "method_commit": "9e7e128821c84c823edf6597e6817777cbd69df6", "expected_host": platform.node(),
            "method_root": "synthetic_method", "orchestration_root": "synthetic_orchestrator",
            "orchestration_manifest_sha256": "a"*64,
            "runtime_packages": {"torch": "2.8.0+cu128", "gsplat": "1.4.0", "nerfstudio": "1.1.5"},
            "runtime_source_files": [{"path": "synthetic", "sha256": "a"*64}],
            "request": {"scene": CORE[0], "method": "ms_splatting_neural", "stage": "GPU_QUALIFICATION",
                "operation": "kernel_packet_parity", "output_class": "nonformal_qualification_only",
                "max_gpu_seconds": 900, "max_iterations": 0, "gpu_budget_seconds_remaining": 86400}}

    def test_training_or_other_environment_not_implemented(self):
        cfg = self.config()
        kernel.validate_request(cfg)
        cfg["request"]["max_iterations"] = 1
        with self.assertRaises(ValueError):
            kernel.validate_request(cfg)
        cfg = self.config()
        cfg["runtime_packages"]["gsplat"] = "2.0.0"
        with self.assertRaises(ValueError):
            kernel.validate_request(cfg)

    def test_missing_notification_or_cpu_gate_fails_before_gpu_query(self):
        for notified in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root, cfg = Path(d), self.config()

                def write(name, value):
                    p = root/name
                    p.write_text(json.dumps(value), encoding="utf-8")
                    return {"path": str(p), "sha256": sha256(p)}

                plan = make_plan("test_kernel")
                cfg["plan"] = write("plan.json", plan)
                message = write("synthetic_notification.json", {"synthetic": True})
                cfg["user_notification_path"] = message["path"]
                cfg["authorization"] = write("auth.json", {"campaign_id": "test_kernel",
                    "plan_sha256": plan["plan_sha256"], "explicit_user_gpu_available": notified,
                    "user_message_evidence_sha256": message["sha256"]})
                cfg["ownership_snapshot"] = write("ownership.json", {"server": "901",
                    "observed_at_unix": time.time(), "process_ownership_verified": True,
                    "foreign_jobs": 0, "unknown_jobs": 0, "own_active_jobs": 0, "active_transfers": 0})
                cfg["cpu_gates"] = {}
                config = write("config.json", cfg)
                with patch.object(kernel, "check_source"), patch.object(kernel, "check_snapshot"), \
                        patch.object(kernel, "idle_samples") as idle, patch.object(kernel, "bounded_child") as child:
                    with self.assertRaises(ValueError):
                        kernel.parent(config["path"], config["sha256"])
                    idle.assert_not_called()
                    child.assert_not_called()


if __name__ == "__main__":
    unittest.main()
