import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidence_agent_rl.demo import main, run_demo


class OfflineDemoTests(unittest.TestCase):
    def test_offline_demo_reports_success_rejection_and_replay(self):
        with (
            patch("socket.socket", side_effect=AssertionError("network forbidden")),
            patch(
                "pathlib.Path.open", side_effect=AssertionError("file access forbidden")
            ),
        ):
            report = run_demo()
        self.assertTrue(report["accepted"]["success"])
        self.assertTrue(report["rejected"]["blocked"])
        self.assertFalse(report["rejected"]["success"])
        self.assertTrue(report["replay_consistent"])
        self.assertEqual(
            [step["action"]["name"] for step in report["accepted"]["trace"]],
            ["search", "retrieve", "inspect_span", "check_citation", "answer"],
        )
        self.assertEqual(report, run_demo())

    def test_installed_module_cli_emits_json(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-m", "evidence_agent_rl.demo"],
                cwd=directory,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)["replay_consistent"])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_explicit_output_is_saved_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--output", str(output)]), 0)
            original = output.read_bytes()
            self.assertTrue(json.loads(original)["accepted"]["success"])
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main(["--output", str(output)])
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), original)
