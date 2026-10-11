# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Seed-provenance gate refusal and positive controls for OMN-18786."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci import ci_summary_gate

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "check_seed_provenance.py"
)

# Import the check functions directly for unit testing
sys.path.insert(0, str(SCRIPT_PATH.parent))
from check_seed_provenance import (
    _has_provenance,
    _is_seed_or_demo,
    _publishes_events,
    check_scripts,
)


@pytest.mark.unit
class TestHelperFunctions:
    def test_is_seed_or_demo_matches_seed(self) -> None:
        assert _is_seed_or_demo(Path("seed-infisical.py"))

    def test_is_seed_or_demo_matches_demo(self) -> None:
        assert _is_seed_or_demo(Path("demo_runtime_verification.py"))

    def test_is_seed_or_demo_rejects_other(self) -> None:
        assert not _is_seed_or_demo(Path("check_topic_drift.py"))

    def test_publishes_events_detects_kafka_producer(self) -> None:
        assert _publishes_events("producer = AIOKafkaProducer(bootstrap_servers=...)")

    def test_publishes_events_detects_send_and_wait(self) -> None:
        assert _publishes_events("await producer.send_and_wait(topic, body)")

    def test_publishes_events_detects_emit(self) -> None:
        assert _publishes_events("event_bus.emit(envelope)")

    def test_publishes_events_false_for_docstring_only(self) -> None:
        # "produces" (with s) does not match \bproduce\b
        assert not _publishes_events("re-running against a correct realm produces all")

    def test_has_provenance_true(self) -> None:
        assert _has_provenance('payload["data_provenance"] = "demo_seeded"')

    def test_has_provenance_false(self) -> None:
        assert not _has_provenance("event_type = 'baselines.computed'")


@pytest.mark.unit
class TestCheckScripts:
    def test_no_warnings_when_provenance_present(self, tmp_path: Path) -> None:
        script = tmp_path / "seed_example.py"
        script.write_text(
            "async def run():\n"
            '    payload = {"data_provenance": "demo_seeded"}\n'
            '    await producer.send_and_wait("topic", json.dumps(payload).encode())\n'
        )
        warnings = check_scripts(tmp_path)
        assert warnings == []

    def test_warning_when_provenance_missing(self, tmp_path: Path) -> None:
        script = tmp_path / "seed_example.py"
        script.write_text(
            "async def run():\n"
            '    payload = {"event_type": "foo"}\n'
            '    await producer.send_and_wait("topic", json.dumps(payload).encode())\n'
        )
        warnings = check_scripts(tmp_path)
        assert len(warnings) == 1
        assert "data_provenance" in warnings[0]
        assert "seed_example.py" in warnings[0]

    def test_non_seed_demo_scripts_not_checked(self, tmp_path: Path) -> None:
        script = tmp_path / "publish_pr_merged_event.py"
        script.write_text(
            'async def run():\n    await producer.send_and_wait("topic", b"data")\n'
        )
        warnings = check_scripts(tmp_path)
        assert warnings == []

    def test_seed_script_without_event_publish_not_warned(self, tmp_path: Path) -> None:
        script = tmp_path / "seed_config.py"
        script.write_text(
            'def run():\n    client.create_secret(key="FOO", value="bar")\n'
        )
        warnings = check_scripts(tmp_path)
        assert warnings == []

    def test_multiple_flagged_scripts(self, tmp_path: Path) -> None:
        for name in ("seed_a.py", "demo_b.py"):
            (tmp_path / name).write_text(
                'await producer.send_and_wait("topic", b"payload")\n'
            )
        warnings = check_scripts(tmp_path)
        assert len(warnings) == 2


@pytest.mark.unit
class TestCLI:
    @staticmethod
    def run_check(scripts_dir: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--scripts-dir", str(scripts_dir)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    @pytest.mark.parametrize("name", ["seed_example.py", "demo_example.py"])
    def test_missing_payload_provenance_fails_then_added_field_passes(
        self, tmp_path: Path, name: str
    ) -> None:
        script = tmp_path / name
        script.write_text(
            'payload = {"event_type": "foo"}\n'
            'await producer.send_and_wait("topic", json.dumps(payload).encode())\n'
        )
        red = self.run_check(tmp_path)
        assert red.returncode == 1, red.stdout + red.stderr
        assert name in red.stdout
        assert "data_provenance" in red.stdout

        script.write_text(
            'payload = {"event_type": "foo", "data_provenance": "demo_seeded"}\n'
            'await producer.send_and_wait("topic", json.dumps(payload).encode())\n'
        )
        green = self.run_check(tmp_path)
        assert green.returncode == 0, green.stdout + green.stderr
        assert "clean" in green.stdout

    @pytest.mark.parametrize("kind", ["missing", "empty", "not-directory"])
    def test_invalid_scan_root_refuses(self, tmp_path: Path, kind: str) -> None:
        root = tmp_path / "scripts"
        if kind == "empty":
            root.mkdir()
        elif kind == "not-directory":
            root.write_text("not a directory")
        result = self.run_check(root)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "ERROR" in result.stdout

    def test_unreadable_candidate_refuses(self, tmp_path: Path) -> None:
        (tmp_path / "seed_invalid.py").write_bytes(b"\xff")
        result = self.run_check(tmp_path)
        assert result.returncode == 1, result.stdout + result.stderr
        assert "seed_invalid.py" in result.stdout
        assert "ERROR" in result.stdout

    def test_unreadable_file_is_not_silently_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "seed_unreadable.py").touch()

        def refuse_read(self: Path, **kwargs: object) -> str:
            raise PermissionError("fixture: unreadable candidate")

        monkeypatch.setattr(Path, "read_text", refuse_read)
        findings = check_scripts(tmp_path)
        assert len(findings) == 1
        assert "seed_unreadable.py" in findings[0]
        assert "ERROR" in findings[0]

    def test_any_finding_fails_even_with_a_valid_sibling(self, tmp_path: Path) -> None:
        (tmp_path / "seed_bad.py").write_text('await producer.send_and_wait("t", b"x")')
        (tmp_path / "seed_good.py").write_text(
            'payload = {"data_provenance": "demo_seeded"}\n'
            'await producer.send_and_wait("t", payload)'
        )
        assert self.run_check(tmp_path).returncode == 1


@pytest.mark.unit
class TestGateWiring:
    context = "Seed Provenance Check"
    repo_root = SCRIPT_PATH.parent.parent

    def test_workflow_runs_blocking_check_and_controls_on_every_pr(self) -> None:
        workflow = yaml.safe_load(
            (self.repo_root / ".github/workflows/seed-provenance-check.yml").read_text()
        )
        # PyYAML's YAML 1.1 loader treats the Actions 'on' key as True.
        triggers = workflow.get("on", workflow.get(True))
        for event in ("pull_request", "merge_group", "push"):
            assert event in triggers
            assert not triggers[event]  # no path/branch/type filter
        job = workflow["jobs"]["seed-provenance-check"]
        assert job["name"] == self.context
        assert "if" not in job
        assert "needs" not in job
        assert not job.get("continue-on-error", False)
        steps = job["steps"]
        assert all(not step.get("continue-on-error", False) for step in steps)
        runs = "\n".join(step.get("run", "") for step in steps)
        assert "python scripts/check_seed_provenance.py --scripts-dir scripts/" in runs
        assert "pytest tests/unit/scripts/test_seed_provenance_check.py" in runs

    def test_precommit_runs_the_same_gate(self) -> None:
        config = yaml.safe_load(
            (self.repo_root / ".pre-commit-config.yaml").read_text()
        )
        hook = next(
            hook
            for repo in config["repos"]
            for hook in repo["hooks"]
            if hook["id"] == "seed-provenance-check"
        )
        assert (
            hook["entry"]
            == "uv run python scripts/check_seed_provenance.py --scripts-dir scripts/"
        )
        assert hook["pass_filenames"] is False
        assert hook["always_run"] is True
        assert "pre-commit" in hook["stages"]

    @pytest.mark.parametrize("conclusion", ["failure", "skipped", "cancelled", None])
    def test_nonpassing_or_absent_context_cannot_green_ci_summary(
        self, conclusion: str | None
    ) -> None:
        assert self.context in ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS
        rows = [
            {"name": name, "status": "completed", "conclusion": "success"}
            for name in ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS
            if name != self.context
        ]
        if conclusion is not None:
            rows.append(
                {"name": self.context, "status": "completed", "conclusion": conclusion}
            )
        code, report = ci_summary_gate.evaluate(
            [],
            strict_gates=(),
            skippable_gates=(),
            check_runs=rows,
            external_contexts=ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS,
        )
        expected = (
            ci_summary_gate.EXIT_PENDING
            if conclusion is None
            else ci_summary_gate.EXIT_FAILURE
        )
        assert code == expected, report
        assert self.context in report

    def test_green_context_passes_ci_summary(self) -> None:
        assert self.context in ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS
        rows = [
            {"name": name, "status": "completed", "conclusion": "success"}
            for name in ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS
        ]
        code, report = ci_summary_gate.evaluate(
            [],
            strict_gates=(),
            skippable_gates=(),
            check_runs=rows,
            external_contexts=ci_summary_gate.EXPECTED_EXTERNAL_CONTEXTS,
        )
        assert code == ci_summary_gate.EXIT_SUCCESS, report
