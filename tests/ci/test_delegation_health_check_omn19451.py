# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19451 -- the delegation-health check on runtime PRs.

AC1: a replayed red run makes the check fail on a runtime-affecting PR and the
     failure names the red run id.
AC2: a typed fix-forward label naming a ticket admits the PR and is recorded; a
     label with no ticket id is refused.
AC3: the check marked required with fewer than 7 days of shadow runs fails.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import delegation_health_check as dh

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
CONFIG = Path("config/delegation_health_check.yaml")


def _run(run_id: int, *, conclusion: str, repo: str) -> dict[str, Any]:
    stamp = (NOW - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": run_id,
        "conclusion": conclusion,
        "created_at": stamp,
        "run_started_at": stamp,
        "run_attempt": 1,
        "event": "schedule",
        "display_title": "verdict",
        "head_branch": "dev",
        "head_sha": "0" * 40,
        "status": "completed",
        "html_url": f"https://github.com/{repo}/actions/runs/{run_id}",
    }


def _replay(monkeypatch: Any, red: dict[str, int]) -> None:
    """Every source is green except the workflows named in ``red`` (id per workflow)."""

    def fake(path: str) -> bytes:
        repo = path.split("repos/")[1].split("/actions")[0]
        for workflow, run_id in red.items():
            if f"/workflows/{workflow}/" in path:
                return json.dumps(
                    {"workflow_runs": [_run(run_id, conclusion="failure", repo=repo)]}
                ).encode()
        return json.dumps(
            {"workflow_runs": [_run(1, conclusion="success", repo=repo)]}
        ).encode()

    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", fake)


def _sources() -> tuple[dh.ModelVerdictSource, ...]:
    return dh.load_config(CONFIG).sources


def _check(
    *, runtime: bool = True, labels: tuple[str, ...] = ()
) -> tuple[int, str, dict[str, Any]]:
    out = io.StringIO()
    record: dict[str, Any] = {}
    code = dh.evaluate_delegation_health(
        _sources(),
        runtime_affecting=runtime,
        labels=labels,
        out=out,
        now=NOW,
        record=record,
    )
    return code, out.getvalue(), record


class TestAC1RedVerdictFailsRuntimePRs:
    def test_a_red_nightly_fails_a_runtime_pr_and_names_the_run(
        self, monkeypatch: Any
    ) -> None:
        _replay(monkeypatch, {"delegation-regression-nightly.yml": 35832924275})
        code, output, _ = _check()
        assert code == 1
        assert "35832924275" in output
        assert "delegation-regression-nightly" in output

    def test_a_red_m4_verdict_fails_and_names_the_run(self, monkeypatch: Any) -> None:
        _replay(monkeypatch, {"m4-customer-pass-verdict.yml": 777001})
        code, output, _ = _check()
        assert code == 1
        assert "777001" in output

    def test_every_red_is_named_not_just_the_first(self, monkeypatch: Any) -> None:
        _replay(
            monkeypatch,
            {
                "delegation-regression-nightly.yml": 111,
                "m4-c17-customer-surface-verdict.yml": 222,
            },
        )
        code, output, _ = _check()
        assert code == 1
        assert "111" in output
        assert "222" in output

    def test_all_green_passes(self, monkeypatch: Any) -> None:
        _replay(monkeypatch, {})
        code, _, _ = _check()
        assert code == 0

    def test_a_non_runtime_pr_is_not_read_at_all(self, monkeypatch: Any) -> None:
        _replay(monkeypatch, {"delegation-regression-nightly.yml": 5})
        code, output, _ = _check(runtime=False)
        assert code == 0
        assert "not runtime-affecting" in output

    def test_an_unreadable_surface_is_red_not_green(self, monkeypatch: Any) -> None:
        def boom(_path: str) -> bytes:
            raise RuntimeError("gh api exited 1")

        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", boom)
        code, output, _ = _check()
        assert code == 1
        assert "unreadable" in output

    def test_the_configured_sources_cover_the_nightly_and_every_m4_verdict(
        self,
    ) -> None:
        names = {s.name for s in _sources()}
        assert "delegation-regression-nightly" in names
        assert {s.workflow for s in _sources() if s.name.startswith("m4-")} == {
            "m4-customer-pass-verdict.yml",
            "m4-c17-customer-surface-verdict.yml",
        }


class TestAC2FixForwardLabel:
    def test_a_typed_label_admits_the_pr_and_records_it(self, monkeypatch: Any) -> None:
        _replay(monkeypatch, {"delegation-regression-nightly.yml": 35832924275})
        code, output, record = _check(labels=("delegation-fix-forward:OMN-19432",))
        assert code == 0
        assert record["admitted_by"] == ["OMN-19432"]
        assert record["red_runs"] == [
            {"source": "delegation-regression-nightly", "run_id": "35832924275"}
        ]
        assert "OMN-19432" in output

    def test_a_label_with_no_ticket_is_refused(self, monkeypatch: Any) -> None:
        _replay(monkeypatch, {"delegation-regression-nightly.yml": 35832924275})
        code, output, record = _check(labels=("delegation-fix-forward",))
        assert code == 1
        assert "ticket" in output
        assert record.get("admitted_by") in (None, [])

    @pytest.mark.parametrize(
        "label",
        [
            "delegation-fix-forward:",
            "delegation-fix-forward:fix-it",
            "delegation-fix-forward:omn-19432",
            "delegation-fix-forward:OMN-",
        ],
    )
    def test_a_malformed_ticket_is_refused(self, monkeypatch: Any, label: str) -> None:
        _replay(monkeypatch, {"delegation-regression-nightly.yml": 5})
        code, _, _ = _check(labels=(label,))
        assert code == 1

    def test_a_label_on_a_green_pr_records_no_red_and_passes(
        self, monkeypatch: Any
    ) -> None:
        _replay(monkeypatch, {})
        code, _, record = _check(labels=("delegation-fix-forward:OMN-19432",))
        assert code == 0
        assert record.get("red_runs") == []

    def test_label_parsing(self) -> None:
        reading = dh.parse_fix_forward_labels(
            ["runtime_change", "delegation-fix-forward:OMN-5", "delegation-fix-forward"]
        )
        assert reading.tickets == ("OMN-5",)
        assert reading.refused == ("delegation-fix-forward",)


class TestAC3ShadowBeforeRequired:
    def test_the_shipped_config_is_shadow(self) -> None:
        cfg = dh.load_config(CONFIG)
        assert dh.validate_rollout(cfg, now=NOW) == []
        assert all(not r.required for r in cfg.repos.values())

    def test_required_with_fewer_than_seven_days_of_shadow_is_refused(self) -> None:
        cfg = dh.load_config(CONFIG)
        cfg.repos["omnimarket"] = dh.ModelRepoRollout(
            shadow_started_at=date(2026, 9, 25), required=True
        )
        errors = dh.validate_rollout(cfg, now=NOW)
        assert len(errors) == 1
        assert "omnimarket" in errors[0]
        assert "7" in errors[0]

    def test_required_after_seven_days_is_allowed(self) -> None:
        cfg = dh.load_config(CONFIG)
        cfg.repos["omnimarket"] = dh.ModelRepoRollout(
            shadow_started_at=date(2026, 9, 22), required=True
        )
        assert dh.validate_rollout(cfg, now=NOW) == []

    def test_infra_ci_summary_registration_matches_the_declared_state(self) -> None:
        from scripts.ci.ci_summary_gate import STRICT_GATE_JOBS

        cfg = dh.load_config(CONFIG)
        registered = dh.CHECK_JOB_NAME in STRICT_GATE_JOBS
        assert registered == cfg.repos["omnibase_infra"].required


class TestWorkflowWiring:
    def test_the_reusable_runs_the_script_from_the_pinned_workflow_sha(self) -> None:
        import yaml

        path = Path(".github/workflows/delegation-health-reusable.yml")
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "workflow_call" in doc[True]
        text = path.read_text(encoding="utf-8")
        assert "github.job_workflow_sha" in text
        assert "scripts/ci/delegation_health_check.py" in text

    def test_the_caller_passes_labels_and_reacts_to_label_changes(self) -> None:
        import yaml

        path = Path(".github/workflows/delegation-health-check.yml")
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        types = doc[True]["pull_request"]["types"]
        assert {"labeled", "unlabeled"} <= set(types)
        job = doc["jobs"]["delegation-health"]
        assert job["with"]["repo_key"] == "omnibase_infra"
        assert "labels" in job["with"]
