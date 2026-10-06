# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19451 -- the delegation-health check on runtime PRs.

AC1: a replayed red run makes the check fail on a runtime-affecting PR and the
     failure names the red run id.
AC2: a typed fix-forward label naming a ticket admits the PR and is recorded; a
     label with no ticket id is refused.
AC3 (amended by operator ruling 2026-09-29T12:02:35Z, OMN-19998): there is no
     shadow state. The check's context is asserted by CI Summary on every pull
     request and merge-queue head, and nothing excludes its red.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import delegation_health_check as dh

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 29, 12, 0, 0, tzinfo=UTC)
CONFIG = Path("config/delegation_health_check.yaml")
CONTEXT = "delegation-health-check / Delegation Health Check"


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
        _replay(monkeypatch, {"m4-c17-customer-surface-verdict.yml": 777001})
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

    def test_the_configured_sources_are_the_nightly_and_the_m4_verdicts(
        self,
    ) -> None:
        assert {s.workflow for s in _sources()} == {
            "delegation-regression-nightly.yml",
            "m4-c17-customer-surface-verdict.yml",
        }

    def test_the_staging_c9_walk_is_not_an_m4_source(self) -> None:
        """C9 grades the staging plane and belongs to M4.5, not M4.

        Operator ruling 2026-09-22T20:35:26Z moved the cloud-plane criteria C8,
        C9 and C10 to M4.5; ruling 2026-09-28T17:47:08Z put staging recovery on
        hold. Reading m4-customer-pass-verdict.yml here blocked every runtime
        PR on a staging outage no runtime PR can repair.
        """
        assert "m4-customer-pass-verdict.yml" not in {s.workflow for s in _sources()}


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

    @pytest.mark.parametrize(
        "label",
        [
            "delegation-fix-forward",
            "delegation-fix-forward:",
            "delegation-fix-forward:fix-it",
            "delegation-fix-forward:omn-19432",
            "delegation-fix-forward:OMN-",
        ],
    )
    def test_an_invalid_label_is_refused_even_when_verdicts_are_green(
        self, monkeypatch: Any, label: str
    ) -> None:
        _replay(monkeypatch, {})
        code, output, record = _check(labels=(label,))
        assert code == 1
        assert label in output
        assert "requires an OMN-<digits> ticket" in output
        assert "all sources green" not in output
        assert record["red_runs"] == []
        assert record["admitted_by"] == []

    @pytest.mark.parametrize("red", [{}, {"delegation-regression-nightly.yml": 5}])
    def test_a_valid_ticket_does_not_hide_an_invalid_label(
        self, monkeypatch: Any, red: dict[str, int]
    ) -> None:
        _replay(monkeypatch, red)
        code, output, record = _check(
            labels=("delegation-fix-forward:OMN-19432", "delegation-fix-forward")
        )
        assert code == 1
        assert "refused labels: delegation-fix-forward" in output
        assert record["admitted_by"] == []

    def test_label_parsing(self) -> None:
        reading = dh.parse_fix_forward_labels(
            ["runtime_change", "delegation-fix-forward:OMN-5", "delegation-fix-forward"]
        )
        assert reading.tickets == ("OMN-5",)
        assert reading.refused == ("delegation-fix-forward",)


class TestAC3NoShadowState:
    def test_the_check_carries_no_shadow_label(self) -> None:
        assert dh.CHECK_JOB_NAME == "Delegation Health Check"

    def test_the_config_declares_no_rollout_state(self) -> None:
        import yaml

        raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        assert set(raw) == {"sources"}

    def test_ci_summary_asserts_the_context_and_excludes_nothing(self) -> None:
        from scripts.ci.ci_summary_gate import (
            EXPECTED_EXTERNAL_CONTEXTS,
            EXTERNAL_SWEEP_EXCLUSIONS,
        )

        assert CONTEXT in EXPECTED_EXTERNAL_CONTEXTS
        assert not any(
            "Delegation Health" in name for name in EXTERNAL_SWEEP_EXCLUSIONS
        )


def _workflow(name: str) -> tuple[dict[Any, Any], str]:
    import yaml

    path = Path(".github/workflows") / name
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text), text


class TestWorkflowWiring:
    def test_the_reusable_runs_the_script_from_the_pinned_workflow_sha(self) -> None:
        doc, text = _workflow("delegation-health-reusable.yml")
        assert "workflow_call" in doc[True]
        assert "github.job_workflow_sha" in text
        assert "scripts/ci/delegation_health_check.py" in text
        assert doc["jobs"]["delegation-health"]["name"] == dh.CHECK_JOB_NAME

    def test_the_caller_reports_on_pull_requests_and_queue_heads(self) -> None:
        doc, _ = _workflow("delegation-health-check.yml")
        triggers = doc[True]
        assert {"labeled", "unlabeled"} <= set(triggers["pull_request"]["types"])
        assert "merge_group" in triggers
        assert "paths" not in (triggers["pull_request"] or {})

    def test_the_caller_job_cannot_skip_as_passed(self) -> None:
        doc, _ = _workflow("delegation-health-check.yml")
        assert list(doc["jobs"]) == ["delegation-health-check"]
        job = doc["jobs"]["delegation-health-check"]
        assert job["with"]["repo_key"] == "omnibase_infra"
        for key in ("needs", "if", "continue-on-error"):
            assert key not in job, key

    def test_the_check_run_is_the_registered_context(self) -> None:
        """`<caller job id> / <called job name>`, with no caller `name:`.

        That is the spelling EXPECTED_EXTERNAL_CONTEXTS asserts and the prefix
        the omniclaude advisory-job gate matches a called workflow's job by.
        """
        caller, _ = _workflow("delegation-health-check.yml")
        reusable, _ = _workflow("delegation-health-reusable.yml")
        (job_id,) = caller["jobs"]
        assert "name" not in caller["jobs"][job_id]
        inner = reusable["jobs"]["delegation-health"]["name"]
        assert f"{job_id} / {inner}" == CONTEXT

    def test_queue_heads_never_share_a_concurrency_group(self) -> None:
        doc, _ = _workflow("delegation-health-check.yml")
        assert "github.ref" in doc["concurrency"]["group"]

    def test_the_reusable_derives_files_and_labels_for_both_events(self) -> None:
        doc, text = _workflow("delegation-health-reusable.yml")
        assert set(doc[True]["workflow_call"]["inputs"]) == {"repo_key"}
        assert "pull_request)" in text and "merge_group)" in text
        assert "github.event.merge_group.head_ref" in text
        assert "steps.event.outputs.files" in text
        assert "steps.event.outputs.labels" in text
        steps = doc["jobs"]["delegation-health"]["steps"]
        assert not any(step.get("continue-on-error") for step in steps)
