# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18866 -- C15 (the chain canary) blocks staging delivery.

Operator ruling 2026-09-23T17:12:15Z binds four chronically red checks to a
blocking surface NOW. C15 measures the deployed ``.201`` dev lane, so its surface
is staging delivery, read through ``lab_pass_receipt.py workflow-verdict``.

The falsifiers use the two runs the design names, with their real ids, shas and
timestamps: run 35869370954 (2026-09-23T13:46:03Z, failure at 0edf5c91) must
refuse and name itself; run 35884023865 (2026-09-23T15:46:29Z, success at
962aefc0) must pass. Every other branch -- absent, stale, cancelled, unreadable,
a manual dispatch laundering a red, a red re-run superseding a green -- must
refuse.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import (
    WORKFLOW_VERDICT_MAX_AGE_CEILING_HOURS,
    evaluate_workflow_verdict,
    main,
)

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnibase_infra"
WORKFLOW = "chain-canary.yml"

RED_ID = 35869370954
RED_SHA = "0edf5c9145876dbe22f6caf7a06b507c5d0fc7d3"
GREEN_ID = 35884023865
GREEN_SHA = "962aefc021b6da9fa17a89a1ad6ea874ecefd566"


def _run(
    run_id: int,
    *,
    conclusion: str,
    started: str,
    sha: str = GREEN_SHA,
    event: str = "schedule",
    branch: str = "dev",
    attempt: int = 1,
    created: str | None = None,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "event": event,
        "status": "completed",
        "conclusion": conclusion,
        "head_branch": branch,
        "head_sha": sha,
        "run_attempt": attempt,
        "created_at": created or started,
        "run_started_at": started,
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


RED = _run(RED_ID, conclusion="failure", started="2026-09-23T13:46:03Z", sha=RED_SHA)
GREEN = _run(GREEN_ID, conclusion="success", started="2026-09-23T15:46:29Z")


class _Runs:
    def __init__(self, runs: list[dict[str, Any]], *, raises: bool = False) -> None:
        self.runs = runs
        self.raises = raises
        self.paths: list[str] = []

    def __call__(self, path: str) -> bytes:
        self.paths.append(path)
        if self.raises:
            msg = "`gh api` exited 1: HTTP 403 Resource not accessible"
            raise RuntimeError(msg)
        return json.dumps({"workflow_runs": self.runs}).encode()


def _verdict(
    runs: _Runs,
    monkeypatch: Any,
    *,
    now: str,
    max_age: float = 3.0,
    events: tuple[str, ...] = ("schedule",),
) -> tuple[int, str]:
    monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", runs)
    out = io.StringIO()
    code = evaluate_workflow_verdict(
        REPO,
        WORKFLOW,
        "dev",
        max_age,
        events,
        out,
        now=datetime.fromisoformat(now.replace("Z", "+00:00")).astimezone(UTC),
    )
    return code, out.getvalue()


class TestKnownRunsFromTheDesign:
    def test_the_known_red_run_refuses_and_names_the_run_and_sha(
        self, monkeypatch: Any
    ) -> None:
        """Known-bad: at 14:30Z the newest measurement was the 13:46 red."""
        code, output = _verdict(_Runs([RED]), monkeypatch, now="2026-09-23T14:30:00Z")
        assert code == 1
        assert f"run {RED_ID} at {RED_SHA}" in output
        assert "concluded 'failure'" in output
        assert "There is no override flag" in output

    def test_the_known_green_run_passes(self, monkeypatch: Any) -> None:
        """Known-good: at 17:30Z the newest measurement was the 15:46 green."""
        code, output = _verdict(
            _Runs([GREEN, RED]), monkeypatch, now="2026-09-23T17:30:00Z"
        )
        assert code == 0
        assert f"run {GREEN_ID} concluded success" in output

    def test_a_newer_red_supersedes_an_older_green(self, monkeypatch: Any) -> None:
        newer_red = _run(1, conclusion="failure", started="2026-09-23T17:46:00Z")
        code, output = _verdict(
            _Runs([newer_red, GREEN]), monkeypatch, now="2026-09-23T18:30:00Z"
        )
        assert code == 1
        assert "run 1 at" in output


class TestFailClosed:
    def test_no_run_at_all_refuses(self, monkeypatch: Any) -> None:
        """Absence control: a runner that never picked the job up yields nothing."""
        code, output = _verdict(_Runs([]), monkeypatch, now="2026-09-23T17:30:00Z")
        assert code == 1
        assert "an absent measurement is not a pass" in output

    def test_a_stale_green_refuses(self, monkeypatch: Any) -> None:
        code, output = _verdict(_Runs([GREEN]), monkeypatch, now="2026-09-23T18:47:00Z")
        assert code == 1
        assert "past the 3h bound" in output
        assert str(GREEN_ID) in output

    @pytest.mark.parametrize(
        "conclusion",
        ["cancelled", "timed_out", "startup_failure", "skipped", "neutral", "stale"],
    )
    def test_an_instrument_fault_blocks_like_any_red(
        self, monkeypatch: Any, conclusion: str
    ) -> None:
        run = _run(7, conclusion=conclusion, started="2026-09-23T17:00:00Z")
        code, output = _verdict(_Runs([run]), monkeypatch, now="2026-09-23T17:30:00Z")
        assert code == 1
        assert f"concluded {conclusion!r}" in output

    def test_an_unreadable_surface_refuses_rather_than_passing(
        self, monkeypatch: Any
    ) -> None:
        code, output = _verdict(
            _Runs([], raises=True), monkeypatch, now="2026-09-23T17:30:00Z"
        )
        assert code == 1
        assert "unreadable" in output

    def test_a_listing_with_no_runs_key_refuses(self, monkeypatch: Any) -> None:
        def bad(_path: str) -> bytes:
            return b'{"total_count": 0}'

        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", bad)
        out = io.StringIO()
        assert (
            evaluate_workflow_verdict(REPO, WORKFLOW, "dev", 3.0, ["schedule"], out)
            == 1
        )

    @pytest.mark.parametrize(
        "max_age", [0.0, -1.0, WORKFLOW_VERDICT_MAX_AGE_CEILING_HOURS + 1]
    )
    def test_a_window_that_is_not_a_freshness_bound_refuses(
        self, monkeypatch: Any, max_age: float
    ) -> None:
        code, _ = _verdict(
            _Runs([GREEN]), monkeypatch, now="2026-09-23T17:30:00Z", max_age=max_age
        )
        assert code == 1


class TestNoLaundering:
    def test_a_manual_dispatch_green_does_not_launder_a_scheduled_red(
        self, monkeypatch: Any
    ) -> None:
        """A dispatch can point the probe elsewhere; it is not admitted for C15."""
        dispatch = _run(
            9,
            conclusion="success",
            started="2026-09-23T14:10:00Z",
            event="workflow_dispatch",
        )
        code, output = _verdict(
            _Runs([dispatch, RED]), monkeypatch, now="2026-09-23T14:30:00Z"
        )
        assert code == 1
        assert f"run {RED_ID}" in output
        assert "1 other(s) ignored" in output

    def test_a_green_on_another_branch_is_ignored(self, monkeypatch: Any) -> None:
        feature = _run(
            10, conclusion="success", started="2026-09-23T14:10:00Z", branch="feat"
        )
        code, _ = _verdict(
            _Runs([feature, RED]), monkeypatch, now="2026-09-23T14:30:00Z"
        )
        assert code == 1

    def test_a_red_rerun_of_an_older_run_supersedes_a_newer_green(
        self, monkeypatch: Any
    ) -> None:
        """Newest ATTEMPT wins in both directions: a re-run is a fresh measurement."""
        rerun = _run(
            RED_ID,
            conclusion="failure",
            started="2026-09-23T17:10:00Z",
            created="2026-09-23T13:46:03Z",
            sha=RED_SHA,
            attempt=2,
        )
        code, output = _verdict(
            _Runs([GREEN, rerun]), monkeypatch, now="2026-09-23T17:30:00Z"
        )
        assert code == 1
        assert "attempt 2" in output

    def test_a_green_rerun_of_the_red_run_passes(self, monkeypatch: Any) -> None:
        rerun = _run(
            RED_ID,
            conclusion="success",
            started="2026-09-23T14:40:00Z",
            created="2026-09-23T13:46:03Z",
            sha=RED_SHA,
            attempt=2,
        )
        code, _ = _verdict(_Runs([rerun]), monkeypatch, now="2026-09-23T15:00:00Z")
        assert code == 0

    def test_the_query_reads_completed_runs_on_the_branch(
        self, monkeypatch: Any
    ) -> None:
        runs = _Runs([GREEN])
        _verdict(runs, monkeypatch, now="2026-09-23T17:30:00Z")
        assert runs.paths == [
            f"repos/{REPO}/actions/workflows/{WORKFLOW}/runs"
            "?branch=dev&status=completed&per_page=50"
        ]


class TestCli:
    def test_the_cli_requires_an_admitted_event(self) -> None:
        with pytest.raises(SystemExit):
            main(
                [
                    "workflow-verdict",
                    "--repo",
                    REPO,
                    "--workflow",
                    WORKFLOW,
                    "--branch",
                    "dev",
                    "--max-age-hours",
                    "3",
                ]
            )


_DELIVER = Path(".github/workflows/deliver-dev-candidate-to-staging.yml")


def _gate_step() -> dict[str, Any]:
    workflow = yaml.safe_load(_DELIVER.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["lab-pass-gate"]["steps"]
    # OMN-19311 added a second workflow-verdict step (D11), so the C15 step is
    # found by the workflow it reads rather than by the subcommand alone.
    matches = [
        s
        for s in steps
        if "workflow-verdict" in str(s.get("run", ""))
        and "chain-canary.yml" in str(s.get("run", ""))
    ]
    assert len(matches) == 1, "the lab-pass-gate job must read the C15 verdict once"
    return dict(matches[0])


class TestDeliveryWiring:
    def test_the_lab_pass_gate_job_reads_the_chain_canary_verdict(self) -> None:
        run = str(_gate_step()["run"])
        assert "--workflow chain-canary.yml" in run
        assert "--branch dev" in run
        assert "--max-age-hours 3" in run
        assert "--event schedule" in run
        assert "--event workflow_dispatch" not in run
        assert "set -euo pipefail" in run

    def test_the_verdict_step_has_no_condition_and_cannot_be_skipped(self) -> None:
        step = _gate_step()
        assert "if" not in step, "a condition on the C15 step is a skip path"
        assert "continue-on-error" not in step

    def test_the_dispatch_still_needs_the_gate_job(self) -> None:
        workflow = yaml.safe_load(_DELIVER.read_text(encoding="utf-8"))
        assert "lab-pass-gate" in workflow["jobs"]["dispatch-to-staging"]["needs"]
        assert "continue-on-error" not in workflow["jobs"]["lab-pass-gate"]
