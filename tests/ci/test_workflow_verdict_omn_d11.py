# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19311 -- D11 blocks staging delivery through the workflow-verdict reader.

Operator ruling 2026-09-23T17:12:15Z: D11 (omnimarket
``delegation-regression-nightly.yml``, the Layer-2 golden corpus against the
stability-test lane) is bound to a blocking surface now, while it is red. The
surface is staging delivery, on every path, read through the one
``workflow-verdict`` reader (OMN-18866) with D11's own arguments.

The falsifier pair, replayed from the live API shape:

  known-bad : run 35832924275 (2026-09-23, conclusion failure, five hard
              breaks including I5 escalating to gemini-2.5-flash with 0/0
              tokens) refuses, and the refusal names the run id.
  stale     : a success 27 hours old refuses.
  known-good: a fresh scheduled success, or a fresh stability-test dispatch,
              admits.

plus the laundering control this ticket adds to the reader: a green dispatch
aimed at the dev lane is not the stability-test verdict and does not admit.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import evaluate_workflow_verdict, main

pytestmark = pytest.mark.unit

WORKFLOW_PATH = Path(".github/workflows/deliver-dev-candidate-to-staging.yml")

REPO = "OmniNode-ai/omnimarket"
WORKFLOW = "delegation-regression-nightly.yml"
EVENTS = ("schedule", "workflow_dispatch")
TOKEN = "lane=stability-test"

#: The live newest completed run when the binding landed, as the REST API
#: returned it (fields trimmed to the ones the reader uses).
RUN_35832924275: dict[str, Any] = {
    "conclusion": "failure",
    "created_at": "2026-09-23T07:40:13Z",
    "run_started_at": "2026-09-23T07:40:13Z",
    "run_attempt": 1,
    "display_title": "Delegation Regression (nightly)",
    "event": "schedule",
    "head_branch": "dev",
    "head_sha": "03d2ad0fdf9fdcc19de6badc35b1abe922ffe384",
    "html_url": "https://github.com/OmniNode-ai/omnimarket/actions/runs/35832924275",
    "id": 35832924275,
    "status": "completed",
    "updated_at": "2026-09-23T08:18:14Z",
}

NOW = datetime(2026, 9, 23, 18, 0, 0, tzinfo=UTC)


def _run(
    run_id: int,
    *,
    conclusion: str = "success",
    started: datetime = NOW - timedelta(hours=2),
    event: str = "schedule",
    title: str = "Delegation Regression (nightly) lane=stability-test",
) -> dict[str, Any]:
    stamp = started.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": run_id,
        "conclusion": conclusion,
        "created_at": stamp,
        "run_started_at": stamp,
        "run_attempt": 1,
        "event": event,
        "display_title": title,
        "head_branch": "dev",
        "head_sha": "0" * 40,
        "status": "completed",
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


def _read(
    monkeypatch: Any, runs: list[dict[str, Any]], *, token: str = TOKEN
) -> tuple[int, str]:
    monkeypatch.setattr(
        "scripts.ci.lab_pass_receipt._gh_api",
        lambda _path: json.dumps({"workflow_runs": runs}).encode(),
    )
    out = io.StringIO()
    code = evaluate_workflow_verdict(
        REPO,
        WORKFLOW,
        "dev",
        26,
        EVENTS,
        out,
        now=NOW,
        dispatch_title_contains=token,
    )
    return code, out.getvalue()


class TestTheD11Verdict:
    def test_the_live_red_run_refuses_and_names_the_run(self, monkeypatch: Any) -> None:
        code, output = _read(monkeypatch, [RUN_35832924275])
        assert code == 1
        assert "35832924275" in output
        assert "'failure'" in output

    def test_a_27_hour_old_success_is_stale(self, monkeypatch: Any) -> None:
        code, output = _read(monkeypatch, [_run(7, started=NOW - timedelta(hours=27))])
        assert code == 1
        assert "past the 26h bound" in output

    def test_a_fresh_scheduled_success_admits(self, monkeypatch: Any) -> None:
        code, output = _read(monkeypatch, [_run(42), RUN_35832924275])
        assert code == 0
        assert "run 42 concluded success" in output

    def test_a_stability_dispatch_is_a_new_measurement_and_counts(
        self, monkeypatch: Any
    ) -> None:
        dispatch = _run(
            100, event="workflow_dispatch", started=NOW - timedelta(minutes=50)
        )
        code, _ = _read(monkeypatch, [dispatch, RUN_35832924275])
        assert code == 0

    def test_a_green_dev_lane_dispatch_cannot_launder_a_stability_red(
        self, monkeypatch: Any
    ) -> None:
        dev_dispatch = _run(
            99,
            event="workflow_dispatch",
            title="Delegation Regression (nightly) lane=dev",
            started=NOW - timedelta(minutes=50),
        )
        code, output = _read(monkeypatch, [dev_dispatch, RUN_35832924275])
        assert code == 1
        assert "35832924275" in output
        assert (
            "admitted only when the run title carries 'lane=stability-test'" in output
        )

    def test_a_dispatch_with_no_lane_in_its_title_does_not_count(
        self, monkeypatch: Any
    ) -> None:
        untitled = _run(
            101,
            event="workflow_dispatch",
            title="Delegation Regression (nightly)",
            started=NOW - timedelta(minutes=50),
        )
        code, _ = _read(monkeypatch, [untitled])
        assert code == 1

    def test_without_the_token_the_same_dispatch_would_have_laundered_it(
        self, monkeypatch: Any
    ) -> None:
        """Positive control: the token is what refuses the dev-lane dispatch."""
        dev_dispatch = _run(
            99,
            event="workflow_dispatch",
            title="Delegation Regression (nightly) lane=dev",
            started=NOW - timedelta(minutes=50),
        )
        code, _ = _read(monkeypatch, [dev_dispatch, RUN_35832924275], token="")
        assert code == 0

    def test_the_token_does_not_touch_scheduled_runs(self, monkeypatch: Any) -> None:
        """A scheduled run takes no inputs: pre-run-name nights still count."""
        old_title = dict(RUN_35832924275, conclusion="success")
        old_title["run_started_at"] = "2026-09-23T07:40:13Z"
        code, _ = _read(monkeypatch, [old_title])
        assert code == 0

    def test_the_cli_carries_the_token(self, monkeypatch: Any, capsys: Any) -> None:
        dev_dispatch = _run(
            99,
            event="workflow_dispatch",
            title="Delegation Regression (nightly) lane=dev",
            started=datetime.now(tz=UTC) - timedelta(minutes=5),
        )
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._gh_api",
            lambda _path: json.dumps(
                {"workflow_runs": [dev_dispatch, RUN_35832924275]}
            ).encode(),
        )
        code = main(
            [
                "workflow-verdict",
                "--repo",
                REPO,
                "--workflow",
                WORKFLOW,
                "--branch",
                "dev",
                "--max-age-hours",
                "26",
                "--event",
                "schedule",
                "--event",
                "workflow_dispatch",
                "--dispatch-title-contains",
                TOKEN,
            ]
        )
        assert code == 1
        assert "35832924275" in capsys.readouterr().out


class TestWiring:
    """The verdict is only a gate if the delivery workflow runs it on every path."""

    def _gate_job(self) -> dict[str, Any]:
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        job: dict[str, Any] = workflow["jobs"]["lab-pass-gate"]
        return job

    def _d11_step(self) -> dict[str, Any]:
        matches = [
            s
            for s in self._gate_job()["steps"]
            if "delegation-regression-nightly.yml" in str(s.get("run", ""))
        ]
        assert len(matches) == 1, "exactly one D11 verdict step is expected"
        step: dict[str, Any] = matches[0]
        return step

    def test_the_d11_read_runs_on_every_path(self) -> None:
        step = self._d11_step()
        assert "if" not in step, (
            "the D11 read must bind the push path too: the push path vendors "
            "omnimarket dev HEAD"
        )
        assert "continue-on-error" not in step
        token_step = next(
            s for s in self._gate_job()["steps"] if s.get("id") == "d11-token"
        )
        assert "if" not in token_step
        assert "continue-on-error" not in token_step
        assert "continue-on-error" not in self._gate_job()

    def test_the_d11_read_runs_before_any_receipt_read(self) -> None:
        """Its verdict is printed on every delivery, whatever the receipts say."""
        runs = [str(s.get("run", "")) for s in self._gate_job()["steps"]]
        d11 = next(i for i, r in enumerate(runs) if "delegation-regression" in r)
        first_gate = next(
            i for i, r in enumerate(runs) if "lab_pass_receipt.py gate" in r
        )
        assert d11 < first_gate

    def test_the_d11_read_names_the_governed_verdict(self) -> None:
        run = str(self._d11_step()["run"])
        assert "workflow-verdict" in run
        assert "--repo OmniNode-ai/omnimarket" in run
        assert "--branch dev" in run
        assert "--max-age-hours 26" in run
        assert "--event schedule" in run
        assert "--event workflow_dispatch" in run
        assert '--dispatch-title-contains "lane=stability-test"' in run
        assert "set -euo pipefail" in run

    def test_the_dispatch_still_needs_the_gate_job(self) -> None:
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        dispatch = workflow["jobs"]["dispatch-to-staging"]
        assert "lab-pass-gate" in dispatch["needs"]
        assert "if" not in dispatch
