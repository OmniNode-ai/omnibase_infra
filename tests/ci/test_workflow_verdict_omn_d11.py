# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19311 -- D11 blocks staging delivery through a workflow-verdict read.

Operator ruling 2026-09-23T17:12:15Z: D11 (omnimarket
``delegation-regression-nightly.yml``, the Layer-2 golden corpus against the
stability-test lane) is bound to a blocking surface now, while it is red. The
surface is staging delivery, on every path.

The falsifier pair, replayed from the live API shape:

  known-bad : run 35832924275 (2026-09-23, conclusion failure, five hard
              breaks including I5 escalating to gemini-2.5-flash with 0/0
              tokens) refuses, and the refusal names the run id.
  stale     : a success 27 hours old refuses.
  known-good: a fresh success admits.

plus the laundering control: a green dispatch aimed at the dev lane is not the
stability-test verdict and does not admit.
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci.lab_pass_receipt import (
    ModelWorkflowVerdictRequest,
    build_parser,
    evaluate_workflow_verdict,
    judge_workflow_runs,
    main,
)

pytestmark = pytest.mark.unit

WORKFLOW_PATH = Path(".github/workflows/deliver-dev-candidate-to-staging.yml")

#: The live newest completed run at the time the binding landed, as the REST API
#: returned it (fields trimmed to the ones the reader uses).
RUN_35832924275: dict[str, Any] = {
    "conclusion": "failure",
    "created_at": "2026-09-23T07:40:13Z",
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

D11 = ModelWorkflowVerdictRequest(
    repo="OmniNode-ai/omnimarket",
    workflow="delegation-regression-nightly.yml",
    branch="dev",
    max_age_hours=26,
    dispatch_title_contains="lane=stability-test",
)


def _run(
    run_id: int,
    *,
    conclusion: str = "success",
    finished: datetime = NOW - timedelta(hours=2),
    event: str = "schedule",
    title: str = "Delegation Regression (nightly) lane=stability-test",
    branch: str = "dev",
    status: str = "completed",
) -> dict[str, Any]:
    stamp = finished.strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": run_id,
        "conclusion": conclusion,
        "created_at": (finished - timedelta(minutes=40)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at": stamp,
        "event": event,
        "display_title": title,
        "head_branch": branch,
        "head_sha": "0" * 40,
        "status": status,
        "html_url": f"https://github.com/OmniNode-ai/omnimarket/actions/runs/{run_id}",
    }


def _judge(runs: list[dict[str, Any]]) -> tuple[int, str]:
    out = io.StringIO()
    code = judge_workflow_runs(runs, D11, NOW, out)
    return code, out.getvalue()


class TestJudge:
    def test_the_live_red_run_refuses_and_names_the_run(self) -> None:
        code, output = _judge([RUN_35832924275])
        assert code == 1
        assert "35832924275" in output
        assert "'failure'" in output
        assert "workflow verdict gate FAILED" in output

    def test_an_older_green_does_not_rescue_a_newer_red(self) -> None:
        older_green = _run(1, finished=NOW - timedelta(hours=20))
        code, output = _judge([RUN_35832924275, older_green])
        assert code == 1
        assert "35832924275" in output

    def test_a_fresh_success_admits(self) -> None:
        code, output = _judge([_run(42), RUN_35832924275])
        assert code == 0
        assert "workflow verdict gate PASSED" in output
        assert "run 42" in output

    def test_a_27_hour_old_success_is_stale(self) -> None:
        code, output = _judge([_run(7, finished=NOW - timedelta(hours=27))])
        assert code == 1
        assert "beyond the 26h bound" in output
        assert "7" in output

    @pytest.mark.parametrize(
        "conclusion",
        ["cancelled", "timed_out", "skipped", "neutral", "startup_failure", ""],
    )
    def test_every_non_success_conclusion_refuses(self, conclusion: str) -> None:
        code, _ = _judge([_run(9, conclusion=conclusion)])
        assert code == 1

    def test_no_runs_is_a_refusal_not_a_pass(self) -> None:
        code, output = _judge([])
        assert code == 1
        assert "An absent verdict is not a pass" in output

    def test_a_green_dev_lane_dispatch_cannot_launder_a_stability_red(self) -> None:
        dev_dispatch = _run(
            99,
            event="workflow_dispatch",
            title="Delegation Regression (nightly) lane=dev",
            finished=NOW - timedelta(minutes=10),
        )
        code, output = _judge([dev_dispatch, RUN_35832924275])
        assert code == 1
        assert "35832924275" in output
        assert "ignored" in output

    def test_a_stability_dispatch_is_a_new_measurement_and_counts(self) -> None:
        dispatch = _run(
            100,
            event="workflow_dispatch",
            title="Delegation Regression (nightly) lane=stability-test",
            finished=NOW - timedelta(minutes=10),
        )
        code, _ = _judge([dispatch, RUN_35832924275])
        assert code == 0

    def test_a_dispatch_with_no_lane_in_its_title_does_not_count(self) -> None:
        untitled = _run(
            101,
            event="workflow_dispatch",
            title="Delegation Regression (nightly)",
            finished=NOW - timedelta(minutes=10),
        )
        code, _ = _judge([untitled])
        assert code == 1

    def test_another_branch_does_not_count(self) -> None:
        code, _ = _judge([_run(5, branch="feature/x")])
        assert code == 1

    def test_an_in_progress_run_is_not_the_verdict(self) -> None:
        running = _run(6, status="in_progress", conclusion="")
        code, output = _judge([running, RUN_35832924275])
        assert code == 1
        assert "35832924275" in output

    def test_an_unreadable_finish_time_refuses(self) -> None:
        run = _run(8)
        run["updated_at"] = "yesterday"
        code, output = _judge([run])
        assert code == 1
        assert "unknown age" in output


class TestTransport:
    def test_the_read_is_scoped_to_completed_runs_on_the_branch(
        self, monkeypatch: Any
    ) -> None:
        seen: list[str] = []

        def fake(path: str) -> bytes:
            seen.append(path)
            return json.dumps({"workflow_runs": [RUN_35832924275]}).encode()

        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", fake)
        code = evaluate_workflow_verdict(D11, io.StringIO(), now=NOW)
        assert code == 1
        assert seen == [
            "repos/OmniNode-ai/omnimarket/actions/workflows/"
            "delegation-regression-nightly.yml/runs?branch=dev&status=completed"
            "&per_page=50"
        ]

    def test_an_unreadable_surface_refuses(self, monkeypatch: Any) -> None:
        def broken(path: str) -> bytes:
            msg = "`gh api` exited 1: HTTP 403"
            raise RuntimeError(msg)

        monkeypatch.setattr("scripts.ci.lab_pass_receipt._gh_api", broken)
        out = io.StringIO()
        assert evaluate_workflow_verdict(D11, out, now=NOW) == 1
        assert "unreadable" in out.getvalue()

    def test_a_body_with_no_run_list_refuses(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._gh_api", lambda _p: b'{"message": "x"}'
        )
        assert evaluate_workflow_verdict(D11, io.StringIO(), now=NOW) == 1

    def test_the_cli_refuses_the_live_red(self, monkeypatch: Any, capsys: Any) -> None:
        monkeypatch.setattr(
            "scripts.ci.lab_pass_receipt._gh_api",
            lambda _p: json.dumps({"workflow_runs": [RUN_35832924275]}).encode(),
        )
        code = main(
            [
                "workflow-verdict",
                "--repo",
                "OmniNode-ai/omnimarket",
                "--workflow",
                "delegation-regression-nightly.yml",
                "--branch",
                "dev",
                "--max-age-hours",
                "26",
                "--dispatch-title-contains",
                "lane=stability-test",
            ]
        )
        assert code == 1
        assert "35832924275" in capsys.readouterr().out


class TestRequest:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"repo": "omnimarket"},
            {"workflow": "delegation-regression-nightly"},
            {"branch": " "},
            {"max_age_hours": 0},
        ],
    )
    def test_a_malformed_request_is_refused(self, kwargs: dict[str, Any]) -> None:
        base: dict[str, Any] = {
            "repo": "OmniNode-ai/omnimarket",
            "workflow": "delegation-regression-nightly.yml",
            "branch": "dev",
            "max_age_hours": 26,
        }
        base.update(kwargs)
        with pytest.raises(ValueError):
            ModelWorkflowVerdictRequest(**base)

    def test_there_is_no_override_flag_on_the_verdict_reader(self) -> None:
        subparsers = next(
            action
            for action in build_parser()._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        options = {
            option
            for action in subparsers.choices["workflow-verdict"]._actions
            for option in action.option_strings
        }
        for banned in (
            "--force",
            "--skip",
            "--allow-missing",
            "--warn-only",
            "--allow-stale",
            "--allow-failure",
        ):
            assert banned not in options


class TestWiring:
    """The verdict is only a gate if the delivery workflow runs it on every path."""

    def _gate_job(self) -> dict[str, Any]:
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        job: dict[str, Any] = workflow["jobs"]["lab-pass-gate"]
        return job

    def _d11_step(self) -> dict[str, Any]:
        steps = self._gate_job()["steps"]
        matches = [s for s in steps if "workflow-verdict" in str(s.get("run", ""))]
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

    def test_the_d11_read_names_the_governed_verdict(self) -> None:
        run = str(self._d11_step()["run"])
        assert "--repo OmniNode-ai/omnimarket" in run
        assert "--workflow delegation-regression-nightly.yml" in run
        assert "--branch dev" in run
        assert "--max-age-hours 26" in run
        assert '--dispatch-title-contains "lane=stability-test"' in run
        assert "set -euo pipefail" in run

    def test_the_dispatch_still_needs_the_gate_job(self) -> None:
        workflow = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
        dispatch = workflow["jobs"]["dispatch-to-staging"]
        assert "lab-pass-gate" in dispatch["needs"]
        assert "if" not in dispatch
