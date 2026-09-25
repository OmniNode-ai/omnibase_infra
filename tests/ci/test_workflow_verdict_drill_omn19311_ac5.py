# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19311 AC5 -- the planted-red drill on the workflow-verdict reader.

AC5's literal falsifier ("a live delivery run refuses on the current D11 red
and names the run id") was already satisfied once, live, by delivery runs
35906201995 and 35909427274 refusing on nightly run 35832924275 (TERMINAL
2026-09-23T20:55:51Z lane=d11-block-r3). That proof is now STALE: D11 turned
GREEN at 2026-09-24T05:33:57Z (TERMINAL lane=d11-fix-18349-r2, D11 dispatch
run 35959394254 concluded success) -- there is no live red left to prove a
refusal against, and OMN-18349/OMN-18626 landing was always going to remove
it (plan row 13's own premise). This module is the mechanism that keeps AC5
provable after the real red is gone: a self-expiring switch that makes the
verdict reader RETURN a red, so staging delivery's refusal path is exercised
for real, without depending on a real defect existing at proof time.

Design constraints, tested below:
  * ONE DIRECTION ONLY. The drill can force a red; there is no drill flag
    that forces a green. A misuse of this mechanism can only make delivery
    MORE conservative, never less.
  * SELF-EXPIRING. Past its own `until` timestamp, the drill has NO effect at
    all -- no operator action turns it off. This is what makes it safe to
    trigger and forget.
  * BOUNDED. `until` may not be set more than DRILL_MAX_WINDOW_HOURS in the
    future, so a mistyped date cannot leave delivery closed for days.
  * FAILS CLOSED ON MISCONFIGURATION. A ticket with no expiry, an expiry with
    no ticket, an unparseable timestamp, or a malformed ticket id all refuse
    -- a broken drill flag must never silently do nothing AND must never
    silently force red forever.

The positive controls below are the reason a green here means anything.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from scripts.ci.lab_pass_receipt import (
    DRILL_MAX_WINDOW_HOURS,
    evaluate_workflow_verdict,
)

pytestmark = pytest.mark.unit

REPO = "OmniNode-ai/omnimarket"
WORKFLOW = "delegation-regression-nightly.yml"
EVENTS = ("schedule", "workflow_dispatch")
NOW = datetime(2026, 9, 24, 18, 0, 0, tzinfo=UTC)


def _green_run(run_id: int = 1) -> dict[str, Any]:
    stamp = (NOW - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": run_id,
        "conclusion": "success",
        "created_at": stamp,
        "run_started_at": stamp,
        "run_attempt": 1,
        "event": "schedule",
        "display_title": "Delegation Regression (nightly)",
        "head_branch": "dev",
        "head_sha": "0" * 40,
        "status": "completed",
        "html_url": f"https://github.com/{REPO}/actions/runs/{run_id}",
    }


def _evaluate(
    monkeypatch: Any,
    *,
    runs: list[dict[str, Any]] | None = None,
    now: datetime = NOW,
    drill_red_until: str = "",
    drill_ticket: str = "",
) -> tuple[int, str]:
    monkeypatch.setattr(
        "scripts.ci.lab_pass_receipt._gh_api",
        lambda _path: json.dumps({"workflow_runs": runs or [_green_run()]}).encode(),
    )
    out = io.StringIO()
    code = evaluate_workflow_verdict(
        REPO,
        WORKFLOW,
        "dev",
        26,
        EVENTS,
        out,
        now=now,
        drill_red_until=drill_red_until,
        drill_ticket=drill_ticket,
    )
    return code, out.getvalue()


class TestDrillInactiveByDefault:
    def test_no_drill_flags_is_ordinary_evaluation(self, monkeypatch: Any) -> None:
        """Positive control: a real green run passes when the drill is unset."""
        code, output = _evaluate(monkeypatch)
        assert code == 0
        assert "DRILL" not in output


class TestDrillForcesRed:
    def test_active_drill_refuses_even_though_the_real_run_is_green(
        self, monkeypatch: Any
    ) -> None:
        until = (NOW + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        code, output = _evaluate(
            monkeypatch,
            drill_red_until=until,
            drill_ticket="OMN-19311",
        )
        assert code == 1
        assert "DRILL" in output
        assert "OMN-19311" in output

    def test_the_drill_banner_names_its_own_expiry(self, monkeypatch: Any) -> None:
        until = (NOW + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        _code, output = _evaluate(
            monkeypatch, drill_red_until=until, drill_ticket="OMN-19311"
        )
        assert until in output


class TestDrillSelfExpires:
    def test_an_expired_drill_has_no_effect_and_the_real_green_passes(
        self, monkeypatch: Any
    ) -> None:
        """Past `until`, no operator action is needed -- the drill just stops."""
        until = (NOW - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        code, output = _evaluate(
            monkeypatch, drill_red_until=until, drill_ticket="OMN-19311"
        )
        assert code == 0
        assert "DRILL" not in output or "expired" in output.lower()

    def test_an_expired_drill_still_lets_a_real_red_refuse_for_its_own_reason(
        self, monkeypatch: Any
    ) -> None:
        until = (NOW - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        red_run = dict(_green_run(run_id=2), conclusion="failure")
        code, output = _evaluate(
            monkeypatch,
            runs=[red_run],
            drill_red_until=until,
            drill_ticket="OMN-19311",
        )
        assert code == 1
        assert "'failure'" in output


class TestDrillFailsClosedOnMisconfiguration:
    def test_ticket_without_expiry_refuses(self, monkeypatch: Any) -> None:
        code, output = _evaluate(monkeypatch, drill_ticket="OMN-19311")
        assert code == 1
        assert "drill" in output.lower()

    def test_expiry_without_ticket_refuses(self, monkeypatch: Any) -> None:
        until = (NOW + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        code, output = _evaluate(monkeypatch, drill_red_until=until)
        assert code == 1
        assert "drill" in output.lower()

    def test_malformed_ticket_refuses(self, monkeypatch: Any) -> None:
        until = (NOW + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        code, _output = _evaluate(
            monkeypatch, drill_red_until=until, drill_ticket="not-a-ticket"
        )
        assert code == 1

    def test_unparseable_expiry_refuses(self, monkeypatch: Any) -> None:
        code, _output = _evaluate(
            monkeypatch, drill_red_until="not-a-date", drill_ticket="OMN-19311"
        )
        assert code == 1

    def test_window_wider_than_the_ceiling_refuses(self, monkeypatch: Any) -> None:
        too_far = NOW + timedelta(hours=DRILL_MAX_WINDOW_HOURS + 1)
        code, output = _evaluate(
            monkeypatch,
            drill_red_until=too_far.strftime("%Y-%m-%dT%H:%M:%SZ"),
            drill_ticket="OMN-19311",
        )
        assert code == 1
        assert str(DRILL_MAX_WINDOW_HOURS) in output or "ceiling" in output.lower()

    def test_window_at_the_ceiling_is_accepted(self, monkeypatch: Any) -> None:
        """Boundary check: exactly at the ceiling is armed, not refused-as-too-wide."""
        at_ceiling = NOW + timedelta(hours=DRILL_MAX_WINDOW_HOURS)
        code, output = _evaluate(
            monkeypatch,
            drill_red_until=at_ceiling.strftime("%Y-%m-%dT%H:%M:%SZ"),
            drill_ticket="OMN-19311",
        )
        assert code == 1
        assert "DRILL" in output


class TestDrillCannotForceGreen:
    def test_no_drill_parameter_can_force_a_pass_on_a_real_red(
        self, monkeypatch: Any
    ) -> None:
        """There is no green-forcing input at all -- only a red-forcing one."""
        red_run = dict(_green_run(run_id=3), conclusion="failure")
        # An expired drill window is the only way a drill call ever reaches
        # real evaluation with the real run data; confirm it still refuses.
        until = (NOW - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        code, _output = _evaluate(
            monkeypatch,
            runs=[red_run],
            drill_red_until=until,
            drill_ticket="OMN-19311",
        )
        assert code == 1

    def test_the_function_signature_carries_no_force_green_parameter(self) -> None:
        import inspect

        params = set(inspect.signature(evaluate_workflow_verdict).parameters)
        assert not any("green" in p.lower() for p in params)


class TestDrillWiring:
    """The drill is only live if the sender workflow can actually pass it."""

    def _workflow(self) -> dict[Any, Any]:
        from pathlib import Path

        import yaml

        path = Path(".github/workflows/deliver-dev-candidate-to-staging.yml")
        loaded: dict[Any, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded

    def _d11_step(self) -> dict[str, Any]:
        job = self._workflow()["jobs"]["lab-pass-gate"]
        matches = [
            s
            for s in job["steps"]
            if "delegation-regression-nightly.yml" in str(s.get("run", ""))
        ]
        assert len(matches) == 1
        step: dict[str, Any] = matches[0]
        return step

    def test_workflow_dispatch_declares_both_drill_inputs_empty_by_default(
        self,
    ) -> None:
        # PyYAML parses the `on:` mapping key as the boolean True in YAML 1.1,
        # so the workflow's own top-level key is read back as True, not "on".
        inputs = self._workflow()[True]["workflow_dispatch"]["inputs"]
        assert inputs["drill_red_until"]["default"] == ""
        assert inputs["drill_red_until"]["required"] is False
        assert inputs["drill_ticket"]["default"] == ""
        assert inputs["drill_ticket"]["required"] is False

    def test_the_d11_step_only_appends_drill_flags_when_drill_ticket_is_set(
        self,
    ) -> None:
        run = str(self._d11_step()["run"])
        assert "DRILL_TICKET" in run
        assert "--drill-red-until" in run
        assert "--drill-ticket" in run
        assert 'if [ -n "${DRILL_TICKET:-}" ]' in run

    def test_the_d11_step_still_names_every_existing_argument(self) -> None:
        """The drill wiring must not have displaced the ordinary arguments."""
        run = str(self._d11_step()["run"])
        assert "--repo OmniNode-ai/omnimarket" in run
        assert "--branch dev" in run
        assert "--max-age-hours 26" in run
        assert "--event schedule" in run
        assert "--event workflow_dispatch" in run
        assert '--dispatch-title-contains "lane=stability-test"' in run
        assert "set -euo pipefail" in run
