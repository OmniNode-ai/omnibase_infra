# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The scheduled DLQ monitor must name what it goes red for (OMN-18088).

WHY THIS TESTS THE WORKFLOW'S OWN BYTES
---------------------------------------
The defect was a claim and its evidence living in two different places. The
``Post summary`` step asserted "the failing step's log names each offending
topic with its arrival count, its bound, and its retained depth" while the
gating decision was a ``RuntimeHostError`` raised inside the node -- swallowed
by ``RuntimeLocal``, dropped by receipt mode, and never written anywhere. The
summary was confidently wrong on every alerting run, and nothing could tell,
because no test read what the run actually emitted.

So this file does not re-implement the renderer. It EXTRACTS the Python the
workflow ships, in the heredoc it ships it in, and runs that against captured
payloads. A renderer that stops naming offenders fails here even if a
copy-of-a-copy in a test fixture still would have passed.

The payloads are the two live runs at head ``1eef12c79`` / ``0b2749b9a``:
``34387859397`` (``suppress_alert_exit=true``, 67 topics, 3 alerting) and
``34399417506`` (the scheduled alert run whose result was all nulls).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "dlq-depth-monitor.yml"
)
RESULT_FILE = "dlq-monitor-result.json"

_EVENTS = "onex.dlq.omnibase-infra.events.v1"
_COMMANDS = "onex.dlq.omnibase-infra.commands.v1"
_QUARANTINE = "onex.dlq.omnibase-infra.quarantine.v1"
_QUIET = "onex.dlq.omnimarket.node-aislop-sweep.v1"


def _extract_gate_script() -> str:
    """Pull the renderer out of the workflow's ``<<'PY' ... PY`` heredoc."""
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    opens = [i for i, line in enumerate(lines) if line.strip().endswith("<<'PY'")]
    assert len(opens) == 1, (
        f"expected exactly one embedded Python heredoc in {WORKFLOW.name}, "
        f"found {len(opens)} — this test extracts by shape, so a second one "
        f"makes the extraction ambiguous rather than silently wrong."
    )
    start = opens[0] + 1
    closes = [i for i in range(start, len(lines)) if lines[i].strip() == "PY"]
    assert closes, "unterminated heredoc"
    return textwrap.dedent("\n".join(lines[start : closes[0]]))


def _verdict(
    topic: str,
    *,
    arrivals: int,
    retained: int,
    bound: int = 0,
    verdict: str = "alert_arrivals",
) -> dict[str, Any]:
    return {
        "topic": topic,
        "partition_count": 1,
        "log_start_offset": 0,
        "high_watermark": retained,
        "window_start_offset": 0,
        "retained_depth": retained,
        "arrivals_in_window": arrivals,
        "arrivals_per_minute": arrivals / 30,
        "max_arrivals_per_window": bound,
        "override_reason": "",
        "verdict": verdict,
        "window_seconds": 1800,
        "evaluated_at": "2026-09-09T18:15:32.848927Z",
    }


def _monitor_payload(*, alerting: bool, suppressed: bool = False) -> dict[str, Any]:
    """A ModelSkillResult carrying a real ModelDlqDepthMonitorResult."""
    verdicts = [_verdict(_QUIET, arrivals=0, retained=4_000, verdict="ok")]
    if alerting:
        verdicts = [
            _verdict(_EVENTS, arrivals=250_382, retained=2_308_145),
            _verdict(_COMMANDS, arrivals=70, retained=1_535),
            _verdict(_QUARANTINE, arrivals=65, retained=970_546),
            *verdicts,
        ]
    return {
        "skill_name": "node_dlq_depth_monitor_effect",
        "status": "success",
        "exit_code": 0,
        "result_model": (
            "omnibase_infra.nodes.node_dlq_depth_monitor_effect.models."
            "model_dlq_depth_monitor_result.ModelDlqDepthMonitorResult"
        ),
        "result": {
            "correlation_id": "52662dbf-8e04-4ac3-aff5-e58461bed4a7",
            "evaluated_at": "2026-09-09T18:15:32.848927Z",
            "window_seconds": 1800,
            "topics_matched": 67,
            "suppress_alert_exit": suppressed,
            "alert_exit_requested": alerting and not suppressed,
            "evaluation": {
                "correlation_id": "52662dbf-8e04-4ac3-aff5-e58461bed4a7",
                "evaluated_at": "2026-09-09T18:15:32.848927Z",
                "window_seconds": 1800,
                "verdicts": verdicts,
                "topics_observed": 67,
                "topics_alerting": 3 if alerting else 0,
                "total_arrivals_in_window": 250_517 if alerting else 0,
                "total_retained_depth": 3_280_496,
                "alert_triggered": alerting,
            },
        },
    }


# The exact shape run 34399417506 emitted: red, and saying nothing.
_NULL_PAYLOAD_FROM_RUN_34399417506 = {
    "skill_name": "node_dlq_depth_monitor_effect",
    "status": "failed",
    "exit_code": 1,
    "result_model": (
        "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
    ),
    "result": {
        "workflow_result": "failed",
        "exit_code": 1,
        "terminal_payload": None,
        "handler_result": None,
        "error": "",
        "capture_log": "RuntimeLocal: result=failed\n",
    },
}


def _run_gate(tmp_path: Path, file_body: str | None) -> tuple[int, str]:
    """Run the workflow's renderer in ``tmp_path``; return (exit code, summary)."""
    if file_body is not None:
        (tmp_path / RESULT_FILE).write_text(file_body, encoding="utf-8")
    summary = tmp_path / "step-summary.md"
    summary.touch()
    completed = subprocess.run(
        [sys.executable, "-c", _extract_gate_script()],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_STEP_SUMMARY": str(summary)},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "Traceback" not in completed.stderr, completed.stderr
    return completed.returncode, summary.read_text(encoding="utf-8")


class TestTheAlertingRunNamesItsOffenders:
    """AC1 — red, with every offending topic and its depth on the page."""

    def test_alerting_run_exits_non_zero(self, tmp_path: Path) -> None:
        code, _ = _run_gate(tmp_path, json.dumps(_monitor_payload(alerting=True)))

        assert code == 1

    def test_alerting_run_names_every_offender_with_its_depth(
        self, tmp_path: Path
    ) -> None:
        _, summary = _run_gate(tmp_path, json.dumps(_monitor_payload(alerting=True)))

        for topic, arrivals, retained in (
            (_EVENTS, "250382", "2308145"),
            (_COMMANDS, "70", "1535"),
            (_QUARANTINE, "65", "970546"),
        ):
            assert topic in summary
            assert arrivals in summary
            assert retained in summary

    def test_a_quiet_topic_is_not_reported_as_an_offender(self, tmp_path: Path) -> None:
        """POSITIVE CONTROL for the filter: naming everything is also wrong."""
        _, summary = _run_gate(tmp_path, json.dumps(_monitor_payload(alerting=True)))

        assert "3 DLQ sink(s) exceeded a declared bound" in summary
        assert _QUIET not in summary


class TestTheGreenPaths:
    """AC2 — the characterization run and the clean run both stay green."""

    def test_suppressed_run_stays_green_and_still_reports_the_breach(
        self, tmp_path: Path
    ) -> None:
        code, summary = _run_gate(
            tmp_path, json.dumps(_monitor_payload(alerting=True, suppressed=True))
        )

        assert code == 0
        assert _EVENTS in summary
        assert "does NOT gate this run" in summary

    def test_clean_run_is_green_and_says_so(self, tmp_path: Path) -> None:
        code, summary = _run_gate(
            tmp_path, json.dumps(_monitor_payload(alerting=False))
        )

        assert code == 0
        assert "No DLQ sink exceeded its bound this window." in summary


class TestEveryUnreadableCaseFailsClosed:
    """A sweep that did not happen must never read as a clean bill of health."""

    def test_the_null_payload_from_run_34399417506_is_named_as_a_malfunction(
        self, tmp_path: Path
    ) -> None:
        code, summary = _run_gate(
            tmp_path, json.dumps(_NULL_PAYLOAD_FROM_RUN_34399417506)
        )

        assert code == 1
        assert "THE PROBE DID NOT COMPLETE" in summary
        # The distinction the old summary could not make: this red is a
        # malfunction, not a DLQ breach, and saying so is the whole point.
        assert "not a DLQ alert" in summary

    def test_absent_result_file_fails_closed(self, tmp_path: Path) -> None:
        code, summary = _run_gate(tmp_path, None)

        assert code == 1
        assert "NO RESULT" in summary

    def test_empty_result_file_fails_closed(self, tmp_path: Path) -> None:
        code, summary = _run_gate(tmp_path, "\n")

        assert code == 1
        assert "NO RESULT" in summary

    def test_unparseable_result_fails_closed(self, tmp_path: Path) -> None:
        code, summary = _run_gate(tmp_path, "Traceback (most recent call last):")

        assert code == 1
        assert "UNREADABLE RESULT" in summary

    def test_a_monitor_result_missing_the_gate_field_fails_closed(
        self, tmp_path: Path
    ) -> None:
        payload = _monitor_payload(alerting=True)
        del payload["result"]["alert_exit_requested"]

        code, summary = _run_gate(tmp_path, json.dumps(payload))

        assert code == 1
        assert "MALFORMED RESULT" in summary


class TestTheWorkflowStillCarriesTheGate:
    """Structural pins, so the step cannot be quietly reverted to prose."""

    def test_the_gate_step_runs_on_always(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")

        assert "- name: Name the offenders and gate the run" in text
        assert "alert_exit_requested" in text

    def test_the_node_no_longer_gates_by_raising(self) -> None:
        """POSITIVE CONTROL: the handler is where the old gate lived."""
        handler = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_dlq_depth_monitor_effect"
            / "handlers"
            / "handler_dlq_depth_monitor.py"
        )
        source = handler.read_text(encoding="utf-8")

        assert "DLQ arrival alert" in source, (
            "the offender-list message is the positive control for this "
            "matcher; if it moved, this assertion is checking nothing."
        )
        alert_block = source.split("DLQ arrival alert")[1][:400]
        assert "raise" not in alert_block
