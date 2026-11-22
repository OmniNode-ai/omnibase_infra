# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev build job a7474c42: a runtime tick is not a source event the apply gate grades.

The recorded incident
---------------------
Dev deploy job a7474c42 (``fixtures/dev_job_a7474c42/job_record.json``) failed
at the ``runtime`` phase because the container never read healthy:
``projection_apply_divergence: 1/95 projection(s) consumed at least 20
event(s) and upserted ZERO rows ... EvalResultsProjectionWriter``.

The container log excerpt (``container_log_excerpt.txt``) shows what the writer
"consumed": almost every dispatch is ``onex.intent.platform.runtime-tick.v1``,
the timer the writer subscribes to so it can sweep expired deadlines. A tick
that finds nothing to sweep writes zero rows by design, yet
``_record_projection_apply`` counted it as one consumed source event, so any
projection subscribed to the tick reads as "consumed and wrote nothing" the
moment its real source topic is quiet.

These tests replay the recorded log lines through the real recording seam and
the real divergence evaluation.

Related Tickets:
    - OMN-26419: this regression
    - OMN-18910: the apply-flow dimensions this feeds, unchanged here
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import _record_projection_apply
from omnibase_infra.runtime.health.projection_apply_flow import (
    APPLY_DIVERGENCE_MIN_CONSUMED,
    evaluate_projection_apply_flow,
    projection_apply_divergence_status,
)
from omnibase_infra.runtime.observability.projection_apply_counters import (
    get_projection_apply_counters,
    reset_projection_apply_counters_for_test,
)
from omnibase_infra.topics.platform_topic_suffixes import SUFFIX_RUNTIME_TICK

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "dev_job_a7474c42"

_ZERO_ROW_LINE = re.compile(
    r"handler=(?P<handler>\w+) topic=(?P<topic>\S+) "
    r"event_type=\S+ rows_upserted=(?P<rows>\d+)"
)


class EvalResultsProjectionWriter:
    """Stands in for the recorded handler: the seam keys on the class name."""


@pytest.fixture(autouse=True)
def _fresh_counters() -> None:
    reset_projection_apply_counters_for_test()


def _recorded_dispatches() -> list[tuple[str, int]]:
    """The (topic, rows_upserted) of every zero-row line in the recorded excerpt."""
    lines = (FIXTURES / "container_log_excerpt.txt").read_text().splitlines()
    parsed = [m for line in lines if (m := _ZERO_ROW_LINE.search(line))]
    assert len(parsed) == len(lines), "every recorded line is a zero-row dispatch"
    assert {m["handler"] for m in parsed} == {"EvalResultsProjectionWriter"}
    return [(m["topic"], int(m["rows"])) for m in parsed]


def _replay(dispatches: list[tuple[str, int]]) -> str:
    handler = EvalResultsProjectionWriter()
    for topic, rows in dispatches:
        _record_projection_apply(
            handler_instance=handler,
            topic=topic,
            rows_upserted=rows,
            rows_refused=0,
        )
    counters = get_projection_apply_counters()
    counters.close_window()
    verdict = evaluate_projection_apply_flow(
        windows=counters.retained_windows(),
        registered_projections=counters.registered_projections(),
    )
    return projection_apply_divergence_status(verdict)


def test_a7474c42_job_record_names_the_rule_and_step() -> None:
    record = json.loads((FIXTURES / "job_record.json").read_text())
    assert record["correlation_id"].startswith("a7474c42")
    assert record["status"] == "failed"
    assert record["phase_results"]["runtime"] == "failed"
    assert "FAIL [runtime_degraded]" in record["errors"][0]
    assert "projection_apply_divergence" in record["errors"][0]
    assert "EvalResultsProjectionWriter" in record["errors"][0]


def test_a7474c42_recorded_log_is_mostly_ticks_over_the_floor() -> None:
    dispatches = _recorded_dispatches()
    ticks = [d for d in dispatches if d[0] == SUFFIX_RUNTIME_TICK]
    assert len(ticks) >= APPLY_DIVERGENCE_MIN_CONSUMED
    assert all(rows == 0 for _, rows in dispatches)


def test_a7474c42_runtime_ticks_do_not_degrade_a_quiet_projection() -> None:
    assert _replay(_recorded_dispatches()) == "HEALTHY"


def test_a7474c42_ticks_leave_the_projection_in_scope() -> None:
    _replay(_recorded_dispatches())
    assert get_projection_apply_counters().registered_projections() == (
        "EvalResultsProjectionWriter",
    )


def test_a7474c42_source_events_that_write_nothing_still_degrade() -> None:
    """Positive control: the gate is not weakened, a real source topic still grades."""
    source_topic = next(
        topic for topic, _ in _recorded_dispatches() if topic != SUFFIX_RUNTIME_TICK
    )
    dispatches = [(source_topic, 0)] * APPLY_DIVERGENCE_MIN_CONSUMED
    assert _replay(dispatches) == "DEGRADED"


def test_a7474c42_divergence_floor_is_unchanged() -> None:
    assert APPLY_DIVERGENCE_MIN_CONSUMED == 20


def test_a7474c42_recorded_rule_clears_on_the_recorded_log() -> None:
    """The rule the job record names no longer fires on the log it was recorded with."""
    record = json.loads((FIXTURES / "job_record.json").read_text())
    assert "projection_apply_divergence" in record["errors"][0]
    assert _replay(_recorded_dispatches()) == "HEALTHY"


def test_a7474c42_gate_still_grades_source_events_while_ticks_are_excluded() -> None:
    """The floor, the source-topic grading and the tick exclusion hold together."""
    assert APPLY_DIVERGENCE_MIN_CONSUMED == 20
    source_topic = next(
        topic for topic, _ in _recorded_dispatches() if topic != SUFFIX_RUNTIME_TICK
    )
    assert _replay([(source_topic, 0)] * APPLY_DIVERGENCE_MIN_CONSUMED) == "DEGRADED"
    reset_projection_apply_counters_for_test()
    assert (
        _replay([(SUFFIX_RUNTIME_TICK, 0)] * (APPLY_DIVERGENCE_MIN_CONSUMED * 5))
        == "HEALTHY"
    )
