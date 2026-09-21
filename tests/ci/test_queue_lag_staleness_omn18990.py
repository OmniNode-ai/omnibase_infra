# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verify job refuses to size a wait from a stale queue count (OMN-18990).

WHY THE READER CHECKS THIS TOO
------------------------------
The agent refuses a stale sample at its own surface, and that is the primary
repair. This reader enforces the same bound because THIS is the side that does
the arithmetic: ``commands_ahead x mean service time`` is what decides whether
the wait is worth starting, and a reader that trusted any integer it was
handed would inherit the defect from any agent that had not yet self-updated.

Measured 2026-09-21, receipt artifact ``10632895303``. The payload read at
09:18:50Z reported ``commands_ahead=0`` from a sample taken before the running
rebuild began. The job derived a 1560s bound from it, watched for 26 minutes
and wrote FAIL. The agent reached the command at 10:01:09Z, 42m21s in.

THE THREE CASES, AND WHY THE THIRD IS NOT THE SECOND
-----------------------------------------------------
* age present and inside the bound -> read it, exactly as before;
* age present and beyond the bound -> UNREAD, naming the age;
* age ABSENT -> read it, exactly as before. An absent age is a deploy agent
  that predates this change, which is the previous behaviour by name and not
  a measurement anyone can see is out of date.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from ci.check_dev_lane_staleness import (
    MAX_QUEUE_LAG_AGE_SECONDS,
    read_agent_queue,
)

pytestmark = pytest.mark.unit

_AGENT = "http://agent:8098"
_OBSERVED = "2026-09-21T08:40:50+00:00"


def _opener(status: int, body: str):  # type: ignore[no-untyped-def]
    def _fetch(url: str, timeout: float) -> tuple[int, str]:
        return status, body

    return _fetch


def _payload(**overrides: object) -> str:
    payload: dict[str, object] = {
        "commands_ahead": 0,
        "store_depth": 0,
        "control_topic_lag": 0,
        "control_topic_lag_reason": "",
        "control_topic_lag_basis": "committed",
        "control_topic_lag_observed_at": _OBSERVED,
        "control_topic_lag_age_seconds": 5.0,
        "mean_service_time_seconds": 1402.2,
        "service_sample_size": 10,
        "in_flight_correlation_id": None,
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestAStaleCountIsNotRead:
    def test_the_incidents_own_payload_reads_unknown(self) -> None:
        """Fails on the pre-change tree, where this returns ``commands_ahead=0``."""
        facts = read_agent_queue(
            _AGENT, opener=_opener(200, _payload(control_topic_lag_age_seconds=2280.0))
        )
        assert facts.commands_ahead is None
        assert "2280s ago" in facts.unread_reason
        assert "does not poll while a rebuild runs" in facts.unread_reason

    def test_a_stale_nonzero_count_is_refused_too(self) -> None:
        facts = read_agent_queue(
            _AGENT,
            opener=_opener(
                200,
                _payload(commands_ahead=3, control_topic_lag_age_seconds=600.0),
            ),
        )
        assert facts.commands_ahead is None

    def test_the_bound_is_inclusive(self) -> None:
        inside = read_agent_queue(
            _AGENT,
            opener=_opener(
                200,
                _payload(
                    commands_ahead=1,
                    control_topic_lag_age_seconds=MAX_QUEUE_LAG_AGE_SECONDS,
                ),
            ),
        )
        outside = read_agent_queue(
            _AGENT,
            opener=_opener(
                200,
                _payload(
                    commands_ahead=1,
                    control_topic_lag_age_seconds=MAX_QUEUE_LAG_AGE_SECONDS + 1,
                ),
            ),
        )
        assert inside.commands_ahead == 1
        assert outside.commands_ahead is None

    def test_an_unreadable_age_is_refused_rather_than_ignored(self) -> None:
        facts = read_agent_queue(
            _AGENT, opener=_opener(200, _payload(control_topic_lag_age_seconds="soon"))
        )
        assert facts.commands_ahead is None
        assert "not an age" in facts.unread_reason


class TestAFreshCountStillSizesTheWait:
    """The positive control. Without it, refusing everything passes the class above."""

    def test_a_fresh_zero_is_still_a_zero(self) -> None:
        facts = read_agent_queue(_AGENT, opener=_opener(200, _payload()))
        assert facts.commands_ahead == 0
        assert facts.queue_position_at_start == 1
        assert facts.unread_reason == ""

    def test_a_fresh_count_still_derives_a_bound(self) -> None:
        facts = read_agent_queue(
            _AGENT,
            opener=_opener(
                200, _payload(commands_ahead=1, control_topic_lag_age_seconds=10.0)
            ),
        )
        assert facts.commands_ahead == 1
        assert (
            facts.derived_wait_bound_seconds(lane_budget_seconds=900, margin_seconds=0)
            is not None
        )


class TestAnAgentThatPredatesTheAgeFieldIsUnchanged:
    def test_an_absent_age_reads_exactly_as_before(self) -> None:
        """Absent is not stale. An old agent serves no age and is read as it was."""
        body = json.dumps(
            {
                "commands_ahead": 2,
                "store_depth": 1,
                "control_topic_lag": 1,
                "mean_service_time_seconds": 1000.0,
                "service_sample_size": 5,
                "in_flight_correlation_id": None,
            }
        )
        facts = read_agent_queue(_AGENT, opener=_opener(200, body))
        assert facts.commands_ahead == 2
        assert facts.unread_reason == ""

    def test_an_explicitly_null_age_reads_as_before(self) -> None:
        facts = read_agent_queue(
            _AGENT,
            opener=_opener(
                200,
                _payload(commands_ahead=2, control_topic_lag_age_seconds=None),
            ),
        )
        assert facts.commands_ahead == 2
