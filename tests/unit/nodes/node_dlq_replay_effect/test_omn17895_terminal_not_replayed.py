# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17895 — a boundary failure terminal is quarantined, never replayed.

WHAT WAS MEASURED (dev lane ``omnibase-infra``, 2026-09-04, read-only). Of the
11,340 records on ``onex.evt.omnimarket.swarm-fanout-completed.v1``, **11,328**
carried ``x-replayed-by=node_dlq_replay_effect`` with ``x-replay-count`` 1-5.
Bucketing one burst by replay count gives 12 originals, then 24, 48, 96, 192,
384 — an exact doubling per round, because the record is a
``ModelBoundaryFailureTerminal`` that TWO downstream consumers each reject and
each DLQ, and ``replay_message`` republishes every DLQ record independently to
``message.original_topic``. Sum 2^0..2^5 = **63x amplification per terminal**,
and the cap at ``max_replay_count=5`` is the only reason it terminates.

The same signature was live on two unrelated topics in the same window —
``onex.evt.omnimarket.redeploy-completed.v1`` (high-watermark 6,378) and
``onex.evt.omnibase-infra.runtime-manifest-published.v1`` (3,824), both at
``x-replay-count=5`` under one correlation — so this is not swarm-specific.

WHY REPLAY IS THE WRONG VERB FOR THIS RECORD. Replay exists to re-attempt work
that might succeed on a second look. A boundary failure terminal is not work: it
is the runtime's own ANSWER that some work will never be attempted again. It
carries no request payload, so every consumer that dispatches its topic's
event_type fails ``model_validate`` on it deterministically — replaying it
cannot succeed on attempt 2 any more than on attempt 1, and each attempt
multiplies. ``handler_wiring._is_boundary_failure_terminal_record`` already
refuses to terminalize such a record for the same reason (OMN-17432, "a terminal
is an answer, not a request"); the DLQ leg was left unguarded.

Quarantine, not drop: ``onex.dlq.omnibase-infra.quarantine.v1`` is the durable
sink ``should_replay``'s other refusals already use (OMN-12619), so the record
stays reclassifiable instead of vanishing.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

_LIVE_TERMINAL_TOPIC = "onex.evt.omnimarket.swarm-fanout-completed.v1"  # onex-topic-allow: quotes the measured live record


def _config() -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic="onex.dlq.omnibase-infra.events.v1",  # onex-topic-allow: the live DLQ this node drains
    )


def _dlq_message(original_value: str, *, retry_count: int = 0) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_LIVE_TERMINAL_TOPIC,
        original_value=original_value,
        correlation_id=uuid4(),
        retry_count=retry_count,
        error_type="HandlerDispatchFailureError",
        dlq_offset=8223570,
        dlq_partition=0,
    )


def _boundary_terminal_envelope_json() -> str:
    """The envelope shape read off offsets 11337-11339 on the dev broker."""
    return json.dumps(
        {
            "payload_type": "ModelBoundaryFailureTerminal",
            "source_tool": "auto-wiring-boundary",
            "target_tool": _LIVE_TERMINAL_TOPIC,
            "event_type": "omnimarket.swarm-fanout-completed",
            "correlation_id": str(uuid4()),
            "payload": {
                "status": "failed",
                "failure_class": "HandlerDispatchFailureError",
                "retryable": True,
                "failure_reason": (
                    "no dispatcher for omnimarket.delegation-escalation-triggered"
                ),
                "origin_topic": "onex.evt.omnimarket.delegation-escalation-triggered.v1",  # onex-topic-allow: quotes the measured live record
            },
        }
    )


@pytest.mark.unit
def test_a_boundary_failure_terminal_is_not_eligible_for_replay() -> None:
    """RED before the fix: this record was replayed 5 times, doubling each round."""
    eligible, reason = should_replay(
        _dlq_message(_boundary_terminal_envelope_json()), _config()
    )
    assert eligible is False, (
        "the replay engine re-published a ModelBoundaryFailureTerminal onto its "
        "original topic — 11,328 of 11,340 live records on that topic carried "
        "x-replayed-by=node_dlq_replay_effect"
    )
    assert "terminal" in reason.lower(), reason


@pytest.mark.unit
def test_the_refusal_holds_on_the_very_first_attempt() -> None:
    """Not merely capped at 5 — refused at ``retry_count=0``.

    The cap already bounded the storm at 63x. The claim here is that the
    amplification never starts, so the burst size goes to zero rather than to
    2^0..2^4.
    """
    for retry_count in (0, 1, 4):
        eligible, _ = should_replay(
            _dlq_message(_boundary_terminal_envelope_json(), retry_count=retry_count),
            _config(),
        )
        assert eligible is False, retry_count


@pytest.mark.unit
def test_an_ordinary_record_is_still_replayed() -> None:
    """The guard is payload-typed, not a blanket refusal on the topic."""
    ordinary = json.dumps(
        {
            "payload_type": "ModelSwarmFanoutResult",
            "event_type": "omnimarket.swarm-fanout-completed",
            "correlation_id": str(uuid4()),
            "payload": {"dispatches": 3, "wall_latency_ms": 120},
        }
    )
    eligible, _ = should_replay(_dlq_message(ordinary), _config())
    assert eligible is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "original_value",
    ["", "not json at all", "[1, 2, 3]", "null"],
)
def test_an_unparseable_value_is_not_mistaken_for_a_terminal(
    original_value: str,
) -> None:
    """A value the guard cannot read is left to the existing predicates.

    The guard must add a refusal, never remove eligibility from records it
    cannot classify — that would be a silent behaviour change for every
    malformed record already in the DLQ.
    """
    eligible, _ = should_replay(_dlq_message(original_value), _config())
    assert eligible is True
