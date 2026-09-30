# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19937: board probe results become durable, keyed events."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_board_probe_effect.contract_topics import (
    board_probe_result_topic,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_board_probe_result_publisher import (
    HandlerBoardProbeResultPublisher,
)
from omnibase_infra.nodes.node_board_probe_effect.models import (
    EnumBoardCheckId,
    EnumBoardCheckSurfaceClass,
    EnumBoardProbeOutcome,
    EnumBoardSubjectKind,
    ModelBoardProbeResult,
    ModelBoardProbeResultEvent,
    board_probe_result_event_from,
)

pytestmark = [pytest.mark.unit]

FINISHED_AT = datetime(2026, 9, 28, 20, 15, tzinfo=UTC)
CHAIN_CANARY_CONTRACT = (
    Path(__file__).resolve().parents[4]
    / "src/omnibase_infra/nodes/node_chain_canary_effect/contract.yaml"
)


def _event(**overrides: object) -> ModelBoardProbeResultEvent:
    fields: dict[str, object] = {
        "check_id": "forwarder_refused_topic",
        "subject_kind": EnumBoardSubjectKind.LANE,
        "subject": "dev",
        "repo": "OmniNode-ai/omnibase_infra",
        "sha": "a" * 40,
        "surface_instance": "compose-dev",
        "execution_id": "run-123",
        "outcome": EnumBoardProbeOutcome.FAIL,
        "reasons": ("forwarder refused one topic",),
        "evidence_items": ("onex.cmd.github.webhook-delivery.v1",),
        "finished_at": FINISHED_AT,
    }
    fields.update(overrides)
    return ModelBoardProbeResultEvent.model_validate(fields)


def test_event_refuses_an_empty_execution_id() -> None:
    with pytest.raises(ValidationError, match="execution_id"):
        _event(execution_id="")


def test_event_refuses_a_naive_finished_at() -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        _event(finished_at=datetime(2026, 9, 28, 20, 15))


def test_event_key_and_partition_key_are_stable() -> None:
    event = _event()

    assert event.key == (
        "forwarder_refused_topic",
        EnumBoardSubjectKind.LANE,
        "OmniNode-ai/omnibase_infra",
        "a" * 40,
        "compose-dev",
        "run-123",
    )
    assert event.partition_key == "dev"


def test_board_probe_result_mapper_preserves_probe_evidence() -> None:
    result = ModelBoardProbeResult(
        check_id=EnumBoardCheckId.FORWARDER_REFUSED_TOPIC,
        surface_class=EnumBoardCheckSurfaceClass.LAB_HARDWARE,
        subject="dev",
        outcome=EnumBoardProbeOutcome.FAIL,
        reasons=("forwarder refused one topic",),
        evidence_items=("onex.cmd.github.webhook-delivery.v1",),
        observed_at=FINISHED_AT,
    )

    event = board_probe_result_event_from(
        result,
        subject_kind=EnumBoardSubjectKind.LANE,
        repo="OmniNode-ai/omnibase_infra",
        sha="a" * 40,
        surface_instance="compose-dev",
        execution_id="run-123",
        finished_at=FINISHED_AT,
    )

    assert event.check_id == result.check_id.value
    assert event.subject == result.subject
    assert event.outcome is result.outcome
    assert event.reasons == result.reasons
    assert event.evidence_items == result.evidence_items


def test_result_topic_is_read_from_the_contract() -> None:
    expected = "onex.evt.omnibase-infra.board-probe-result.v1"

    assert board_probe_result_topic() == expected
    assert board_probe_result_topic(CHAIN_CANARY_CONTRACT) == expected


def test_publisher_returns_the_event_through_the_effect_output_path() -> None:
    event = _event()

    output = asyncio.run(HandlerBoardProbeResultPublisher().handle(event))

    assert output.events == (event,)
