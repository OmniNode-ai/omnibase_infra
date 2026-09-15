# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18385 -- a redacted dead-letter body must not be replayed.

The dead-letter publisher now strips credential-named fields out of
``original_message.value`` before it is written, and records the field NAMES it
removed under ``redacted_fields``. That makes the stored body deliberately
INCOMPLETE, which is a direct consequence the replay path has to answer for:
republishing a gateway attach whose ``access_token`` has been replaced by a
marker sends a command that cannot authenticate, and it dead-letters again on
the next pass.

This is the same shape as the two refusals beside it (OMN-17896's unreadable
body, OMN-18084's dead-letter fixed point): the record stays durable and
reclassifiable on the quarantine sink, and the producer re-issues the command
with a fresh credential. The credential is gone by design and cannot be
recovered from the record -- that is the point of the redaction, not a defect
in it.

A record written BEFORE the publisher emitted the key carries no
``redacted_fields`` and is unaffected: those bodies were never redacted, so
refusing them would be a claim this guard cannot make.
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
from omnibase_infra.utils.util_dlq_credential_redaction import DLQ_REDACTION_MARKER

pytestmark = pytest.mark.unit

# onex-topic-allow: quotes the live topic whose records were dead-lettered
_ATTACH_TOPIC = "onex.cmd.omnibase-infra.gateway-attach-request.v1"


def _config() -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092",
        dlq_topic="onex.dlq.omnibase-infra.commands.v1",  # onex-topic-allow: the live DLQ carrying these records
    )


def _attach_body(*, redacted: bool) -> str:
    token = DLQ_REDACTION_MARKER if redacted else "OMN18385-SENTINEL-NOT-A-TOKEN"
    return json.dumps(
        {
            "correlation_id": str(uuid4()),
            "payload": {
                "access_token": token,
                "edge_instance_id": "edge-1",
            },
        }
    )


def _message(*, redacted: bool) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_ATTACH_TOPIC,
        original_value=_attach_body(redacted=redacted),
        correlation_id=uuid4(),
        error_type="TokenValidationError",
        dlq_offset=137,
        dlq_partition=0,
        redacted_fields=("payload.access_token",) if redacted else (),
    )


def test_a_redacted_body_is_not_eligible_for_replay() -> None:
    eligible, reason = should_replay(_message(redacted=True), _config())
    assert eligible is False
    assert "payload.access_token" in reason
    assert "OMN-18385" in reason


def test_an_unredacted_body_is_still_eligible() -> None:
    """The positive control: the guard must refuse only what it should.

    Without this, a guard that refused every record would pass the test above
    while silently stopping the replay path altogether.
    """
    eligible, reason = should_replay(_message(redacted=False), _config())
    assert eligible is True, reason


def test_a_legacy_record_without_the_key_parses_as_unredacted() -> None:
    """Every record written before this change carries no redaction record."""
    parsed = ModelDlqMessage.from_kafka_message(
        payload={
            "original_topic": _ATTACH_TOPIC,
            "original_message": {"value": _attach_body(redacted=False), "offset": 12},
            "correlation_id": str(uuid4()),
        },
        dlq_offset=12,
        dlq_partition=0,
    )
    assert parsed.redacted_fields == ()
    eligible, _reason = should_replay(parsed, _config())
    assert eligible is True


def test_the_redaction_record_is_read_off_a_live_shaped_envelope() -> None:
    parsed = ModelDlqMessage.from_kafka_message(
        payload={
            "original_topic": _ATTACH_TOPIC,
            "original_message": {"value": _attach_body(redacted=True), "offset": 137},
            "correlation_id": str(uuid4()),
            "redacted_fields": ["payload.access_token"],
        },
        dlq_offset=137,
        dlq_partition=0,
    )
    assert parsed.redacted_fields == ("payload.access_token",)
    eligible, _reason = should_replay(parsed, _config())
    assert eligible is False
