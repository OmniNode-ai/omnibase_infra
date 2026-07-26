# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-repo wire-shape seam-drift pin (OMN-15095).

``wire_shape.parse_delegation_terminal_envelope`` mirrors
``omninode_infra``'s ``docker/onex-api/workflow_terminal_consumer.
parse_terminal_envelope`` (OMN-15093) field-for-field. This module cannot
import that function directly (cross-repo, no shared package -- see
``wire_shape.py``'s module docstring), so it pins the exact literal source
snippet from OMN-15093's landed implementation (as read from
``omninode_infra`` at commit f262cfe8, 2026-07-25) that this mirror must
match. If ``workflow_terminal_consumer.py``'s wire-shape parsing changes,
this test's inline reference snippet -- and ``wire_shape.py`` -- must be
updated together.
"""

from __future__ import annotations

import uuid

from omnibase_infra.verification.workflow_receipt.wire_shape import (
    DELEGATION_COMPLETED_TOPIC,
    DELEGATION_FAILED_TOPIC,
    WorkflowReceiptReplayEnvelopeError,
    parse_delegation_terminal_envelope,
    status_for_topic,
)

# Literal reference copy of the wire shape
# workflow_terminal_consumer.parse_terminal_envelope expects, from
# omninode_infra docker/onex-api/workflow_terminal_consumer.py (OMN-15093,
# commit f262cfe8): a top-level "correlation_id" (falling back to
# payload["correlation_id"]) plus payload["model_used"/"total_tokens"/
# "latency_ms"].
_REFERENCE_ENVELOPE_SHAPE = {
    "correlation_id": "<uuid>",
    "payload": {
        "correlation_id": "<uuid>",
        "model_used": "<str>",
        "total_tokens": "<int>",
        "latency_ms": "<int>",
    },
}


def test_reference_shape_keys_are_exactly_what_the_mirror_parses() -> None:
    assert set(_REFERENCE_ENVELOPE_SHAPE.keys()) == {"correlation_id", "payload"}
    assert set(_REFERENCE_ENVELOPE_SHAPE["payload"].keys()) == {
        "correlation_id",
        "model_used",
        "total_tokens",
        "latency_ms",
    }


def test_status_for_topic_matches_omn_15093_topic_status_map() -> None:
    assert status_for_topic(DELEGATION_COMPLETED_TOPIC) == "completed"
    assert status_for_topic(DELEGATION_FAILED_TOPIC) == "failed"


def test_parse_reproduces_correlation_id_and_payload_fields() -> None:
    correlation_id = uuid.uuid4()
    envelope = {
        "correlation_id": str(correlation_id),
        "payload": {
            "correlation_id": str(correlation_id),
            "model_used": "glm-4.6",
            "total_tokens": 456,
            "latency_ms": 1234,
        },
    }

    result = parse_delegation_terminal_envelope(DELEGATION_COMPLETED_TOPIC, envelope)

    assert result == (correlation_id, "completed", "glm-4.6", 456, 1234)


def test_falls_back_to_payload_correlation_id_when_top_level_absent() -> None:
    correlation_id = uuid.uuid4()
    envelope = {
        "payload": {
            "correlation_id": str(correlation_id),
            "model_used": "glm-4.6",
            "total_tokens": 1,
            "latency_ms": 1,
        },
    }

    result = parse_delegation_terminal_envelope(DELEGATION_FAILED_TOPIC, envelope)

    assert result[0] == correlation_id
    assert result[1] == "failed"


def test_unrecognized_topic_fails_closed() -> None:
    try:
        parse_delegation_terminal_envelope("not.a.real.topic", {})
    except WorkflowReceiptReplayEnvelopeError as exc:
        assert "unrecognized topic" in str(exc)
    else:
        raise AssertionError("expected WorkflowReceiptReplayEnvelopeError")


def test_missing_payload_fails_closed_not_silent() -> None:
    try:
        parse_delegation_terminal_envelope(
            DELEGATION_COMPLETED_TOPIC, {"correlation_id": str(uuid.uuid4())}
        )
    except WorkflowReceiptReplayEnvelopeError as exc:
        assert "payload" in str(exc)
    else:
        raise AssertionError("expected WorkflowReceiptReplayEnvelopeError")
