# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Delegation-terminal wire-shape mirror (OMN-15095).

``omninode_infra``'s ``docker/onex-api/workflow_terminal_consumer.py``
(OMN-15093) owns the real parsing of the two delegation-orchestrator terminal
topics -- ``parse_terminal_envelope(topic, envelope) -> (correlation_id,
new_status, model_used, total_tokens, latency_ms)``. That module cannot be
imported here: ``omninode_infra`` is an application service repo, not a
published library, and this repo (``omnibase_infra``) is a separate,
independently-deployed package. This function is therefore a field-for-field
WIRE-CONTRACT MIRROR of that parser, not a second independent design -- any
change to the wire shape on either side must land here too. A cross-repo
seam-drift test (``test_wire_shape_matches_terminal_consumer.py``) pins the
exact field names/return order against a literal copy of the OMN-15093
docstring so drift is caught mechanically, not by convention.

Follow-up: this hash/wire duplication is exactly the class of thing
``omnibase_compat`` (per CLAUDE.md: "Thin shared structural package for
cross-repo enums, wire DTOs, event envelopes, primitives") exists to solve.
Moving both this parser and ``workflow_receipt_renderer``'s hash helper into
``omnibase_compat`` is real follow-up work, tracked separately rather than
done inline here (out of scope for a replay-verifier ticket).
"""

from __future__ import annotations

import uuid

from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_DELEGATION_COMPLETED,
    SUFFIX_DELEGATION_FAILED,
)

# Re-exported under this module's own names -- the canonical topic-registry
# constants are SUFFIX_DELEGATION_COMPLETED/SUFFIX_DELEGATION_FAILED
# (src/omnibase_infra/topics/platform_topic_suffixes.py); this module never
# hardcodes the literal topic strings itself.
DELEGATION_COMPLETED_TOPIC = SUFFIX_DELEGATION_COMPLETED
DELEGATION_FAILED_TOPIC = SUFFIX_DELEGATION_FAILED

_TOPIC_STATUS: dict[str, str] = {
    DELEGATION_COMPLETED_TOPIC: "completed",
    DELEGATION_FAILED_TOPIC: "failed",
}


class WorkflowReceiptReplayEnvelopeError(RuntimeError):
    """Raised when a replayed delegation-terminal message cannot be parsed.

    Mirrors ``workflow_terminal_consumer.WorkflowTerminalConsumeError`` --
    fails closed rather than silently dropping a malformed record.
    """


def status_for_topic(topic: str) -> str:
    """Return 'completed' or 'failed' for a delegation-terminal topic.

    Raises:
        WorkflowReceiptReplayEnvelopeError: topic is not one of the two real
            delegation-terminal topics.
    """
    try:
        return _TOPIC_STATUS[topic]
    except KeyError as exc:
        raise WorkflowReceiptReplayEnvelopeError(
            f"replay verifier given an unrecognized topic {topic!r}; "
            f"expected one of {sorted(_TOPIC_STATUS)}"
        ) from exc


def parse_delegation_terminal_envelope(
    topic: str, envelope: dict[str, object]
) -> tuple[uuid.UUID, str, str, int, int]:
    """Parse one delegation-terminal wire envelope replayed off Kafka.

    Field-for-field mirror of
    ``workflow_terminal_consumer.parse_terminal_envelope`` (OMN-15093):
    the wire shape is a top-level ``correlation_id`` (falling back to
    ``payload["correlation_id"]``) plus ``payload["model_used"/
    "total_tokens"/"latency_ms"]``.

    Returns:
        ``(correlation_id, status, model_used, total_tokens, latency_ms)``.

    Raises:
        WorkflowReceiptReplayEnvelopeError: unknown topic, or the envelope is
            missing/malformed the fields this function depends on.
    """
    status = status_for_topic(topic)

    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise WorkflowReceiptReplayEnvelopeError(
            f"delegation terminal envelope on {topic!r} is missing a 'payload' dict"
        )

    raw_correlation_id = envelope.get("correlation_id") or payload.get("correlation_id")
    if raw_correlation_id is None:
        raise WorkflowReceiptReplayEnvelopeError(
            f"delegation terminal envelope on {topic!r} has no correlation_id "
            f"at the envelope OR payload level"
        )
    try:
        correlation_id = uuid.UUID(str(raw_correlation_id))
    except (ValueError, AttributeError, TypeError) as exc:
        raise WorkflowReceiptReplayEnvelopeError(
            f"delegation terminal envelope on {topic!r} has a non-UUID "
            f"correlation_id: {raw_correlation_id!r}"
        ) from exc

    try:
        model_used = str(payload["model_used"])
        total_tokens = int(payload["total_tokens"])
        latency_ms = int(payload["latency_ms"])
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkflowReceiptReplayEnvelopeError(
            f"delegation terminal envelope on {topic!r} payload is missing "
            f"or has a malformed model_used/total_tokens/latency_ms: {exc}"
        ) from exc

    return correlation_id, status, model_used, total_tokens, latency_ms


__all__ = [
    "DELEGATION_COMPLETED_TOPIC",
    "DELEGATION_FAILED_TOPIC",
    "WorkflowReceiptReplayEnvelopeError",
    "parse_delegation_terminal_envelope",
    "status_for_topic",
]
