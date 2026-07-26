# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""DLQ / dispatch-drop failure classification (OMN-14492).

Before this enum, both a genuinely-unregistered dispatcher and a payload that
failed its type-scoped ``event_model.model_validate`` (OMN-12416) surfaced as
the identical "No dispatcher found" WARNING and ``NO_DISPATCHER`` dispatch
status, with no ``ValidationError`` detail anywhere in the DLQ envelope. An
operator could not tell "fix the publisher" from "fix the wiring" without a
multi-hour log dig (see OMN-14484, six root-cause theories for one malformed
harness payload).

This enum is intentionally infra-local and orthogonal to the canonical
``omnibase_core.enums.enum_dispatch_status.EnumDispatchStatus`` — it is NOT a
replacement for ``EnumDispatchStatus`` and does not require a core change.
Both ``PUBLISHER_MALFORMED`` and ``NO_DISPATCHER`` still resolve to
``EnumDispatchStatus.NO_DISPATCHER`` for existing status-based branching; this
enum answers the follow-up question "why", carried in
``ModelDispatchResult.error_details['failure_class']`` and threaded through
into the DLQ envelope's structured fields.
"""

from __future__ import annotations

from enum import Enum


class EnumDlqFailureClass(str, Enum):
    """Why a message was dropped/DLQ'd at the dispatch boundary.

    Values:
        PUBLISHER_MALFORMED: A dispatcher IS registered for the event_type,
            but the payload failed ``event_model.model_validate`` for every
            type-scoped candidate (OMN-12416 payload_type matcher rejected
            it). Fix = the PRODUCER/publisher. The real pydantic
            ``ValidationError`` detail is carried alongside this value.
        NO_DISPATCHER: No dispatcher is registered for the message type at
            all — no route matched by topic/category/message_type. Fix = the
            CONSUMER wiring/deploy.
        VALID_UNROUTED: The consumer exists in code but is not loaded on this
            lane/container (registered-but-unroutable, OMN-14488 territory).
            Fix = deploy/lane wiring, or nothing. The dispatch engine cannot
            mechanically distinguish this from ``NO_DISPATCHER`` on its own
            (it only knows what is registered in this process) — operators
            apply this label from deploy/registry context.
        CONSUMER_ERROR: A dispatcher WAS selected and invoked but raised
            during handling (distinct from routing failure).
    """

    PUBLISHER_MALFORMED = "publisher_malformed"
    """Payload failed event_model validation for every type-scoped candidate."""

    NO_DISPATCHER = "no_dispatcher"
    """No route/dispatcher registered for this topic+category+message_type."""

    VALID_UNROUTED = "valid_unrouted"
    """Consumer exists but isn't loaded on this lane/container."""

    CONSUMER_ERROR = "consumer_error"
    """A selected dispatcher raised during handling."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumDlqFailureClass"]
