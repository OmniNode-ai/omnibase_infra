# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared utility for extracting correlation_id and raw_payload from dispatcher envelopes.

The dispatch engine materializes envelopes to dicts before calling dispatchers
(serialization boundary). All dispatchers need to handle both ModelEventEnvelope
objects and materialized dicts — this helper centralizes that logic.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope


def extract_envelope_fields(
    envelope: ModelEventEnvelope[object] | dict[str, object],
) -> tuple[UUID, object]:
    """Extract correlation_id and raw_payload from an envelope.

    Handles both ModelEventEnvelope objects and materialized dicts
    from the dispatch engine (serialization boundary).

    Args:
        envelope: Event envelope or materialized dict.
            Dict format: ``{"payload": {...}, "__debug_trace": {...}}``

    Returns:
        Tuple of (correlation_id, raw_payload).
    """
    if isinstance(envelope, dict):
        debug_trace = envelope.get("__debug_trace", {})
        raw_corr = (
            debug_trace.get("correlation_id") if isinstance(debug_trace, dict) else None
        )
        try:
            correlation_id = UUID(raw_corr) if raw_corr else uuid4()
        except ValueError:
            correlation_id = uuid4()
        raw_payload = envelope.get("payload", {})
    else:
        correlation_id = envelope.correlation_id or uuid4()
        raw_payload = envelope.payload
    return correlation_id, raw_payload
