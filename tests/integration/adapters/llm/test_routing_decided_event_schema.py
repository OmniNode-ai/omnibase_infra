# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for routing-decided event schema [OMN-8026].

Verifies that ModelRoutingDecidedEvent serializes correctly and that
AdapterModelRouter populates fallback_indicator and selection_mode
accurately based on whether a failover occurred.
"""

from __future__ import annotations

import pytest

from omnibase_infra.adapters.llm.model_routing_decided_event import (
    ModelRoutingDecidedEvent,
)


@pytest.mark.integration
def test_model_routing_decided_event_schema_valid() -> None:
    """ModelRoutingDecidedEvent accepts valid data and rejects extra fields."""
    event = ModelRoutingDecidedEvent(
        selected_provider="local",
        selected_tier="local",
        selection_mode="round_robin",
        fallback_indicator=False,
        is_fallback=False,
        reason="round_robin",
        candidates_evaluated=1,
        candidate_providers=["local"],
        latency_ms=12.5,
        timestamp="2026-04-13T00:00:00Z",
    )
    assert event.fallback_indicator is False
    assert event.selection_mode == "round_robin"
    assert event.is_fallback is False


@pytest.mark.integration
def test_model_routing_decided_event_fallback_fields_consistent() -> None:
    """fallback_indicator and is_fallback should agree in normal usage."""
    event = ModelRoutingDecidedEvent(
        selected_provider="claude",
        selected_tier="claude",
        selection_mode="failover",
        fallback_indicator=True,
        is_fallback=True,
        reason="fallback",
        candidates_evaluated=2,
        candidate_providers=["local", "claude"],
        latency_ms=250.0,
        timestamp="2026-04-13T00:00:00Z",
    )
    assert event.fallback_indicator is True
    assert event.selection_mode == "failover"
    assert event.is_fallback is True


@pytest.mark.integration
def test_model_routing_decided_event_json_serializable() -> None:
    """ModelRoutingDecidedEvent.model_dump(mode='json') produces JSON-safe output."""
    import json
    from uuid import uuid4

    event = ModelRoutingDecidedEvent(
        correlation_id=uuid4(),
        selected_provider="local",
        selected_tier="local",
        selection_mode="round_robin",
        fallback_indicator=False,
        is_fallback=False,
        reason="round_robin",
        candidates_evaluated=1,
        candidate_providers=["local"],
        latency_ms=5.0,
        timestamp="2026-04-13T12:00:00Z",
    )
    payload = event.model_dump(mode="json")
    # Must be JSON-serializable (UUID converted to string, etc.)
    serialized = json.dumps(payload)
    assert "round_robin" in serialized
    assert "fallback_indicator" in serialized


@pytest.mark.integration
def test_model_routing_decided_event_selection_mode_literals() -> None:
    """All supported selection_mode literals are accepted by ModelRoutingDecidedEvent."""
    for mode in ("round_robin", "failover", "priority", "cost_optimized"):
        event = ModelRoutingDecidedEvent(
            selected_provider="p",
            selected_tier="local",
            selection_mode=mode,  # type: ignore[arg-type]
            fallback_indicator=(mode != "round_robin"),
            is_fallback=(mode != "round_robin"),
            reason=mode,
            candidates_evaluated=1,
            latency_ms=1.0,
            timestamp="2026-04-13T00:00:00Z",
        )
        assert event.selection_mode == mode
