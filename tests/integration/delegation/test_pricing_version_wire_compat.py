# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: pricing_manifest_version wire-format compatibility (OMN-10949).

Verifies that ``pricing_manifest_version`` survives the cross-model mapping
and JSON wire roundtrip used by the delegation orchestrator and the omnidash
projection consumer:

1. ``ModelTaskDelegatedEvent`` -> Kafka JSON wire -> deserialization preserves
   the new field so omnidash's DelegationProjection can persist it.
2. Backward compatibility: payloads emitted before OMN-10949 lack the field
   and must deserialize with the default of 0, so legacy events do not crash
   the projection consumer.

Unit tests in tests/unit cover individual model defaults and validation.
This module proves the field survives the wire roundtrip that the real
event-bus producer and consumer perform.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
)

pytestmark = [pytest.mark.integration]


def _make_event(*, pricing_manifest_version: int = 0) -> ModelTaskDelegatedEvent:
    return ModelTaskDelegatedEvent(
        timestamp="2026-05-13T10:00:00Z",
        correlation_id=uuid4(),
        task_type="code_generation",
        delegated_to="local-qwen3",
        quality_gate_passed=True,
        pricing_manifest_version=pricing_manifest_version,
    )


class TestPricingVersionWireCompat:
    """Prove pricing_manifest_version survives Kafka wire roundtrip."""

    def test_event_json_wire_roundtrip_preserves_pricing_version(self) -> None:
        """Event -> JSON (Kafka wire) -> deserialization preserves pricing_manifest_version."""
        event = _make_event(pricing_manifest_version=7)
        wire = event.model_dump(mode="json")

        assert wire["pricing_manifest_version"] == 7

        restored = ModelTaskDelegatedEvent.model_validate(wire)
        assert restored.pricing_manifest_version == 7
        assert restored.correlation_id == event.correlation_id

    def test_backward_compat_legacy_payload_without_pricing_version(self) -> None:
        """Legacy payloads emitted before OMN-10949 deserialize with default 0.

        Simulates an omnidash projection consumer receiving an event produced
        before the pricing_manifest_version field was added. The consumer must
        not crash; the default of 0 must apply so the projection row records
        an explicit "unknown manifest" sentinel.
        """
        legacy_payload = {
            "timestamp": "2026-05-09T12:00:00Z",
            "correlation_id": str(uuid4()),
            "task_type": "code_generation",
            "delegated_to": "local-qwen3",
            "quality_gate_passed": True,
        }
        event = ModelTaskDelegatedEvent.model_validate(legacy_payload)
        assert event.pricing_manifest_version == 0

    def test_full_wire_roundtrip_with_nonzero_pricing_version(self) -> None:
        """End-to-end: event with explicit pricing_manifest_version -> wire -> restore."""
        event = _make_event(pricing_manifest_version=42)

        # Producer side: serialize for Kafka publish
        wire = event.model_dump(mode="json")

        # Consumer side: omnidash projection deserialization
        consumed = ModelTaskDelegatedEvent.model_validate(wire)

        assert consumed.pricing_manifest_version == 42
        assert consumed.task_type == "code_generation"
        assert consumed.quality_gate_passed is True
