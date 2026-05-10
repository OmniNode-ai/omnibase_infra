# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: compliance fields wire-format compatibility (OMN-10790).

Verifies that ``tokens_to_compliance`` and ``compliance_attempts`` survive the
full delegation pipeline serialization path:

1. ``ModelDelegationResult`` -> JSON -> ``ModelTaskDelegatedEvent`` field mapping
2. ``ModelTaskDelegatedEvent`` -> Kafka JSON wire format -> deserialization
3. Backward compatibility: payloads without the new fields deserialize with
   correct defaults (existing Kafka consumers must not break).

Unit tests cover individual model defaults and validation. This module proves
the fields survive the cross-model mapping and JSON wire roundtrip that the
real delegation orchestrator and omnidash projection consumer perform.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_result import (
    ModelDelegationResult,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_task_delegated_event import (
    ModelTaskDelegatedEvent,
)

pytestmark = [pytest.mark.integration]


def _make_delegation_result(
    *,
    tokens_to_compliance: int = 0,
    compliance_attempts: int = 1,
) -> ModelDelegationResult:
    """Build a realistic ModelDelegationResult with compliance fields."""
    return ModelDelegationResult(
        correlation_id=uuid4(),
        task_type="code_generation",
        model_used="cyankiwi/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit",
        endpoint_url="http://192.168.86.201:8000",  # onex-allow-internal-ip
        content="def hello():\n    return 'world'",
        quality_passed=True,
        quality_score=0.92,
        latency_ms=1500,
        prompt_tokens=800,
        completion_tokens=200,
        total_tokens=1000,
        fallback_to_claude=False,
        tokens_to_compliance=tokens_to_compliance,
        compliance_attempts=compliance_attempts,
    )


def _result_to_event(result: ModelDelegationResult) -> ModelTaskDelegatedEvent:
    """Map a ModelDelegationResult to a ModelTaskDelegatedEvent.

    Mirrors the field mapping performed by the real delegation orchestrator
    handler (HandlerDelegationWorkflow._build_backward_compat_event).
    """
    return ModelTaskDelegatedEvent(
        timestamp=datetime.now(tz=UTC).isoformat(),
        correlation_id=result.correlation_id,
        task_type=result.task_type,
        delegated_to=result.endpoint_url,
        model_name=result.model_used,
        quality_gate_passed=result.quality_passed,
        delegation_latency_ms=result.latency_ms,
        tokens_to_compliance=result.tokens_to_compliance,
        compliance_attempts=result.compliance_attempts,
    )


class TestComplianceFieldsWireCompat:
    """Prove compliance fields survive cross-model mapping and wire roundtrip."""

    def test_result_to_event_preserves_compliance_fields(self) -> None:
        """Compliance fields map correctly from result to backward-compat event."""
        result = _make_delegation_result(
            tokens_to_compliance=4500,
            compliance_attempts=3,
        )
        event = _result_to_event(result)
        assert event.tokens_to_compliance == 4500
        assert event.compliance_attempts == 3

    def test_event_json_wire_roundtrip(self) -> None:
        """Event -> JSON (Kafka wire) -> deserialization preserves compliance fields."""
        result = _make_delegation_result(
            tokens_to_compliance=7200,
            compliance_attempts=4,
        )
        event = _result_to_event(result)

        # Serialize to JSON (Kafka wire format)
        wire_json = event.model_dump(mode="json")

        # Deserialize as if a consumer received this from Kafka
        restored = ModelTaskDelegatedEvent.model_validate(wire_json)

        assert restored.tokens_to_compliance == 7200
        assert restored.compliance_attempts == 4
        assert restored.correlation_id == event.correlation_id

    def test_backward_compat_event_without_compliance_fields(self) -> None:
        """Existing Kafka payloads without compliance fields deserialize with defaults.

        This simulates an omnidash consumer receiving a payload produced before
        OMN-10790 was deployed -- the old payload lacks the new fields. The
        consumer must not crash; defaults (0 tokens, 1 attempt) must apply.
        """
        legacy_payload = {
            "timestamp": "2026-05-09T12:00:00Z",
            "correlation_id": str(uuid4()),
            "task_type": "code_generation",
            "delegated_to": "local-qwen3",
            "quality_gate_passed": True,
        }
        # No tokens_to_compliance or compliance_attempts in payload
        event = ModelTaskDelegatedEvent.model_validate(legacy_payload)
        assert event.tokens_to_compliance == 0
        assert event.compliance_attempts == 1

    def test_backward_compat_result_without_compliance_fields(self) -> None:
        """Existing result payloads without compliance fields deserialize with defaults."""
        legacy_payload = {
            "correlation_id": str(uuid4()),
            "task_type": "test",
            "model_used": "test-model",
            "endpoint_url": "http://192.168.86.201:8000",  # onex-allow-internal-ip
            "content": "test content",
            "quality_passed": True,
            "quality_score": 0.9,
            "latency_ms": 500,
            "fallback_to_claude": False,
        }
        result = ModelDelegationResult.model_validate(legacy_payload)
        assert result.tokens_to_compliance == 0
        assert result.compliance_attempts == 1

    def test_full_pipeline_roundtrip_with_multi_attempt_compliance(self) -> None:
        """End-to-end: result with multi-attempt compliance -> event -> wire -> restore."""
        result = _make_delegation_result(
            tokens_to_compliance=12000,
            compliance_attempts=5,
        )

        # Step 1: Result -> Event (orchestrator mapping)
        event = _result_to_event(result)

        # Step 2: Event -> JSON wire (Kafka producer)
        wire = event.model_dump(mode="json")

        # Step 3: JSON wire -> Event (Kafka consumer / omnidash projection)
        consumed = ModelTaskDelegatedEvent.model_validate(wire)

        # Verify end-to-end preservation
        assert consumed.tokens_to_compliance == 12000
        assert consumed.compliance_attempts == 5
        assert consumed.task_type == "code_generation"
        assert consumed.quality_gate_passed is True
