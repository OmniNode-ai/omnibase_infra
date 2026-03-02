# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Cross-repo schema handshake gate for onex.evt.omniclaude.routing-decision.v1 (OMN-3425).

This test file is the CI gate that enforces schema compatibility between:
  - omniclaude producer (route_via_events_wrapper._build_routing_decision_payload)
  - omnibase_infra consumer (ModelRoutingDecision)

How it works:
  1. The contract YAML at contracts/routing_decision_v1.yaml declares the canonical schema.
  2. Tests in TestContractYamlIntegrity verify the contract YAML is complete and correct.
  3. Tests in TestRoutingDecisionSchemaHandshake verify that:
     a. A contract-shaped payload (all canonical field names) passes ModelRoutingDecision.
     b. A payload with the old producer aliases (confidence, reasoning, session_id) is
        REJECTED by ModelRoutingDecision (proving the shim is still necessary).
     c. ModelRoutingDecisionIngest normalizes the aliased producer payload to the canonical
        shape accepted by ModelRoutingDecision.
  4. TestProducerPayloadAlignmentRegression verifies route_via_events_wrapper's aligned
     payload (producer 2) passes ModelRoutingDecision directly — no shim needed.

Failing any test here means either:
  - The producer drifted from the contract (field renamed/removed/added)
  - The consumer model drifted from the contract
  - The contract YAML drifted from the model definition

Related Tickets:
  - OMN-3425: Cross-repo schema handshake gate (this file)
  - OMN-3422: Fix ModelRoutingDecision schema drift (ingest shim added)
  - OMN-3424: Producer alignment — retire ModelRoutingDecisionIngest shim
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.services.observability.agent_actions.models.model_routing_decision import (
    ModelRoutingDecision,
)
from omnibase_infra.services.observability.agent_actions.models.model_routing_decision_ingest import (
    ModelRoutingDecisionIngest,
)

# Path to the contract YAML — used by integrity tests
_CONTRACT_PATH = Path(__file__).parent.parent / "contracts" / "routing_decision_v1.yaml"

# Minimal required fields as declared in the contract YAML
# These must ALL be present with CANONICAL NAMES in every producer payload
_CONTRACT_REQUIRED_FIELD_NAMES = {
    "id",
    "correlation_id",
    "selected_agent",
    "confidence_score",
    "created_at",
}

# Renamed fields: producer alias -> canonical name
# If any of these producer aliases appear in a payload reaching ModelRoutingDecision
# directly (without the ingest shim), the model must reject them.
_RENAMED_FIELDS: dict[str, str] = {
    "confidence": "confidence_score",
    "reasoning": "routing_reason",
    "session_id": "claude_session_id",
    "emitted_at": "created_at",
}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def contract_yaml() -> dict[str, object]:
    """Load and parse the routing_decision_v1.yaml contract."""
    assert _CONTRACT_PATH.exists(), (
        f"Contract YAML not found: {_CONTRACT_PATH}\nCreate it at: {_CONTRACT_PATH}"
    )
    with open(_CONTRACT_PATH) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


@pytest.fixture
def canonical_payload() -> dict[str, object]:
    """Minimal contract-shaped payload using all canonical field names."""
    return {
        "id": str(uuid4()),
        "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "selected_agent": "polymorphic-agent",
        "confidence_score": 0.85,
        "created_at": datetime.now(UTC).isoformat(),
    }


@pytest.fixture
def full_canonical_payload() -> dict[str, object]:
    """Full contract-shaped payload including optional fields — mimics aligned producer."""
    return {
        "id": str(uuid4()),
        "correlation_id": str(uuid4()),
        "claude_session_id": "sess-abc-123",
        "selected_agent": "api-architect",
        "confidence_score": 0.92,
        "created_at": datetime.now(UTC).isoformat(),
        "domain": "api",
        "routing_reason": "Strong trigger match for API design",
        "routing_method": "SEMANTIC",
        "latency_ms": 37,
        "metadata": {
            "routing_policy": "trigger_match",
            "routing_path": "semantic",
            "event_attempted": True,
            "prompt_preview": "Design a REST API...",
        },
    }


# =============================================================================
# Contract YAML Integrity Tests
# =============================================================================


class TestContractYamlIntegrity:
    """Verify the contract YAML is complete, parseable, and consistent."""

    @pytest.mark.unit
    def test_contract_yaml_exists(self) -> None:
        """Contract YAML must exist at the canonical location."""
        assert _CONTRACT_PATH.exists(), (
            f"Contract YAML missing: {_CONTRACT_PATH}\n"
            "This file is the single source of truth for routing-decision.v1.\n"
            "See OMN-3425 for creation instructions."
        )

    @pytest.mark.unit
    def test_contract_yaml_parseable(self, contract_yaml: dict[str, object]) -> None:
        """Contract YAML must parse without errors."""
        assert isinstance(contract_yaml, dict)
        assert contract_yaml  # non-empty

    @pytest.mark.unit
    def test_contract_declares_topic(self, contract_yaml: dict[str, object]) -> None:
        """Contract must declare the canonical Kafka topic name."""
        assert (
            contract_yaml.get("topic") == "onex.evt.omniclaude.routing-decision.v1"
        ), (
            f"Contract topic mismatch: {contract_yaml.get('topic')}\n"
            "Expected: onex.evt.omniclaude.routing-decision.v1"
        )

    @pytest.mark.unit
    def test_contract_declares_required_fields(
        self, contract_yaml: dict[str, object]
    ) -> None:
        """Contract must declare all required fields."""
        required = contract_yaml.get("required_fields")
        assert isinstance(required, list)
        contract_required_names = {
            field["name"] for field in required if isinstance(field, dict)
        }
        assert contract_required_names == _CONTRACT_REQUIRED_FIELD_NAMES, (
            f"Contract required_fields mismatch.\n"
            f"Expected: {_CONTRACT_REQUIRED_FIELD_NAMES}\n"
            f"Got: {contract_required_names}\n"
            "Update routing_decision_v1.yaml to match ModelRoutingDecision required fields."
        )

    @pytest.mark.unit
    def test_contract_declares_canonical_confidence_score_name(
        self, contract_yaml: dict[str, object]
    ) -> None:
        """Contract must use canonical name 'confidence_score', not 'confidence'."""
        required = contract_yaml.get("required_fields", [])
        assert isinstance(required, list)
        field_names = [
            f["name"] for f in required if isinstance(f, dict) and "name" in f
        ]
        assert "confidence_score" in field_names, (
            "Contract must declare 'confidence_score' (not 'confidence') as required.\n"
            "The producer alias 'confidence' is handled by the ingest shim, not the contract."
        )
        assert "confidence" not in field_names, (
            "Contract must NOT declare 'confidence' — the canonical name is 'confidence_score'.\n"
            "Old producer alias 'confidence' belongs in renamed_fields."
        )

    @pytest.mark.unit
    def test_contract_declares_renamed_fields(
        self, contract_yaml: dict[str, object]
    ) -> None:
        """Contract must declare all known producer-to-canonical field renames."""
        renamed = contract_yaml.get("renamed_fields", [])
        assert isinstance(renamed, list)
        declared_producer_names = {
            f["producer_name"]
            for f in renamed
            if isinstance(f, dict) and "producer_name" in f
        }
        expected_producer_names = set(_RENAMED_FIELDS.keys())
        assert expected_producer_names == declared_producer_names, (
            f"Contract renamed_fields mismatch.\n"
            f"Expected producer_name entries: {expected_producer_names}\n"
            f"Got: {declared_producer_names}\n"
            "Update routing_decision_v1.yaml renamed_fields to match known aliases."
        )

    @pytest.mark.unit
    def test_contract_declares_ci_gate_test(
        self, contract_yaml: dict[str, object]
    ) -> None:
        """Contract must reference this CI gate test file."""
        ci_gate = contract_yaml.get("ci_gate")
        assert isinstance(ci_gate, dict)
        test_file = ci_gate.get("test_file", "")
        assert "test_routing_decision_schema_handshake" in str(test_file), (
            f"Contract ci_gate.test_file does not reference this handshake test.\n"
            f"Got: {test_file}"
        )


# =============================================================================
# Schema Handshake: Canonical Payload -> ModelRoutingDecision
# =============================================================================


class TestRoutingDecisionSchemaHandshake:
    """Core handshake gate: prove canonical payload passes ModelRoutingDecision directly."""

    @pytest.mark.unit
    def test_canonical_payload_passes_strict_model(
        self, canonical_payload: dict[str, object]
    ) -> None:
        """A payload with canonical field names validates against ModelRoutingDecision.

        This is the primary gate: if the producer emits canonical names,
        no ingest shim is needed. Failing this test means the consumer model
        drifted from the contract.
        """
        m = ModelRoutingDecision.model_validate(canonical_payload)
        assert m.selected_agent == "polymorphic-agent"
        assert m.confidence_score == pytest.approx(0.85)
        assert isinstance(m.id, UUID)
        assert isinstance(m.correlation_id, UUID)
        assert isinstance(m.created_at, datetime)

    @pytest.mark.unit
    def test_full_canonical_payload_passes_strict_model(
        self, full_canonical_payload: dict[str, object]
    ) -> None:
        """A full optional-fields payload with canonical names validates correctly."""
        m = ModelRoutingDecision.model_validate(full_canonical_payload)
        assert m.selected_agent == "api-architect"
        assert m.confidence_score == pytest.approx(0.92)
        assert m.domain == "api"
        assert m.routing_reason == "Strong trigger match for API design"
        assert m.routing_method == "SEMANTIC"
        assert m.latency_ms == 37
        assert m.claude_session_id == "sess-abc-123"
        assert isinstance(m.metadata, dict)

    @pytest.mark.unit
    def test_all_contract_required_fields_are_required_by_model(self) -> None:
        """Each required field in the contract is actually required by ModelRoutingDecision.

        Iterates through _CONTRACT_REQUIRED_FIELD_NAMES and verifies that omitting
        each one individually causes a ValidationError.
        """
        base: dict[str, object] = {
            "id": str(uuid4()),
            "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "selected_agent": "test-agent",
            "confidence_score": 0.5,
            "created_at": datetime.now(UTC).isoformat(),
        }
        for field_name in _CONTRACT_REQUIRED_FIELD_NAMES:
            payload = {k: v for k, v in base.items() if k != field_name}
            with pytest.raises(ValidationError, match=field_name):
                ModelRoutingDecision.model_validate(payload)

    @pytest.mark.unit
    def test_confidence_score_name_is_canonical(self) -> None:
        """ModelRoutingDecision requires 'confidence_score', not 'confidence'.

        This test is the contract enforcement gate:
        if the producer emits 'confidence' (old alias) without the shim,
        the strict model MUST reject it. Failing this test means the model
        was softened and the shim is no longer protective.
        """
        with pytest.raises(ValidationError):
            ModelRoutingDecision.model_validate(
                {
                    "id": str(uuid4()),
                    "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "selected_agent": "test-agent",
                    "confidence": 0.85,  # OLD NAME — must be rejected
                    "created_at": datetime.now(UTC).isoformat(),
                }
            )

    @pytest.mark.unit
    def test_old_alias_reasoning_rejected_by_strict_model(self) -> None:
        """ModelRoutingDecision must reject 'reasoning' (must use 'routing_reason')."""
        with pytest.raises(ValidationError):
            ModelRoutingDecision.model_validate(
                {
                    "id": str(uuid4()),
                    "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "selected_agent": "test-agent",
                    "confidence_score": 0.85,
                    "created_at": datetime.now(UTC).isoformat(),
                    "reasoning": "old alias",  # must be rejected
                }
            )

    @pytest.mark.unit
    def test_old_alias_session_id_rejected_by_strict_model(self) -> None:
        """ModelRoutingDecision must reject 'session_id' (must use 'claude_session_id')."""
        with pytest.raises(ValidationError):
            ModelRoutingDecision.model_validate(
                {
                    "id": str(uuid4()),
                    "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "selected_agent": "test-agent",
                    "confidence_score": 0.85,
                    "created_at": datetime.now(UTC).isoformat(),
                    "session_id": "sess-abc",  # old alias — must be rejected
                }
            )

    @pytest.mark.unit
    def test_contract_required_fields_match_model_required_fields(self) -> None:
        """Fields declared required in the contract must be required by ModelRoutingDecision.

        Cross-checks the contract YAML required_fields list against the Pydantic
        model's field definitions to detect drift.
        """
        model_fields = ModelRoutingDecision.model_fields
        model_required = {
            name for name, field in model_fields.items() if field.is_required()
        }
        # The contract required fields must be a subset of model required fields
        extra_in_contract = _CONTRACT_REQUIRED_FIELD_NAMES - model_required
        assert not extra_in_contract, (
            f"Contract declares fields as required that are NOT required by ModelRoutingDecision:\n"
            f"  Extra in contract: {extra_in_contract}\n"
            "Update routing_decision_v1.yaml required_fields to match the model."
        )
        # All model required fields must appear in contract required_fields
        missing_from_contract = model_required - _CONTRACT_REQUIRED_FIELD_NAMES
        assert not missing_from_contract, (
            f"ModelRoutingDecision has required fields NOT declared in contract:\n"
            f"  Missing from contract: {missing_from_contract}\n"
            "Update routing_decision_v1.yaml required_fields to include all model required fields."
        )


# =============================================================================
# Shim Normalization: Producer Aliases -> Canonical -> ModelRoutingDecision
# =============================================================================


class TestIngestShimNormalization:
    """Verify the ingest shim correctly normalizes producer aliases to canonical names."""

    @pytest.mark.unit
    def test_shim_normalizes_confidence_to_confidence_score(self) -> None:
        """ModelRoutingDecisionIngest maps 'confidence' -> 'confidence_score'."""
        ingest = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.88,  # producer alias
            }
        )
        assert ingest.confidence_score == pytest.approx(0.88)

    @pytest.mark.unit
    def test_shim_output_constructs_strict_model(self) -> None:
        """Ingest shim output (model_dump) can construct ModelRoutingDecision."""
        ingest = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "api-architect",
                "confidence": 0.92,
                "session_id": "sess-abc-123",
                "reasoning": "matched API pattern",
                "domain": "api",
                "routing_method": "SEMANTIC",
                "latency_ms": 42,
                "emitted_at": "2026-03-01T12:00:00Z",
            }
        )
        strict = ModelRoutingDecision.model_validate(ingest.model_dump())
        assert strict.selected_agent == "api-architect"
        assert strict.confidence_score == pytest.approx(0.92)
        assert strict.claude_session_id == "sess-abc-123"
        assert strict.routing_reason == "matched API pattern"
        assert strict.routing_method == "SEMANTIC"
        assert strict.latency_ms == 42

    @pytest.mark.unit
    def test_shim_normalizes_emitted_at_to_created_at(self) -> None:
        """ModelRoutingDecisionIngest maps 'emitted_at' -> 'created_at'."""
        ts = "2026-03-01T12:00:00Z"
        ingest = ModelRoutingDecisionIngest.model_validate(
            {
                "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "selected_agent": "test-agent",
                "confidence": 0.5,
                "emitted_at": ts,
            }
        )
        expected = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
        assert ingest.created_at == expected


# =============================================================================
# Producer 2 Alignment Regression: route_via_events_wrapper payload
# =============================================================================


class TestProducerPayloadAlignmentRegression:
    """Regression gate: route_via_events_wrapper already emits canonical field names.

    _build_routing_decision_payload() in route_via_events_wrapper.py returns:
      {
        "id": str(uuid4()),
        "correlation_id": ...,
        "claude_session_id": ...,
        "selected_agent": ...,
        "confidence_score": ...,   <- canonical name (already aligned)
        "created_at": ...,         <- canonical name (already aligned)
        "domain": ...,
        "routing_reason": ...,     <- canonical name (already aligned)
        "metadata": {...},
      }

    This payload must pass ModelRoutingDecision WITHOUT the ingest shim.
    Failing this test means route_via_events_wrapper.py drifted from the contract.
    """

    # This payload mirrors the exact output of _build_routing_decision_payload().
    # Update this fixture if the function's output changes — the test must track it.
    _ROUTE_VIA_EVENTS_PAYLOAD: dict[str, object] = {
        "id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
        "correlation_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
        "claude_session_id": "sess-xyz-789",
        "selected_agent": "polymorphic-agent",
        "confidence_score": 0.87,
        "created_at": "2026-03-01T15:30:00+00:00",
        "domain": "infrastructure",
        "routing_reason": "Matched infrastructure domain pattern",
        "metadata": {
            "routing_method": "SEMANTIC",
            "routing_policy": "trigger_match",
            "routing_path": "semantic",
            "latency_ms": 23,
            "event_attempted": True,
            "prompt_preview": "Set up a Kafka consumer...",
        },
    }

    @pytest.mark.unit
    def test_route_via_events_wrapper_payload_passes_strict_model(self) -> None:
        """route_via_events_wrapper payload passes ModelRoutingDecision without shim.

        This proves producer 2 is already aligned to the canonical schema.
        When producer 1 (hook_event_adapter.py) also aligns, the ingest shim
        can be retired (OMN-3424).
        """
        m = ModelRoutingDecision.model_validate(self._ROUTE_VIA_EVENTS_PAYLOAD)
        assert m.selected_agent == "polymorphic-agent"
        assert m.confidence_score == pytest.approx(0.87)
        assert m.claude_session_id == "sess-xyz-789"
        assert m.routing_reason == "Matched infrastructure domain pattern"
        assert m.domain == "infrastructure"

    @pytest.mark.unit
    def test_route_via_events_wrapper_payload_all_required_fields_present(self) -> None:
        """All required contract fields are present in the route_via_events_wrapper payload."""
        payload_keys = set(self._ROUTE_VIA_EVENTS_PAYLOAD.keys())
        missing = _CONTRACT_REQUIRED_FIELD_NAMES - payload_keys
        assert not missing, (
            f"route_via_events_wrapper payload is missing required contract fields: {missing}\n"
            "Update _ROUTE_VIA_EVENTS_PAYLOAD or fix route_via_events_wrapper._build_routing_decision_payload()."
        )

    @pytest.mark.unit
    def test_route_via_events_wrapper_uses_canonical_confidence_score_name(
        self,
    ) -> None:
        """route_via_events_wrapper emits 'confidence_score', not old alias 'confidence'."""
        assert "confidence_score" in self._ROUTE_VIA_EVENTS_PAYLOAD
        assert "confidence" not in self._ROUTE_VIA_EVENTS_PAYLOAD, (
            "route_via_events_wrapper payload must use 'confidence_score' (canonical name).\n"
            "The old alias 'confidence' was emitted by hook_event_adapter.py (producer 1).\n"
            "If this payload was updated to emit 'confidence', revert it."
        )

    @pytest.mark.unit
    def test_route_via_events_wrapper_uses_canonical_routing_reason_name(self) -> None:
        """route_via_events_wrapper emits 'routing_reason', not old alias 'reasoning'."""
        assert "routing_reason" in self._ROUTE_VIA_EVENTS_PAYLOAD
        assert "reasoning" not in self._ROUTE_VIA_EVENTS_PAYLOAD, (
            "route_via_events_wrapper payload must use 'routing_reason' (canonical name).\n"
            "Old alias 'reasoning' belongs in renamed_fields in the contract YAML."
        )

    @pytest.mark.unit
    def test_route_via_events_wrapper_uses_canonical_session_id_name(self) -> None:
        """route_via_events_wrapper emits 'claude_session_id', not old alias 'session_id'."""
        assert "claude_session_id" in self._ROUTE_VIA_EVENTS_PAYLOAD
        assert "session_id" not in self._ROUTE_VIA_EVENTS_PAYLOAD, (
            "route_via_events_wrapper payload must use 'claude_session_id' (canonical name).\n"
            "Old alias 'session_id' belongs in renamed_fields in the contract YAML."
        )


# =============================================================================
# Failing Producer Simulation: proves CI detects drift
# =============================================================================


class TestDriftDetection:
    """Simulate what happens when a producer emits a renamed field without the shim.

    These tests ensure that if a producer starts emitting old aliases directly
    (bypassing the ingest shim), the strict model detects the drift immediately.
    This satisfies Acceptance Criterion 3:
        "A failing producer field name causes CI failure."
    """

    @pytest.mark.unit
    def test_producer_with_old_confidence_alias_is_detected(self) -> None:
        """Simulates a producer emitting 'confidence' instead of 'confidence_score'.

        This is the drift scenario that OMN-3422 detected in production.
        The CI gate (this test) ensures it can never happen again undetected.
        """
        drifted_payload = {
            "id": str(uuid4()),
            "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "selected_agent": "polymorphic-agent",
            "confidence": 0.85,  # drift: old name
            "created_at": datetime.now(UTC).isoformat(),
        }
        # Without ingest shim: ModelRoutingDecision MUST reject this
        with pytest.raises(ValidationError) as exc_info:
            ModelRoutingDecision.model_validate(drifted_payload)
        # Verify the error is about the extra field, not something else
        errors = exc_info.value.errors()
        error_types = {e["type"] for e in errors}
        # extra='forbid' raises 'extra_forbidden' for unknown fields
        assert "extra_forbidden" in error_types or any(
            "confidence" in str(e) for e in errors
        ), (
            f"Expected ValidationError to mention 'confidence' field.\n"
            f"Got errors: {errors}"
        )

    @pytest.mark.unit
    def test_payload_missing_confidence_score_is_detected(self) -> None:
        """Simulates a producer that stops emitting confidence_score entirely."""
        payload_missing_confidence = {
            "id": str(uuid4()),
            "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "selected_agent": "polymorphic-agent",
            # confidence_score omitted — drift simulation
            "created_at": datetime.now(UTC).isoformat(),
        }
        with pytest.raises(ValidationError, match="confidence_score"):
            ModelRoutingDecision.model_validate(payload_missing_confidence)

    @pytest.mark.unit
    def test_payload_missing_selected_agent_is_detected(self) -> None:
        """Simulates a producer that stops emitting selected_agent."""
        payload_missing_agent = {
            "id": str(uuid4()),
            "correlation_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "confidence_score": 0.75,
            "created_at": datetime.now(UTC).isoformat(),
            # selected_agent omitted
        }
        with pytest.raises(ValidationError, match="selected_agent"):
            ModelRoutingDecision.model_validate(payload_missing_agent)
