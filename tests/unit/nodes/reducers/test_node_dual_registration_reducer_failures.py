# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for NodeDualRegistrationReducer validation and edge cases.

Tests pure reducer behavior for intent emission, verifying validation logic,
error handling, and metrics tracking per OMN-889 requirements.

Pure Reducer Architecture:
    The reducer is now a PURE component that emits typed intents instead of
    performing I/O. Since there's no I/O, there are no "failures" in the
    traditional sense - the reducer either:
    - Emits intents successfully (status="success")
    - Fails validation (status="failed", no intents)

    Handler/adapter failure scenarios are no longer applicable since the
    reducer doesn't interact with handlers. Instead, these tests focus on:
    - Validation failure scenarios
    - Edge cases in intent building
    - Metrics tracking
    - FSM state behavior
    - Correlation ID propagation

Test Categories:
    1. Validation Failures - Invalid input handling
    2. Validation Success - All node types work
    3. Metrics Tracking - Counter increments
    4. Intent Emission - Correct intent structure
    5. FSM State Behavior - State transitions
    6. Correlation ID Propagation - Tracing support
    7. Node ID Preservation - Identity tracking
    8. Edge Cases - Boundary conditions
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from omnibase_core.models.intents import (
    ModelConsulRegisterIntent,
    ModelPostgresUpsertRegistrationIntent,
)

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.registration import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.nodes.reducers.node_dual_registration_reducer import (
    EnumFSMState,
    NodeDualRegistrationReducer,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_introspection_event() -> ModelNodeIntrospectionEvent:
    """Create a sample introspection event for testing."""
    return ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        capabilities=ModelNodeCapabilities(),
        endpoints={"health": "http://localhost:8080/health"},
        correlation_id=uuid4(),
    )


@pytest.fixture
def fsm_contract_path() -> Path:
    """Path to FSM contract for testing."""
    return (
        Path(__file__).parent.parent.parent.parent.parent
        / "contracts"
        / "fsm"
        / "dual_registration_reducer_fsm.yaml"
    )


@pytest.fixture
def dual_registration_reducer(fsm_contract_path: Path) -> NodeDualRegistrationReducer:
    """Create a pure dual registration reducer for testing.

    Pure Reducer Design:
        The reducer no longer accepts ConsulHandler or DbAdapter because
        it emits typed intents instead of performing I/O operations.
    """
    return NodeDualRegistrationReducer(fsm_contract_path=fsm_contract_path)


# =============================================================================
# TEST CLASS: VALIDATION FAILURES
# =============================================================================


class TestValidationFailures:
    """Test validation failure scenarios."""

    @pytest.mark.asyncio
    async def test_validation_failure_returns_failed_status(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that validation failure returns failed status with no intents."""
        await dual_registration_reducer.initialize()

        # Mock validation to fail
        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Custom validation error"),
        ):
            result = await dual_registration_reducer.execute(sample_introspection_event)

        assert result.status == "failed"
        assert result.consul_intent_emitted is False
        assert result.postgres_intent_emitted is False
        assert len(result.intents) == 0
        assert result.validation_error == "Custom validation error"

    @pytest.mark.asyncio
    async def test_validation_failure_with_null_node_id(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test validation fails when node_id is None."""
        await dual_registration_reducer.initialize()

        # Create mock event with None node_id
        mock_event = MagicMock(spec=ModelNodeIntrospectionEvent)
        mock_event.node_id = None
        mock_event.node_type = "effect"
        mock_event.correlation_id = uuid4()

        validation_passed, error_message = dual_registration_reducer._validate_payload(
            mock_event, uuid4()
        )

        assert validation_passed is False
        assert "node_id" in error_message

    def test_pydantic_rejects_invalid_node_type(self) -> None:
        """Verify Pydantic Literal type rejects invalid node_type values.

        node_type validation is enforced at model construction via the
        Literal["effect", "compute", "reducer", "orchestrator"] type annotation.
        The reducer's _validate_payload method no longer performs this validation
        since Pydantic handles it automatically.

        This test documents that invalid node_type values are rejected before
        they can reach the reducer.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="invalid_type",  # Not in Literal
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={"health": "http://localhost:8080/health"},
            )

        # Verify the error mentions node_type
        assert "node_type" in str(exc_info.value)


# =============================================================================
# TEST CLASS: VALIDATION SUCCESS
# =============================================================================


class TestValidationSuccess:
    """Test that validation passes for valid inputs."""

    @pytest.mark.asyncio
    async def test_validation_passes_for_all_node_types(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test validation passes for all valid node types."""
        await dual_registration_reducer.initialize()

        for node_type in ["effect", "compute", "reducer", "orchestrator"]:
            event = ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type=node_type,  # type: ignore[arg-type]
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={"health": "http://localhost:8080/health"},
            )

            result = await dual_registration_reducer.execute(event)

            assert result.status == "success", f"Failed for node_type: {node_type}"
            assert len(result.intents) == 2

    @pytest.mark.asyncio
    async def test_success_emits_both_intents(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test successful execution emits both Consul and PostgreSQL intents."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        assert result.status == "success"
        assert result.consul_intent_emitted is True
        assert result.postgres_intent_emitted is True
        assert len(result.intents) == 2

        # Verify intent types
        intent_types = {type(intent) for intent in result.intents}
        assert ModelConsulRegisterIntent in intent_types
        assert ModelPostgresUpsertRegistrationIntent in intent_types


# =============================================================================
# TEST CLASS: METRICS TRACKING
# =============================================================================


class TestMetricsTracking:
    """Test that metrics counters are correctly incremented."""

    @pytest.mark.asyncio
    async def test_metrics_increment_success_count(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify success_count is incremented on successful intent emission."""
        await dual_registration_reducer.initialize()

        initial_success = dual_registration_reducer.metrics.success_count
        await dual_registration_reducer.execute(sample_introspection_event)

        assert dual_registration_reducer.metrics.success_count == initial_success + 1

    @pytest.mark.asyncio
    async def test_metrics_increment_failure_count_on_validation_failure(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify failure_count is incremented on validation failure."""
        await dual_registration_reducer.initialize()

        initial_failure = dual_registration_reducer.metrics.failure_count

        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation error"),
        ):
            await dual_registration_reducer.execute(sample_introspection_event)

        assert dual_registration_reducer.metrics.failure_count == initial_failure + 1

    @pytest.mark.asyncio
    async def test_metrics_total_registrations_always_incremented(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify total_registrations is always incremented regardless of outcome."""
        await dual_registration_reducer.initialize()

        initial_total = dual_registration_reducer.metrics.total_registrations

        # First execution - success
        await dual_registration_reducer.execute(sample_introspection_event)
        assert (
            dual_registration_reducer.metrics.total_registrations == initial_total + 1
        )

        # Second execution - validation failure
        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation error"),
        ):
            await dual_registration_reducer.execute(sample_introspection_event)

        assert (
            dual_registration_reducer.metrics.total_registrations == initial_total + 2
        )

    @pytest.mark.asyncio
    async def test_metrics_accumulate_across_multiple_executions(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify metrics accumulate correctly across multiple executions."""
        await dual_registration_reducer.initialize()

        # Execute multiple times
        for i in range(3):
            event = ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="effect",
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={"health": "http://localhost:8080/health"},
            )
            await dual_registration_reducer.execute(event)

        assert dual_registration_reducer.metrics.total_registrations == 3
        assert dual_registration_reducer.metrics.success_count == 3
        assert dual_registration_reducer.metrics.partial_count == 0
        assert dual_registration_reducer.metrics.failure_count == 0


# =============================================================================
# TEST CLASS: INTENT EMISSION
# =============================================================================


class TestIntentEmission:
    """Test correct intent structure and values."""

    @pytest.mark.asyncio
    async def test_consul_intent_service_id_format(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test Consul intent has correct service ID format."""
        await dual_registration_reducer.initialize()

        node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="compute",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        result = await dual_registration_reducer.execute(event)

        consul_intent = next(
            (i for i in result.intents if isinstance(i, ModelConsulRegisterIntent)),
            None,
        )
        assert consul_intent is not None
        assert consul_intent.service_id == f"node-compute-{node_id}"
        assert consul_intent.service_name == "onex-compute"

    @pytest.mark.asyncio
    async def test_consul_intent_tags(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test Consul intent has correct tags."""
        await dual_registration_reducer.initialize()

        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="reducer",
            node_version="2.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        result = await dual_registration_reducer.execute(event)

        consul_intent = next(
            (i for i in result.intents if isinstance(i, ModelConsulRegisterIntent)),
            None,
        )
        assert consul_intent is not None
        assert "node_type:reducer" in consul_intent.tags
        assert "node_version:2.0.0" in consul_intent.tags

    @pytest.mark.asyncio
    async def test_postgres_intent_record_data(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test PostgreSQL intent has correct record data."""
        await dual_registration_reducer.initialize()

        node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type="orchestrator",
            node_version="3.0.0",
            capabilities=ModelNodeCapabilities(postgres=True, kafka=True),
            endpoints={
                "health": "http://localhost:8080/health",
                "api": "http://localhost:8080/api/v1",
            },
        )

        result = await dual_registration_reducer.execute(event)

        postgres_intent = next(
            (
                i
                for i in result.intents
                if isinstance(i, ModelPostgresUpsertRegistrationIntent)
            ),
            None,
        )
        assert postgres_intent is not None
        assert postgres_intent.record.node_id == node_id
        assert postgres_intent.record.node_type == "orchestrator"
        assert postgres_intent.record.node_version == "3.0.0"

    @pytest.mark.asyncio
    async def test_consul_intent_health_check_config(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test Consul intent has correct health check configuration."""
        await dual_registration_reducer.initialize()

        health_endpoint = "http://localhost:9090/healthz"
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": health_endpoint},
        )

        result = await dual_registration_reducer.execute(event)

        consul_intent = next(
            (i for i in result.intents if isinstance(i, ModelConsulRegisterIntent)),
            None,
        )
        assert consul_intent is not None
        assert consul_intent.health_check is not None
        assert consul_intent.health_check["HTTP"] == health_endpoint
        assert consul_intent.health_check["Interval"] == "10s"
        assert consul_intent.health_check["Timeout"] == "5s"


# =============================================================================
# TEST CLASS: PERFORMANCE
# =============================================================================


class TestPerformance:
    """Test performance characteristics of intent building."""

    @pytest.mark.asyncio
    async def test_intent_building_is_fast(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify intent building completes quickly (no I/O)."""
        await dual_registration_reducer.initialize()

        start = time.perf_counter()
        result = await dual_registration_reducer.execute(sample_introspection_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Intent building should be very fast (sub-100ms)
        # since there's no I/O
        assert elapsed_ms < 100, f"Intent building took too long: {elapsed_ms}ms"
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_processing_time_ms_is_recorded(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify processing_time_ms field is populated."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        assert result.processing_time_ms >= 0.0
        # Should be reasonable (under 100ms for pure computation)
        assert result.processing_time_ms < 100.0


# =============================================================================
# TEST CLASS: FSM STATE BEHAVIOR
# =============================================================================


class TestFSMStateBehavior:
    """Test FSM state transitions during various scenarios."""

    @pytest.mark.asyncio
    async def test_fsm_returns_to_idle_after_success(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify FSM returns to idle state after successful execution."""
        await dual_registration_reducer.initialize()

        await dual_registration_reducer.execute(sample_introspection_event)

        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

    @pytest.mark.asyncio
    async def test_fsm_returns_to_idle_after_validation_failure(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify FSM returns to idle state after validation failure."""
        await dual_registration_reducer.initialize()

        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation error"),
        ):
            await dual_registration_reducer.execute(sample_introspection_event)

        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

    @pytest.mark.asyncio
    async def test_fsm_can_process_multiple_events(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify FSM can process multiple events in sequence."""
        await dual_registration_reducer.initialize()

        for i in range(5):
            event = ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="effect",
                node_version="1.0.0",
                capabilities=ModelNodeCapabilities(),
                endpoints={"health": "http://localhost:8080/health"},
            )
            result = await dual_registration_reducer.execute(event)
            assert result.status == "success"
            assert dual_registration_reducer.current_state == EnumFSMState.IDLE

        assert dual_registration_reducer.metrics.total_registrations == 5


# =============================================================================
# TEST CLASS: CORRELATION ID PROPAGATION
# =============================================================================


class TestCorrelationIdPropagation:
    """Test correlation ID is correctly propagated."""

    @pytest.mark.asyncio
    async def test_correlation_id_in_result(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify correlation_id is preserved in result."""
        await dual_registration_reducer.initialize()

        test_correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=test_correlation_id,
        )

        result = await dual_registration_reducer.execute(event)

        assert result.correlation_id == test_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_in_intents(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify correlation_id is propagated to all emitted intents."""
        await dual_registration_reducer.initialize()

        test_correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=test_correlation_id,
        )

        result = await dual_registration_reducer.execute(event, test_correlation_id)

        for intent in result.intents:
            assert intent.correlation_id == test_correlation_id

    @pytest.mark.asyncio
    async def test_correlation_id_in_validation_failure_result(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify correlation_id is preserved in validation failure result."""
        await dual_registration_reducer.initialize()

        test_correlation_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
            correlation_id=test_correlation_id,
        )

        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation error"),
        ):
            result = await dual_registration_reducer.execute(event, test_correlation_id)

        assert result.correlation_id == test_correlation_id


# =============================================================================
# TEST CLASS: NODE ID PRESERVATION
# =============================================================================


class TestNodeIdPreservation:
    """Test node_id is correctly preserved in results."""

    @pytest.mark.asyncio
    async def test_node_id_preserved_in_success_result(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify node_id is preserved in success result."""
        await dual_registration_reducer.initialize()

        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        result = await dual_registration_reducer.execute(event)

        assert result.node_id == test_node_id

    @pytest.mark.asyncio
    async def test_node_id_preserved_in_failure_result(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify node_id is preserved in validation failure result."""
        await dual_registration_reducer.initialize()

        test_node_id = uuid4()
        event = ModelNodeIntrospectionEvent(
            node_id=test_node_id,
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )

        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation error"),
        ):
            result = await dual_registration_reducer.execute(event)

        assert result.node_id == test_node_id


# =============================================================================
# TEST CLASS: EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_execute_without_initialization_raises_error(
        self,
        fsm_contract_path: Path,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Verify execute raises error if reducer not initialized."""
        reducer = NodeDualRegistrationReducer(fsm_contract_path=fsm_contract_path)

        with pytest.raises(RuntimeHostError) as exc_info:
            await reducer.execute(sample_introspection_event)

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_reducer_can_process_after_validation_failure(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Verify reducer can process new events after experiencing validation failure."""
        await dual_registration_reducer.initialize()

        # First event - validation failure
        event1 = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )
        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation error"),
        ):
            result1 = await dual_registration_reducer.execute(event1)
        assert result1.status == "failed"

        # Second event - should succeed (validation not mocked)
        event2 = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="compute",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )
        result2 = await dual_registration_reducer.execute(event2)

        assert result2.status == "success"
        assert len(result2.intents) == 2
        assert dual_registration_reducer.metrics.total_registrations == 2

    @pytest.mark.asyncio
    async def test_empty_endpoints_handled(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that empty endpoints dict is handled correctly."""
        await dual_registration_reducer.initialize()

        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={},  # Empty endpoints
        )

        result = await dual_registration_reducer.execute(event)

        # Should still succeed - health check will be None
        assert result.status == "success"
        assert len(result.intents) == 2

        consul_intent = next(
            (i for i in result.intents if isinstance(i, ModelConsulRegisterIntent)),
            None,
        )
        assert consul_intent is not None
        assert consul_intent.health_check is None

    @pytest.mark.asyncio
    async def test_empty_capabilities_handled(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that empty capabilities is handled correctly."""
        await dual_registration_reducer.initialize()

        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),  # Default empty capabilities
            endpoints={"health": "http://localhost:8080/health"},
        )

        result = await dual_registration_reducer.execute(event)

        assert result.status == "success"
        assert len(result.intents) == 2

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
    ) -> None:
        """Test that shutdown properly resets reducer state."""
        await dual_registration_reducer.initialize()

        # Execute once
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            capabilities=ModelNodeCapabilities(),
            endpoints={"health": "http://localhost:8080/health"},
        )
        await dual_registration_reducer.execute(event)

        # Shutdown
        await dual_registration_reducer.shutdown()

        # Verify state reset
        assert dual_registration_reducer.current_state == EnumFSMState.IDLE

        # Execute again should fail (not initialized)
        with pytest.raises(RuntimeHostError) as exc_info:
            await dual_registration_reducer.execute(event)
        assert "not initialized" in str(exc_info.value)


# =============================================================================
# TEST CLASS: OUTPUT MODEL VALIDATION
# =============================================================================


class TestOutputModelValidation:
    """Test output model structure and validation."""

    @pytest.mark.asyncio
    async def test_output_model_is_frozen(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that output model is frozen (immutable)."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        # Try to modify should raise error
        with pytest.raises(Exception):  # ValidationError or AttributeError
            result.status = "failed"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_output_intents_is_tuple(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that output intents is a tuple (immutable)."""
        await dual_registration_reducer.initialize()

        result = await dual_registration_reducer.execute(sample_introspection_event)

        assert isinstance(result.intents, tuple)

    @pytest.mark.asyncio
    async def test_failed_output_has_empty_intents(
        self,
        dual_registration_reducer: NodeDualRegistrationReducer,
        sample_introspection_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Test that failed output has empty intents tuple."""
        await dual_registration_reducer.initialize()

        with patch.object(
            dual_registration_reducer,
            "_validate_payload",
            return_value=(False, "Validation error"),
        ):
            result = await dual_registration_reducer.execute(sample_introspection_event)

        assert result.status == "failed"
        assert result.intents == ()
        assert len(result.intents) == 0


__all__ = [
    "TestValidationFailures",
    "TestValidationSuccess",
    "TestMetricsTracking",
    "TestIntentEmission",
    "TestPerformance",
    "TestFSMStateBehavior",
    "TestCorrelationIdPropagation",
    "TestNodeIdPreservation",
    "TestEdgeCases",
    "TestOutputModelValidation",
]
