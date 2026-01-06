# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Performance regression tests for RegistrationReducer [OMN-1258].

This test suite validates that the RegistrationReducer meets its documented
performance thresholds:

    - reduce() processing: <300ms per event (target)
    - Intent building: <50ms per intent (includes Consul + PostgreSQL)
    - Idempotency check: <1ms

These tests use safety margins (typically 50% of threshold) to account for:
    - CI environment variance
    - GC pauses
    - Resource contention

The tests are marked with @pytest.mark.performance for selective execution.
Run with: pytest -m performance

Related:
    - RegistrationReducer: Implementation under test
    - PR #114: Review that noted missing performance tests
    - OMN-1258: Migration task that includes test additions
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeIntrospectionEvent,
    ModelNodeMetadata,
)
from omnibase_infra.nodes.reducers import RegistrationReducer
from omnibase_infra.nodes.reducers.models import ModelRegistrationState
from omnibase_infra.nodes.reducers.registration_reducer import (
    PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS,
    PERF_THRESHOLD_INTENT_BUILD_MS,
    PERF_THRESHOLD_REDUCE_MS,
)
from tests.helpers import create_introspection_event

# TYPE_CHECKING imports removed - not currently needed
# Can be added back if type-only imports are required

# =============================================================================
# Test Constants
# =============================================================================

# Safety margin for CI variance (50% of threshold)
# This accounts for GC pauses, resource contention, and CI environment variability
SAFETY_MARGIN = 0.5

# Number of iterations for timing stability
TIMING_ITERATIONS = 10

# Fixed test timestamp for deterministic testing
TEST_TIMESTAMP = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def reducer() -> RegistrationReducer:
    """Create a RegistrationReducer instance for testing.

    Returns:
        A new RegistrationReducer instance.
    """
    return RegistrationReducer()


@pytest.fixture
def initial_state() -> ModelRegistrationState:
    """Create an initial idle state for testing.

    Returns:
        A new ModelRegistrationState in idle status.
    """
    return ModelRegistrationState()


@pytest.fixture
def sample_event() -> ModelNodeIntrospectionEvent:
    """Create a sample introspection event for performance testing.

    This event includes all typical fields to represent a realistic
    performance scenario.

    Returns:
        A valid ModelNodeIntrospectionEvent with all fields populated.
    """
    return ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type="effect",
        node_version="1.0.0",
        correlation_id=uuid4(),
        endpoints={"health": "http://localhost:8080/health"},
        capabilities=ModelNodeCapabilities(postgres=True, read=True, write=True),
        metadata=ModelNodeMetadata(environment="performance-test"),
        timestamp=TEST_TIMESTAMP,
    )


# =============================================================================
# Performance Regression Tests
# =============================================================================


@pytest.mark.unit
@pytest.mark.performance
class TestReducerPerformance:
    """Performance regression tests to catch slowdowns in RegistrationReducer.

    These tests verify that critical operations complete within documented
    thresholds with a safety margin to account for CI variance.

    Thresholds (from registration_reducer.py):
        - PERF_THRESHOLD_REDUCE_MS = 300ms (total reduce operation)
        - PERF_THRESHOLD_INTENT_BUILD_MS = 50ms (intent building)
        - PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS = 1ms (idempotency key generation)
    """

    def test_reduce_operation_under_threshold(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
        sample_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Typical reduce() should complete well under 300ms threshold.

        This test measures a single reduce() call and verifies it completes
        within 50% of the threshold (150ms) to provide safety margin for CI.

        The actual operation typically completes in <5ms, so this threshold
        is generous but catches significant regressions.
        """
        start = time.perf_counter()
        output = reducer.reduce(initial_state, sample_event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Verify the operation completed successfully
        assert output.result.status == "pending"
        assert len(output.intents) == 2  # Consul + PostgreSQL

        # Use 50% of threshold as safety margin for CI variance
        max_allowed_ms = PERF_THRESHOLD_REDUCE_MS * SAFETY_MARGIN
        assert elapsed_ms < max_allowed_ms, (
            f"Reduce took {elapsed_ms:.2f}ms, exceeds {SAFETY_MARGIN * 100:.0f}% "
            f"of {PERF_THRESHOLD_REDUCE_MS}ms threshold (max: {max_allowed_ms:.0f}ms)"
        )

    def test_reduce_operation_average_under_threshold(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Average reduce() over multiple iterations should be stable.

        This test runs multiple iterations to measure average performance,
        which provides more stable measurements than a single run.
        """
        elapsed_times: list[float] = []

        for _ in range(TIMING_ITERATIONS):
            # Create fresh event for each iteration
            event = create_introspection_event()
            state = ModelRegistrationState()

            start = time.perf_counter()
            _ = reducer.reduce(state, event)
            elapsed_ms = (time.perf_counter() - start) * 1000
            elapsed_times.append(elapsed_ms)

        avg_elapsed_ms = sum(elapsed_times) / len(elapsed_times)
        max_elapsed_ms = max(elapsed_times)

        # Average should be well under threshold
        max_allowed_ms = PERF_THRESHOLD_REDUCE_MS * SAFETY_MARGIN
        assert avg_elapsed_ms < max_allowed_ms, (
            f"Average reduce took {avg_elapsed_ms:.2f}ms over {TIMING_ITERATIONS} "
            f"iterations, exceeds {SAFETY_MARGIN * 100:.0f}% of "
            f"{PERF_THRESHOLD_REDUCE_MS}ms threshold"
        )

        # Even the slowest iteration should be under threshold
        assert max_elapsed_ms < PERF_THRESHOLD_REDUCE_MS, (
            f"Max reduce took {max_elapsed_ms:.2f}ms, exceeds full threshold of "
            f"{PERF_THRESHOLD_REDUCE_MS}ms"
        )

    def test_idempotency_key_generation_fast(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Idempotency key generation should complete under 1ms threshold.

        Tests the _derive_deterministic_event_id() method which performs
        SHA-256 hashing. This operation should be extremely fast.

        Note: This method is used as a fallback when correlation_id needs
        to be derived from event content. While correlation_id is required
        in the current model, testing this method ensures the fallback
        path remains performant.
        """
        elapsed_times: list[float] = []

        for _ in range(TIMING_ITERATIONS):
            # Create fresh event for each iteration
            event = ModelNodeIntrospectionEvent(
                node_id=uuid4(),
                node_type="effect",
                node_version="1.0.0",
                correlation_id=uuid4(),
                endpoints={},
                capabilities=ModelNodeCapabilities(),
                metadata=ModelNodeMetadata(),
                timestamp=datetime.now(UTC),
            )

            start = time.perf_counter()
            # Access private method for focused testing
            # This method derives a deterministic UUID from event content hash
            event_id = reducer._derive_deterministic_event_id(event)
            elapsed_ms = (time.perf_counter() - start) * 1000
            elapsed_times.append(elapsed_ms)

            # Verify result is valid UUID
            assert event_id is not None

        avg_elapsed_ms = sum(elapsed_times) / len(elapsed_times)
        max_elapsed_ms = max(elapsed_times)

        # Use threshold directly for this fast operation
        # Typical is ~0.01ms, so even 1ms is generous
        assert avg_elapsed_ms < PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS, (
            f"Average idempotency key generation took {avg_elapsed_ms:.4f}ms, "
            f"exceeds {PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS}ms threshold"
        )

        # Max should also be under threshold
        assert max_elapsed_ms < PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS * 2, (
            f"Max idempotency key generation took {max_elapsed_ms:.4f}ms, "
            f"exceeds 2x threshold of {PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS * 2}ms"
        )

    def test_consul_intent_building_fast(
        self,
        reducer: RegistrationReducer,
        sample_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Consul intent building should complete under 50ms threshold.

        Tests the _build_consul_intent() method which constructs the
        Consul registration intent with service metadata.
        """
        correlation_id = uuid4()
        elapsed_times: list[float] = []

        for _ in range(TIMING_ITERATIONS):
            start = time.perf_counter()
            intent = reducer._build_consul_intent(sample_event, correlation_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            elapsed_times.append(elapsed_ms)

            # Verify intent is valid
            assert intent is not None
            assert intent.intent_type == "extension"

        avg_elapsed_ms = sum(elapsed_times) / len(elapsed_times)

        # Use 50% safety margin
        max_allowed_ms = PERF_THRESHOLD_INTENT_BUILD_MS * SAFETY_MARGIN
        assert avg_elapsed_ms < max_allowed_ms, (
            f"Average Consul intent building took {avg_elapsed_ms:.2f}ms, "
            f"exceeds {SAFETY_MARGIN * 100:.0f}% of {PERF_THRESHOLD_INTENT_BUILD_MS}ms threshold"
        )

    def test_postgres_intent_building_fast(
        self,
        reducer: RegistrationReducer,
        sample_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """PostgreSQL intent building should complete under 50ms threshold.

        Tests the _build_postgres_intent() method which constructs the
        PostgreSQL upsert intent with the full registration record.
        """
        correlation_id = uuid4()
        elapsed_times: list[float] = []

        for _ in range(TIMING_ITERATIONS):
            start = time.perf_counter()
            intent = reducer._build_postgres_intent(sample_event, correlation_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            elapsed_times.append(elapsed_ms)

            # Verify intent is valid
            assert intent is not None
            assert intent.intent_type == "extension"

        avg_elapsed_ms = sum(elapsed_times) / len(elapsed_times)

        # Use 50% safety margin
        max_allowed_ms = PERF_THRESHOLD_INTENT_BUILD_MS * SAFETY_MARGIN
        assert avg_elapsed_ms < max_allowed_ms, (
            f"Average PostgreSQL intent building took {avg_elapsed_ms:.2f}ms, "
            f"exceeds {SAFETY_MARGIN * 100:.0f}% of {PERF_THRESHOLD_INTENT_BUILD_MS}ms threshold"
        )

    def test_validation_is_fast(
        self,
        reducer: RegistrationReducer,
        sample_event: ModelNodeIntrospectionEvent,
    ) -> None:
        """Event validation should be negligible in overall processing time.

        Tests the _validate_event() method which performs field validation.
        This should be extremely fast as it's just attribute checks.
        """
        elapsed_times: list[float] = []

        for _ in range(TIMING_ITERATIONS):
            start = time.perf_counter()
            result = reducer._validate_event(sample_event)
            elapsed_ms = (time.perf_counter() - start) * 1000
            elapsed_times.append(elapsed_ms)

            # Verify validation passes
            assert result.is_valid

        avg_elapsed_ms = sum(elapsed_times) / len(elapsed_times)

        # Validation should be sub-millisecond
        assert avg_elapsed_ms < 1.0, (
            f"Average validation took {avg_elapsed_ms:.4f}ms, "
            f"expected <1ms for simple attribute checks"
        )

    def test_idempotency_check_is_fast(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """Duplicate event check should be under 1ms threshold.

        Tests ModelRegistrationState.is_duplicate_event() which is
        a simple UUID comparison.
        """
        event_id = uuid4()
        state = ModelRegistrationState(last_processed_event_id=event_id)
        different_id = uuid4()

        elapsed_times: list[float] = []

        for _ in range(TIMING_ITERATIONS):
            start = time.perf_counter()
            # Test both match and non-match paths
            _ = state.is_duplicate_event(event_id)  # True path
            _ = state.is_duplicate_event(different_id)  # False path
            elapsed_ms = (time.perf_counter() - start) * 1000
            elapsed_times.append(elapsed_ms)

        avg_elapsed_ms = sum(elapsed_times) / len(elapsed_times)

        # Two UUID comparisons should be well under threshold
        assert avg_elapsed_ms < PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS, (
            f"Average idempotency check took {avg_elapsed_ms:.4f}ms for 2 checks, "
            f"exceeds {PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS}ms threshold"
        )


@pytest.mark.unit
@pytest.mark.performance
class TestReducerPerformanceEdgeCases:
    """Performance tests for edge cases and complex scenarios."""

    def test_reduce_with_empty_endpoints_still_fast(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Reduce with empty endpoints should not have performance penalty."""
        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            correlation_id=uuid4(),
            endpoints={},  # Empty endpoints
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
            timestamp=datetime.now(UTC),
        )

        start = time.perf_counter()
        output = reducer.reduce(initial_state, event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert output.result.status == "pending"

        max_allowed_ms = PERF_THRESHOLD_REDUCE_MS * SAFETY_MARGIN
        assert elapsed_ms < max_allowed_ms, (
            f"Reduce with empty endpoints took {elapsed_ms:.2f}ms, exceeds threshold"
        )

    def test_reduce_with_many_endpoints_still_fast(
        self,
        reducer: RegistrationReducer,
        initial_state: ModelRegistrationState,
    ) -> None:
        """Reduce with many endpoints should still meet threshold."""
        # Create event with many endpoints
        endpoints = {
            f"endpoint_{i}": f"http://localhost:808{i}/path" for i in range(50)
        }
        endpoints["health"] = "http://localhost:8080/health"

        event = ModelNodeIntrospectionEvent(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            correlation_id=uuid4(),
            endpoints=endpoints,
            capabilities=ModelNodeCapabilities(),
            metadata=ModelNodeMetadata(),
            timestamp=datetime.now(UTC),
        )

        start = time.perf_counter()
        output = reducer.reduce(initial_state, event)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert output.result.status == "pending"

        max_allowed_ms = PERF_THRESHOLD_REDUCE_MS * SAFETY_MARGIN
        assert elapsed_ms < max_allowed_ms, (
            f"Reduce with many endpoints took {elapsed_ms:.2f}ms, exceeds threshold"
        )

    def test_reduce_reset_is_fast(
        self,
        reducer: RegistrationReducer,
    ) -> None:
        """reduce_reset() should be fast as it's a simple state transition."""
        failed_state = ModelRegistrationState(
            status="failed",
            failure_reason="consul_failed",
            node_id=uuid4(),
        )

        elapsed_times: list[float] = []

        for _ in range(TIMING_ITERATIONS):
            # Generate fresh reset_event_id per iteration to avoid idempotency
            # early-exit bias (duplicate detection would skip full reset)
            reset_event_id = uuid4()

            start = time.perf_counter()
            output = reducer.reduce_reset(failed_state, reset_event_id)
            elapsed_ms = (time.perf_counter() - start) * 1000
            elapsed_times.append(elapsed_ms)

            assert output.result.status == "idle"

        avg_elapsed_ms = sum(elapsed_times) / len(elapsed_times)

        # Reset should be very fast (no intents, simple state copy)
        # Use a fraction of the reduce threshold
        max_allowed_ms = PERF_THRESHOLD_REDUCE_MS * 0.1  # 10% of reduce threshold
        assert avg_elapsed_ms < max_allowed_ms, (
            f"Average reduce_reset took {avg_elapsed_ms:.2f}ms, "
            f"expected <{max_allowed_ms:.0f}ms"
        )


@pytest.mark.unit
@pytest.mark.performance
class TestThresholdConstants:
    """Tests to verify threshold constants are properly defined and accessible."""

    def test_threshold_constants_are_positive(self) -> None:
        """Verify all threshold constants are positive numbers."""
        assert PERF_THRESHOLD_REDUCE_MS > 0, "Reduce threshold must be positive"
        assert PERF_THRESHOLD_INTENT_BUILD_MS > 0, (
            "Intent build threshold must be positive"
        )
        assert PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS > 0, (
            "Idempotency threshold must be positive"
        )

    def test_threshold_hierarchy_is_logical(self) -> None:
        """Verify threshold hierarchy makes sense.

        The reduce threshold should be larger than the sum of component thresholds
        since reduce includes validation, idempotency check, and intent building.
        """
        # Reduce threshold should accommodate at least:
        # - 2 intent builds (Consul + PostgreSQL)
        # - 1 idempotency check
        # - validation overhead
        component_sum = (
            2 * PERF_THRESHOLD_INTENT_BUILD_MS + PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS
        )

        assert component_sum < PERF_THRESHOLD_REDUCE_MS, (
            f"Sum of component thresholds ({component_sum}ms) should be less than "
            f"reduce threshold ({PERF_THRESHOLD_REDUCE_MS}ms)"
        )

    def test_thresholds_are_documented_values(self) -> None:
        """Verify thresholds match documented default values.

        These are the documented defaults. If these fail, either the
        documentation or implementation needs updating.
        """
        # These match the documented values in the module docstring
        # and the environment variable defaults
        assert PERF_THRESHOLD_REDUCE_MS == 300.0, (
            f"Reduce threshold changed from documented 300ms to {PERF_THRESHOLD_REDUCE_MS}ms"
        )
        assert PERF_THRESHOLD_INTENT_BUILD_MS == 50.0, (
            f"Intent build threshold changed from documented 50ms to {PERF_THRESHOLD_INTENT_BUILD_MS}ms"
        )
        assert PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS == 1.0, (
            f"Idempotency threshold changed from documented 1ms to {PERF_THRESHOLD_IDEMPOTENCY_CHECK_MS}ms"
        )


__all__ = [
    "TestReducerPerformance",
    "TestReducerPerformanceEdgeCases",
    "TestThresholdConstants",
]
