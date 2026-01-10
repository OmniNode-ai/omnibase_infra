# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ServiceNodeSelector.

This test suite validates the node selection service that implements various
strategies for selecting nodes from a candidate list (FIRST, RANDOM, ROUND_ROBIN).
Note: LEAST_LOADED is not yet implemented and raises RuntimeHostError.

Test Organization:
    - TestEnumSelectionStrategy: Enum definition tests
    - TestSelectionStrategyFirst: FIRST strategy tests
    - TestSelectionStrategyRandom: RANDOM strategy tests
    - TestSelectionStrategyRoundRobin: ROUND_ROBIN strategy tests
    - TestSelectionStrategyLeastLoaded: LEAST_LOADED strategy tests (RuntimeHostError)
    - TestServiceNodeSelectorEdgeCases: Edge cases and error handling
    - TestServiceNodeSelectorThreadSafety: Async lock thread safety tests

Coverage Goals:
    - >90% code coverage for service
    - All selection strategies tested
    - Round robin state tracking verified
    - Edge cases (empty list, single item) covered
    - Thread safety with asyncio.Lock verified

Related Tickets:
    - OMN-1135: ServiceCapabilityQuery Implementation
    - OMN-1134: Registry Projection Extensions for Capabilities
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.enums import EnumRegistrationState
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.models.projection import ModelRegistrationProjection
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.services import EnumSelectionStrategy, ServiceNodeSelector

# =============================================================================
# Test Constants
# =============================================================================

DEFAULT_DOMAIN = "registration"
"""Default domain for registration queries."""


# =============================================================================
# Test Helpers
# =============================================================================


def create_mock_projection(
    entity_id: UUID | None = None,
    state: EnumRegistrationState = EnumRegistrationState.ACTIVE,
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
    capability_tags: list[str] | None = None,
) -> ModelRegistrationProjection:
    """Create a mock projection with sensible defaults.

    Args:
        entity_id: Optional fixed entity ID (for stable test assertions)
        state: Registration state (default: ACTIVE)
        node_type: Node kind (default: EFFECT)
        capability_tags: List of capability tags

    Returns:
        ModelRegistrationProjection with test data
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id if entity_id is not None else uuid4(),
        domain=DEFAULT_DOMAIN,
        current_state=state,
        node_type=node_type,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(),
        capability_tags=capability_tags or [],
        intent_types=[],
        protocols=[],
        contract_type="effect",
        last_applied_event_id=uuid4(),
        last_applied_offset=100,
        registered_at=now,
        updated_at=now,
    )


def create_candidate_list(count: int = 5) -> list[ModelRegistrationProjection]:
    """Create a list of candidate projections for testing.

    Args:
        count: Number of candidates to create

    Returns:
        List of ModelRegistrationProjection instances
    """
    return [create_mock_projection() for _ in range(count)]


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.unit
class TestEnumSelectionStrategy:
    """Tests for EnumSelectionStrategy enum definition."""

    def test_strategy_has_first_value(self) -> None:
        """Should have FIRST strategy value."""
        assert hasattr(EnumSelectionStrategy, "FIRST")
        assert EnumSelectionStrategy.FIRST.value == "first"

    def test_strategy_has_random_value(self) -> None:
        """Should have RANDOM strategy value."""
        assert hasattr(EnumSelectionStrategy, "RANDOM")
        assert EnumSelectionStrategy.RANDOM.value == "random"

    def test_strategy_has_round_robin_value(self) -> None:
        """Should have ROUND_ROBIN strategy value."""
        assert hasattr(EnumSelectionStrategy, "ROUND_ROBIN")
        assert EnumSelectionStrategy.ROUND_ROBIN.value == "round_robin"

    def test_strategy_has_least_loaded_value(self) -> None:
        """Should have LEAST_LOADED strategy value."""
        assert hasattr(EnumSelectionStrategy, "LEAST_LOADED")
        assert EnumSelectionStrategy.LEAST_LOADED.value == "least_loaded"

    def test_strategy_is_string_enum(self) -> None:
        """Should be a StrEnum for string serialization."""
        assert str(EnumSelectionStrategy.FIRST) == "first"
        assert str(EnumSelectionStrategy.RANDOM) == "random"


@pytest.mark.unit
class TestSelectionStrategyFirst:
    """Tests for FIRST selection strategy."""

    @pytest.mark.asyncio
    async def test_first_strategy_returns_first_candidate(self) -> None:
        """Should always return the first candidate in the list.

        Given: A list of candidates
        When: select is called with FIRST strategy
        Then: Returns the first element in the list
        """
        candidates = create_candidate_list(5)

        selector = ServiceNodeSelector()
        result = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.FIRST,
        )

        assert result is not None
        assert result.entity_id == candidates[0].entity_id

    @pytest.mark.asyncio
    async def test_first_strategy_with_single_candidate(self) -> None:
        """Should return the only candidate when list has one element."""
        candidates = create_candidate_list(1)

        selector = ServiceNodeSelector()
        result = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.FIRST,
        )

        assert result is not None
        assert result.entity_id == candidates[0].entity_id

    @pytest.mark.asyncio
    async def test_first_strategy_is_deterministic(self) -> None:
        """Should return the same result on repeated calls."""
        candidates = create_candidate_list(3)

        selector = ServiceNodeSelector()
        results = [
            await selector.select(
                candidates=candidates, strategy=EnumSelectionStrategy.FIRST
            )
            for _ in range(10)
        ]

        # All results should be the same
        assert all(r.entity_id == results[0].entity_id for r in results)


@pytest.mark.unit
class TestSelectionStrategyRandom:
    """Tests for RANDOM selection strategy."""

    @pytest.mark.asyncio
    async def test_random_strategy_returns_valid_candidate(self) -> None:
        """Should return a candidate that exists in the input list.

        Given: A list of candidates
        When: select is called with RANDOM strategy
        Then: Returns one of the candidates from the list
        """
        candidates = create_candidate_list(5)

        selector = ServiceNodeSelector()
        result = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.RANDOM,
        )

        assert result is not None
        assert result.entity_id in [c.entity_id for c in candidates]

    @pytest.mark.asyncio
    async def test_random_strategy_with_single_candidate(self) -> None:
        """Should return the only candidate when list has one element."""
        candidates = create_candidate_list(1)

        selector = ServiceNodeSelector()
        result = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.RANDOM,
        )

        assert result is not None
        assert result.entity_id == candidates[0].entity_id

    @pytest.mark.asyncio
    async def test_random_strategy_distributes_selections(self) -> None:
        """Should distribute selections across candidates over many calls.

        Note: This is a statistical test - with 1000 selections across 5 candidates,
        each should get at least some selections (probability of any one getting
        0 is astronomically low with proper randomness).
        """
        candidates = create_candidate_list(5)
        candidate_ids = {c.entity_id for c in candidates}

        selector = ServiceNodeSelector()
        selected_ids = set()

        for _ in range(1000):
            result = await selector.select(
                candidates=candidates,
                strategy=EnumSelectionStrategy.RANDOM,
            )
            if result:
                selected_ids.add(result.entity_id)

        # All candidates should have been selected at least once
        assert selected_ids == candidate_ids


@pytest.mark.unit
class TestSelectionStrategyRoundRobin:
    """Tests for ROUND_ROBIN selection strategy."""

    @pytest.mark.asyncio
    async def test_round_robin_cycles_through_candidates(self) -> None:
        """Should cycle through candidates in order.

        Given: A list of 3 candidates
        When: select is called 6 times with ROUND_ROBIN strategy
        Then: Returns candidates in rotating order: 0, 1, 2, 0, 1, 2
        """
        candidates = create_candidate_list(3)

        selector = ServiceNodeSelector()
        selection_key = "test.capability"

        results = []
        for _ in range(6):
            result = await selector.select(
                candidates=candidates,
                strategy=EnumSelectionStrategy.ROUND_ROBIN,
                selection_key=selection_key,
            )
            results.append(result)

        # Should cycle: 0, 1, 2, 0, 1, 2
        expected_pattern = candidates + candidates  # Repeat pattern
        for i, result in enumerate(results):
            assert result.entity_id == expected_pattern[i].entity_id

    @pytest.mark.asyncio
    async def test_round_robin_tracks_state_per_key(self) -> None:
        """Should maintain separate rotation state for different keys.

        Given: Two different selection keys
        When: select is called with different keys
        Then: Each key tracks its own rotation state independently
        """
        candidates = create_candidate_list(3)

        selector = ServiceNodeSelector()

        # Key A: select twice
        a_result_1 = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.ROUND_ROBIN,
            selection_key="key_a",
        )
        a_result_2 = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.ROUND_ROBIN,
            selection_key="key_a",
        )

        # Key B: select once (should start fresh)
        b_result_1 = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.ROUND_ROBIN,
            selection_key="key_b",
        )

        # Key A should be at position 2 (0, 1 already selected)
        # Key B should be at position 0 (fresh start)
        assert a_result_1.entity_id == candidates[0].entity_id
        assert a_result_2.entity_id == candidates[1].entity_id
        assert b_result_1.entity_id == candidates[0].entity_id

    @pytest.mark.asyncio
    async def test_round_robin_resets_when_candidate_list_changes(self) -> None:
        """Should handle changing candidate list sizes gracefully.

        Given: Round robin state for a key
        When: Candidate list size changes
        Then: Continues cycling with modulo arithmetic
        """
        candidates_3 = create_candidate_list(3)
        candidates_5 = create_candidate_list(5)

        selector = ServiceNodeSelector()
        selection_key = "resizing_test"

        # Use 3-item list
        await selector.select(
            candidates=candidates_3,
            strategy=EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )
        await selector.select(
            candidates=candidates_3,
            strategy=EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )

        # Switch to 5-item list - should handle gracefully
        result = await selector.select(
            candidates=candidates_5,
            strategy=EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )

        # Result should be valid (from 5-item list)
        assert result is not None
        assert result.entity_id in [c.entity_id for c in candidates_5]

    @pytest.mark.asyncio
    async def test_round_robin_with_single_candidate(self) -> None:
        """Should always return the same candidate when list has one element."""
        candidates = create_candidate_list(1)

        selector = ServiceNodeSelector()
        selection_key = "single_test"

        results = [
            await selector.select(
                candidates=candidates,
                strategy=EnumSelectionStrategy.ROUND_ROBIN,
                selection_key=selection_key,
            )
            for _ in range(5)
        ]

        # All results should be the same (only one candidate)
        assert all(r.entity_id == candidates[0].entity_id for r in results)


@pytest.mark.unit
class TestSelectionStrategyLeastLoaded:
    """Tests for LEAST_LOADED selection strategy (RuntimeHostError)."""

    @pytest.mark.asyncio
    async def test_least_loaded_raises_runtime_host_error(self) -> None:
        """Should raise RuntimeHostError when LEAST_LOADED is requested.

        Given: LEAST_LOADED strategy (requires load metrics)
        When: select is called with multiple candidates
        Then: Raises RuntimeHostError with clear message and context

        Note: Full implementation requires load metrics from monitoring system.
        """
        candidates = create_candidate_list(3)

        selector = ServiceNodeSelector()

        with pytest.raises(RuntimeHostError) as exc_info:
            await selector.select(
                candidates=candidates,
                strategy=EnumSelectionStrategy.LEAST_LOADED,
            )

        assert "LEAST_LOADED selection strategy is not yet implemented" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_least_loaded_with_empty_list(self) -> None:
        """Should return None when candidate list is empty (before RuntimeHostError)."""
        candidates: list[ModelRegistrationProjection] = []

        selector = ServiceNodeSelector()
        result = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.LEAST_LOADED,
        )

        # Empty list returns None before checking strategy
        assert result is None

    @pytest.mark.asyncio
    async def test_least_loaded_with_single_candidate(self) -> None:
        """Should return single candidate without error (optimization path)."""
        candidates = create_candidate_list(1)

        selector = ServiceNodeSelector()
        result = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.LEAST_LOADED,
        )

        # Single candidate returns early before checking strategy
        assert result is not None
        assert result.entity_id == candidates[0].entity_id


@pytest.mark.unit
class TestServiceNodeSelectorEdgeCases:
    """Tests for edge cases in ServiceNodeSelector."""

    @pytest.mark.asyncio
    async def test_empty_candidates_returns_none(self) -> None:
        """Should return None when candidate list is empty.

        Given: An empty candidate list
        When: select is called with any strategy
        Then: Returns None (not raises exception)
        """
        candidates: list[ModelRegistrationProjection] = []

        selector = ServiceNodeSelector()

        # Test with each strategy (empty list returns early before strategy check)
        for strategy in EnumSelectionStrategy:
            result = await selector.select(
                candidates=candidates,
                strategy=strategy,
            )
            assert result is None

    @pytest.mark.asyncio
    async def test_single_candidate_all_strategies(self) -> None:
        """Should return the single candidate for all strategies.

        Given: A list with exactly one candidate
        When: select is called with any strategy
        Then: Returns that candidate (single candidate optimization)
        """
        candidates = create_candidate_list(1)

        selector = ServiceNodeSelector()

        # Single candidate returns early before strategy check
        for strategy in EnumSelectionStrategy:
            result = await selector.select(
                candidates=candidates,
                strategy=strategy,
            )
            assert result is not None
            assert result.entity_id == candidates[0].entity_id

    @pytest.mark.asyncio
    async def test_round_robin_uses_default_key_when_not_specified(self) -> None:
        """Should use default key for ROUND_ROBIN when not specified.

        Given: ROUND_ROBIN strategy
        When: select is called without selection_key parameter
        Then: Uses a default key ("_default") for state tracking
        """
        candidates = create_candidate_list(3)

        selector = ServiceNodeSelector()

        # Implementation uses default key when not provided
        result = await selector.select(
            candidates=candidates,
            strategy=EnumSelectionStrategy.ROUND_ROBIN,
            # No selection_key provided
        )

        # Should work with default key
        assert result is not None
        assert result.entity_id == candidates[0].entity_id

        # Verify state was tracked under "_default" key
        state = await selector.get_round_robin_state()
        assert "_default" in state


@pytest.mark.unit
class TestServiceNodeSelectorInstantiation:
    """Tests for ServiceNodeSelector instantiation and configuration."""

    def test_instantiation_with_defaults(self) -> None:
        """Should instantiate with default configuration."""
        selector = ServiceNodeSelector()
        assert selector is not None
        # Verify lock is initialized
        assert hasattr(selector, "_round_robin_lock")
        assert isinstance(selector._round_robin_lock, asyncio.Lock)

    @pytest.mark.asyncio
    async def test_round_robin_state_isolation(self) -> None:
        """Each ServiceNodeSelector instance should have isolated state.

        Given: Two separate ServiceNodeSelector instances
        When: Both use ROUND_ROBIN with the same key
        Then: Their rotation states are independent
        """
        candidates = create_candidate_list(3)

        selector1 = ServiceNodeSelector()
        selector2 = ServiceNodeSelector()
        selection_key = "shared_key"

        # Advance selector1 twice
        await selector1.select(
            candidates,
            EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )
        await selector1.select(
            candidates,
            EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )

        # selector2 should start fresh
        result = await selector2.select(
            candidates,
            EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )
        assert result.entity_id == candidates[0].entity_id  # First candidate

    @pytest.mark.asyncio
    async def test_reset_round_robin_state(self) -> None:
        """Should be able to reset round robin state for a key.

        Given: Round robin state accumulated for a key
        When: reset_round_robin_state is called
        Then: Next selection starts from beginning
        """
        candidates = create_candidate_list(3)

        selector = ServiceNodeSelector()
        selection_key = "clearable_key"

        # Advance state
        await selector.select(
            candidates,
            EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )
        await selector.select(
            candidates,
            EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )

        # Reset state for specific key
        await selector.reset_round_robin_state(selection_key)

        # Should start from beginning
        result = await selector.select(
            candidates,
            EnumSelectionStrategy.ROUND_ROBIN,
            selection_key=selection_key,
        )
        assert result.entity_id == candidates[0].entity_id


@pytest.mark.unit
class TestSelectionStrategyExhaustiveness:
    """Tests for selection strategy exhaustiveness handling."""

    @pytest.mark.asyncio
    async def test_all_enum_values_have_explicit_handlers(self) -> None:
        """Should explicitly handle all EnumSelectionStrategy values.

        Given: All defined EnumSelectionStrategy values
        When: select is called with each strategy
        Then: Each strategy is handled explicitly (not falling through to else)

        This test verifies the exhaustiveness guarantee: the service uses
        assert_never to ensure compile-time checking that all enum values
        are handled. At runtime, this test confirms no unexpected fallback
        behavior exists.

        Note: LEAST_LOADED raises RuntimeHostError (expected behavior).
        """
        candidates = create_candidate_list(3)
        selector = ServiceNodeSelector()

        handled_strategies = []
        for strategy in EnumSelectionStrategy:
            try:
                result = await selector.select(
                    candidates=candidates,
                    strategy=strategy,
                    selection_key="exhaustiveness_test",
                )
                # Strategy was handled (returned a result)
                assert result is not None
                handled_strategies.append(strategy)
            except RuntimeHostError:
                # LEAST_LOADED raises RuntimeHostError - this is expected
                assert strategy == EnumSelectionStrategy.LEAST_LOADED
                handled_strategies.append(strategy)
            except AssertionError:
                # assert_never raises AssertionError for unhandled cases
                pytest.fail(
                    f"Strategy {strategy} not handled! "
                    "Update select() to handle this enum value."
                )

        # Verify all strategies were tested
        assert set(handled_strategies) == set(EnumSelectionStrategy)

    def test_selection_strategy_enum_values_documented(self) -> None:
        """Should have consistent enum values with service docstring.

        Given: The EnumSelectionStrategy enum
        When: Checking documented strategies
        Then: All four strategies (FIRST, RANDOM, ROUND_ROBIN, LEAST_LOADED) exist
        """
        expected_strategies = {"FIRST", "RANDOM", "ROUND_ROBIN", "LEAST_LOADED"}
        actual_strategies = {s.name for s in EnumSelectionStrategy}

        assert actual_strategies == expected_strategies, (
            f"Enum strategies changed! Expected {expected_strategies}, "
            f"got {actual_strategies}. Update select() method to handle "
            "new strategies."
        )


@pytest.mark.unit
class TestServiceNodeSelectorThreadSafety:
    """Tests for thread safety with asyncio.Lock."""

    @pytest.mark.asyncio
    async def test_concurrent_round_robin_selections(self) -> None:
        """Should handle concurrent round-robin selections safely.

        Given: A shared ServiceNodeSelector instance
        When: Multiple coroutines make concurrent round-robin selections
        Then: All selections are valid and no race conditions occur
        """
        candidates = create_candidate_list(5)
        selector = ServiceNodeSelector()
        selection_key = "concurrent_test"

        async def make_selection() -> ModelRegistrationProjection | None:
            return await selector.select(
                candidates=candidates,
                strategy=EnumSelectionStrategy.ROUND_ROBIN,
                selection_key=selection_key,
            )

        # Run 100 concurrent selections
        tasks = [make_selection() for _ in range(100)]
        results = await asyncio.gather(*tasks)

        # All results should be valid candidates
        for result in results:
            assert result is not None
            assert result.entity_id in [c.entity_id for c in candidates]

    @pytest.mark.asyncio
    async def test_concurrent_state_access(self) -> None:
        """Should handle concurrent get/reset operations safely.

        Given: A shared ServiceNodeSelector instance
        When: Multiple coroutines concurrently get and reset state
        Then: No race conditions occur
        """
        candidates = create_candidate_list(3)
        selector = ServiceNodeSelector()
        selection_key = "state_test"

        async def select_and_get_state() -> dict[str, int]:
            await selector.select(
                candidates=candidates,
                strategy=EnumSelectionStrategy.ROUND_ROBIN,
                selection_key=selection_key,
            )
            return await selector.get_round_robin_state()

        async def reset_state() -> None:
            await selector.reset_round_robin_state(selection_key)

        # Run concurrent operations
        tasks = []
        for i in range(50):
            if i % 5 == 0:
                tasks.append(reset_state())
            else:
                tasks.append(select_and_get_state())

        # Should complete without errors
        await asyncio.gather(*tasks)


__all__: list[str] = []
