# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for non-standard __bool__ behavior in result models.

This module tests the custom __bool__ implementations in models that override
the default Pydantic behavior. Standard Pydantic models return True for any
valid instance, but these models return True only when specific conditions
are met.

Models Tested:
    - ModelReducerExecutionResult: Returns True only if has_intents (intents non-empty)
    - ModelCategoryMatchResult: Returns True only if matched is True

Why This Matters:
    These models enable idiomatic conditional checks like `if result:` to mean
    "if there is work to do" rather than "if the model exists". This is a
    significant deviation from typical Pydantic behavior and requires explicit
    documentation and thorough testing.

Example:
    >>> result = ModelReducerExecutionResult.empty()
    >>> if result:
    ...     print("Has work")
    ... else:
    ...     print("No work")
    No work
    >>> # Note: A standard Pydantic model would print "Has work" here!

Related:
    - CLAUDE.md: Section on "Custom __bool__ for Result Models"
    - ADR: docs/decisions/adr-custom-bool-result-models.md (if exists)
    - PR #92 review feedback: CRITICAL - missing tests for __bool__ behavior

.. versionadded:: 0.7.0
    Created as part of PR #92 review to address CRITICAL testing gap.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.models.validation.model_category_match_result import (
    ModelCategoryMatchResult,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_intent_payload import (
    ModelConsulIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_registration_intent import (
    ModelConsulRegistrationIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_intent_payload import (
    ModelPostgresIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_upsert_intent import (
    ModelPostgresUpsertIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_execution_result import (
    ModelReducerExecutionResult,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_state import (
    ModelReducerState,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_consul_intent() -> ModelConsulRegistrationIntent:
    """Create a sample Consul registration intent for testing."""
    return ModelConsulRegistrationIntent(
        operation="register",
        node_id=uuid4(),
        correlation_id=uuid4(),
        payload=ModelConsulIntentPayload(
            service_name="test-service",
            tags=("test", "unit"),
            meta=(("env", "test"),),
        ),
    )


@pytest.fixture
def sample_postgres_intent() -> ModelPostgresUpsertIntent:
    """Create a sample PostgreSQL upsert intent for testing."""
    return ModelPostgresUpsertIntent(
        operation="upsert",
        node_id=uuid4(),
        correlation_id=uuid4(),
        payload=ModelPostgresIntentPayload(
            node_id=uuid4(),
            node_type="effect",
            node_version="1.0.0",
            correlation_id=uuid4(),
            timestamp="2025-01-01T00:00:00Z",
        ),
    )


@pytest.fixture
def initial_state() -> ModelReducerState:
    """Create an initial reducer state for testing."""
    return ModelReducerState.initial()


# ============================================================================
# Tests for ModelReducerExecutionResult.__bool__
# ============================================================================


@pytest.mark.unit
class TestModelReducerExecutionResultBool:
    """Tests for non-standard __bool__ behavior in ModelReducerExecutionResult.

    The ModelReducerExecutionResult overrides __bool__ to return True only when
    the intents tuple is non-empty. This enables idiomatic usage like:

        if result:  # True only if there are intents to process
            execute_intents(result.intents)

    This is a significant deviation from standard Pydantic behavior where
    bool(model) always returns True for any valid model instance.
    """

    def test_bool_true_with_single_consul_intent(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """Verify bool(result) is True when a single Consul intent is present."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent,),
        )

        assert bool(result) is True
        assert result  # Idiomatic usage should also pass

    def test_bool_true_with_single_postgres_intent(
        self,
        initial_state: ModelReducerState,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """Verify bool(result) is True when a single PostgreSQL intent is present."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_postgres_intent,),
        )

        assert bool(result) is True
        assert result  # Idiomatic usage should also pass

    def test_bool_true_with_multiple_intents(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """Verify bool(result) is True when multiple intents are present."""
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent, sample_postgres_intent),
        )

        assert bool(result) is True
        assert result  # Idiomatic usage
        assert result.intent_count == 2

    def test_bool_false_without_intents(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """Verify bool(result) is False when no intents are present.

        This is the key test demonstrating the non-standard behavior:
        a valid, constructed model instance returns False for bool().
        """
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(),  # Empty tuple
        )

        assert bool(result) is False

        # Demonstrate the if-statement behavior
        if result:
            pytest.fail("Expected False for result with no intents")

    def test_bool_false_for_empty_factory(self) -> None:
        """Verify bool(ModelReducerExecutionResult.empty()) is False.

        The empty() factory method creates a result with initial state and
        no intents - bool() should return False.
        """
        result = ModelReducerExecutionResult.empty()

        assert bool(result) is False
        assert result.state is not None  # Model is valid
        assert result.intents == ()  # But no intents
        assert not result  # Idiomatic usage - not result should be True

    def test_bool_false_for_no_change_factory(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """Verify bool(ModelReducerExecutionResult.no_change(state)) is False.

        The no_change() factory preserves state but has no intents.
        """
        result = ModelReducerExecutionResult.no_change(initial_state)

        assert bool(result) is False
        assert result.state == initial_state  # State preserved
        assert not result  # Idiomatic: no work to do

    def test_bool_true_for_with_intents_factory(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """Verify bool(ModelReducerExecutionResult.with_intents(...)) is True."""
        result = ModelReducerExecutionResult.with_intents(
            state=initial_state,
            intents=[sample_consul_intent],  # List is converted to tuple
        )

        assert bool(result) is True
        assert result  # Idiomatic: has work to do

    def test_bool_matches_has_intents_property(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """Verify bool(result) == result.has_intents in all cases."""
        # Case 1: Empty result
        empty_result = ModelReducerExecutionResult.empty()
        assert bool(empty_result) == empty_result.has_intents

        # Case 2: No change result
        no_change_result = ModelReducerExecutionResult.no_change(initial_state)
        assert bool(no_change_result) == no_change_result.has_intents

        # Case 3: Result with intents
        with_intents_result = ModelReducerExecutionResult.with_intents(
            state=initial_state,
            intents=[sample_consul_intent],
        )
        assert bool(with_intents_result) == with_intents_result.has_intents

    def test_bool_differs_from_none_check(
        self,
    ) -> None:
        """Verify that bool(result) differs from `result is not None`.

        This demonstrates why the non-standard __bool__ is useful but also
        potentially confusing. Users should understand the difference.
        """
        result = ModelReducerExecutionResult.empty()

        # Model exists (is not None) - this is True
        assert result is not None

        # But bool(result) is False because no intents
        assert bool(result) is False

        # Documenting the potentially surprising behavior
        if result is not None:
            # We're here - the model exists
            pass

        if not result:
            # We're also here - despite the model existing!
            # This is because __bool__ returns False for empty intents
            pass

    def test_bool_in_conditional_expression(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """Verify __bool__ works correctly in conditional expressions."""
        empty_result = ModelReducerExecutionResult.empty()
        work_result = ModelReducerExecutionResult.with_intents(
            state=initial_state,
            intents=[sample_consul_intent],
        )

        # Ternary expression
        message_empty = "work" if empty_result else "no work"
        message_work = "work" if work_result else "no work"

        assert message_empty == "no work"
        assert message_work == "work"

    def test_bool_in_all_any(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """Verify __bool__ works with all() and any() built-ins."""
        empty1 = ModelReducerExecutionResult.empty()
        empty2 = ModelReducerExecutionResult.no_change(initial_state)
        with_work = ModelReducerExecutionResult.with_intents(
            state=initial_state,
            intents=[sample_consul_intent],
        )

        # any() should find the one with work
        results_with_one_work = [empty1, with_work, empty2]
        assert any(results_with_one_work) is True

        # all() should be False if any result has no work
        assert all(results_with_one_work) is False

        # all() with all empty should be False
        all_empty = [empty1, empty2]
        assert all(all_empty) is False

        # any() with all empty should be False
        assert any(all_empty) is False


# ============================================================================
# Tests for ModelCategoryMatchResult.__bool__
# ============================================================================


@pytest.mark.unit
class TestModelCategoryMatchResultBool:
    """Tests for non-standard __bool__ behavior in ModelCategoryMatchResult.

    The ModelCategoryMatchResult overrides __bool__ to return True only when
    the matched field is True. This enables idiomatic usage like:

        if result:  # True only if a category was matched
            process_category(result.category)

    This is a significant deviation from standard Pydantic behavior where
    bool(model) always returns True for any valid model instance.
    """

    def test_bool_true_when_matched_with_event_category(self) -> None:
        """Verify bool(result) is True when matched=True with EVENT category."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        assert bool(result) is True
        assert result  # Idiomatic usage
        assert result.matched is True
        assert result.category == EnumMessageCategory.EVENT

    def test_bool_true_when_matched_with_command_category(self) -> None:
        """Verify bool(result) is True when matched=True with COMMAND category."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.COMMAND
        )

        assert bool(result) is True
        assert result  # Idiomatic usage

    def test_bool_true_when_matched_with_intent_category(self) -> None:
        """Verify bool(result) is True when matched=True with INTENT category."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.INTENT
        )

        assert bool(result) is True
        assert result  # Idiomatic usage

    def test_bool_true_when_matched_with_projection(self) -> None:
        """Verify bool(result) is True when matched=True with PROJECTION output type."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.PROJECTION
        )

        assert bool(result) is True
        assert result  # Idiomatic usage
        assert result.is_projection is True

    def test_bool_true_when_matched_without_category(self) -> None:
        """Verify bool(result) is True when matched=True but category is None.

        This tests the case where a decorator was found but the specific
        category couldn't be determined (e.g., generic @message_type decorator).
        """
        result = ModelCategoryMatchResult.matched_without_category()

        assert bool(result) is True
        assert result  # Idiomatic usage
        assert result.matched is True
        assert result.category is None
        assert result.has_category is False

    def test_bool_false_when_not_matched(self) -> None:
        """Verify bool(result) is False when matched=False.

        This is the key test demonstrating the non-standard behavior:
        a valid, constructed model instance returns False for bool().
        """
        result = ModelCategoryMatchResult.not_matched()

        assert bool(result) is False

        # Demonstrate the if-statement behavior
        if result:
            pytest.fail("Expected False for not_matched result")

    def test_bool_false_for_not_matched_factory(self) -> None:
        """Verify bool(ModelCategoryMatchResult.not_matched()) is False."""
        result = ModelCategoryMatchResult.not_matched()

        assert bool(result) is False
        assert result.matched is False  # Model is valid
        assert result.category is None
        assert not result  # Idiomatic: not result should be True

    def test_bool_matches_matched_property(self) -> None:
        """Verify bool(result) == result.matched in all cases."""
        # Case 1: Not matched
        not_matched = ModelCategoryMatchResult.not_matched()
        assert bool(not_matched) == not_matched.matched

        # Case 2: Matched without category
        matched_no_cat = ModelCategoryMatchResult.matched_without_category()
        assert bool(matched_no_cat) == matched_no_cat.matched

        # Case 3: Matched with category
        matched_with_cat = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )
        assert bool(matched_with_cat) == matched_with_cat.matched

    def test_bool_differs_from_none_check(self) -> None:
        """Verify that bool(result) differs from `result is not None`.

        This demonstrates why the non-standard __bool__ is useful but also
        potentially confusing. Users should understand the difference.
        """
        result = ModelCategoryMatchResult.not_matched()

        # Model exists (is not None) - this is True
        assert result is not None

        # But bool(result) is False because not matched
        assert bool(result) is False

    def test_bool_in_conditional_expression(self) -> None:
        """Verify __bool__ works correctly in conditional expressions."""
        not_matched = ModelCategoryMatchResult.not_matched()
        matched = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        # Ternary expression
        message_not = "matched" if not_matched else "not matched"
        message_matched = "matched" if matched else "not matched"

        assert message_not == "not matched"
        assert message_matched == "matched"

    def test_bool_in_all_any(self) -> None:
        """Verify __bool__ works with all() and any() built-ins."""
        not_matched1 = ModelCategoryMatchResult.not_matched()
        not_matched2 = ModelCategoryMatchResult.not_matched()
        matched = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.COMMAND
        )

        # any() should find the one that matched
        results = [not_matched1, matched, not_matched2]
        assert any(results) is True

        # all() should be False if any result didn't match
        assert all(results) is False

        # all() with all not matched should be False
        all_not_matched = [not_matched1, not_matched2]
        assert all(all_not_matched) is False

        # any() with all not matched should be False
        assert any(all_not_matched) is False

    def test_bool_with_message_category_types(self) -> None:
        """Verify __bool__ works correctly with all EnumMessageCategory values."""
        for category in EnumMessageCategory:
            result = ModelCategoryMatchResult.matched_with_category(category)
            assert bool(result) is True
            assert result.matched is True
            assert result.category == category
            assert result.is_message_category is True
            assert result.is_projection is False

    def test_bool_with_node_output_type(self) -> None:
        """Verify __bool__ works correctly with EnumNodeOutputType.PROJECTION."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.PROJECTION
        )

        assert bool(result) is True
        assert result.matched is True
        assert result.category == EnumNodeOutputType.PROJECTION
        assert result.is_message_category is False
        assert result.is_projection is True


# ============================================================================
# Comparison Tests: Standard Pydantic vs Custom __bool__
# ============================================================================


@pytest.mark.unit
class TestBoolBehaviorDocumentation:
    """Tests that document and verify the deviation from standard Pydantic behavior.

    These tests serve as documentation for developers who may be surprised
    by the non-standard __bool__ behavior. They explicitly compare what
    standard Pydantic would do vs what these models do.
    """

    def test_standard_pydantic_always_true(self) -> None:
        """Demonstrate standard Pydantic: bool(model) is always True.

        For comparison, a standard Pydantic BaseModel returns True for any
        valid instance, regardless of field values.
        """
        from pydantic import BaseModel

        class StandardModel(BaseModel):
            value: int = 0
            name: str = ""

        # Even with "empty" values, bool() is True
        model = StandardModel()
        assert bool(model) is True  # Standard Pydantic behavior

    def test_reducer_result_differs_from_standard(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """Document how ModelReducerExecutionResult differs from standard Pydantic."""
        # Create a valid model with no intents
        result = ModelReducerExecutionResult(state=initial_state, intents=())

        # Model is fully valid
        assert result.state == initial_state
        assert result.intents == ()

        # BUT bool() is False - different from standard Pydantic!
        assert bool(result) is False

        # This is intentional to enable:
        if not result:
            # "No work to do" semantics
            pass

    def test_category_match_differs_from_standard(self) -> None:
        """Document how ModelCategoryMatchResult differs from standard Pydantic."""
        # Create a valid model with matched=False
        result = ModelCategoryMatchResult(matched=False, category=None)

        # Model is fully valid
        assert result.matched is False
        assert result.category is None

        # BUT bool() is False - different from standard Pydantic!
        assert bool(result) is False

        # This is intentional to enable:
        if not result:
            # "No match found" semantics
            pass

    def test_workarounds_for_existence_check(self) -> None:
        """Document how to check model existence vs work-to-do."""
        empty_result = ModelReducerExecutionResult.empty()

        # If you need to check if the model exists:
        assert empty_result is not None  # Works: checks identity

        # If you need to check if there's work:
        assert not empty_result  # Uses __bool__: checks has_intents

        # Explicit and recommended approach:
        assert not empty_result.has_intents  # Most readable


# ============================================================================
# Tests for ModelReducerExecutionResult Immutability (frozen=True)
# ============================================================================


@pytest.mark.unit
class TestModelReducerExecutionResultImmutability:
    """Tests for frozen model immutability in ModelReducerExecutionResult.

    ModelReducerExecutionResult is a frozen Pydantic model (frozen=True),
    which ensures thread safety and prevents accidental state mutation.
    Attempting to mutate any field should raise a ValidationError.

    The model uses tuple[ProtocolRegistrationIntent, ...] for the intents field
    instead of list to maintain full immutability - tuples are immutable
    containers, while lists would allow mutation even with frozen=True.

    Related:
        - CLAUDE.md: Section on "Frozen Model with Tuple Fields"
        - PR #92 review feedback: CRITICAL - frozen model immutability verification
        - Pydantic docs: https://docs.pydantic.dev/latest/concepts/config/#frozen

    .. versionadded:: 0.7.0
        Created as part of PR #92 review to verify frozen model behavior.
    """

    def test_mutation_of_state_field_raises_validation_error(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """Verify that attempting to mutate the state field raises ValidationError.

        Frozen Pydantic models prevent field assignment after construction.
        This ensures thread safety and immutable state semantics.
        """
        from pydantic import ValidationError

        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent,),
        )

        # Attempting to assign a new state should raise ValidationError
        new_state = ModelReducerState(pending_registrations=99)
        with pytest.raises(ValidationError) as exc_info:
            result.state = new_state  # type: ignore[misc]

        # Verify the error is about frozen instance
        assert (
            "frozen" in str(exc_info.value).lower()
            or "immutable" in str(exc_info.value).lower()
        )

    def test_mutation_of_intents_field_raises_validation_error(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """Verify that attempting to mutate the intents field raises ValidationError.

        The intents field is a tuple (immutable container) and the model is frozen.
        Attempting to replace the entire field should raise ValidationError.
        """
        from pydantic import ValidationError

        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent,),
        )

        # Attempting to assign new intents should raise ValidationError
        new_intents = (sample_postgres_intent,)
        with pytest.raises(ValidationError) as exc_info:
            result.intents = new_intents  # type: ignore[misc]

        # Verify the error is about frozen instance
        assert (
            "frozen" in str(exc_info.value).lower()
            or "immutable" in str(exc_info.value).lower()
        )

    def test_intents_field_is_tuple_not_list(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
    ) -> None:
        """Verify that the intents field is a tuple, not a list.

        Using tuple instead of list ensures full immutability:
        - Frozen model prevents field reassignment
        - Tuple prevents in-place mutation of the container contents

        This is critical for thread safety - concurrent access is safe
        because the data structure cannot be modified.
        """
        result = ModelReducerExecutionResult(
            state=initial_state,
            intents=(sample_consul_intent,),
        )

        # Verify the field is a tuple, not a list
        assert isinstance(result.intents, tuple)
        assert not isinstance(result.intents, list)

    def test_with_intents_factory_converts_list_to_tuple(
        self,
        initial_state: ModelReducerState,
        sample_consul_intent: ModelConsulRegistrationIntent,
        sample_postgres_intent: ModelPostgresUpsertIntent,
    ) -> None:
        """Verify that with_intents() factory converts list input to tuple.

        The factory method accepts Sequence (including list) for convenience,
        but always stores as tuple for immutability.
        """
        # Pass a list to the factory
        result = ModelReducerExecutionResult.with_intents(
            state=initial_state,
            intents=[sample_consul_intent, sample_postgres_intent],  # List input
        )

        # Result should have tuple, not list
        assert isinstance(result.intents, tuple)
        assert len(result.intents) == 2

    def test_empty_factory_returns_empty_tuple(self) -> None:
        """Verify that empty() factory returns an empty tuple for intents."""
        result = ModelReducerExecutionResult.empty()

        assert result.intents == ()
        assert isinstance(result.intents, tuple)
        assert len(result.intents) == 0

    def test_no_change_factory_returns_empty_tuple(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """Verify that no_change() factory returns an empty tuple for intents."""
        result = ModelReducerExecutionResult.no_change(initial_state)

        assert result.intents == ()
        assert isinstance(result.intents, tuple)

    def test_frozen_model_is_hashable(
        self,
        initial_state: ModelReducerState,
    ) -> None:
        """Verify that frozen model instances are hashable.

        Frozen Pydantic models should be hashable, which enables:
        - Use as dictionary keys
        - Use in sets
        - Memoization and caching

        Note: This requires all nested types to also be hashable.
        ModelReducerState is also frozen, and tuples are hashable.
        """
        result1 = ModelReducerExecutionResult(state=initial_state, intents=())
        result2 = ModelReducerExecutionResult(state=initial_state, intents=())

        # Should be hashable - if not, this will raise TypeError
        hash1 = hash(result1)
        hash2 = hash(result2)

        # Same content should produce same hash
        assert hash1 == hash2

        # Should be usable in a set
        result_set = {result1, result2}
        assert len(result_set) == 1  # Duplicates removed

    def test_model_config_has_frozen_true(self) -> None:
        """Verify that the model config explicitly sets frozen=True.

        This is a documentation test that ensures the model configuration
        is correct and won't be accidentally changed.
        """
        config = ModelReducerExecutionResult.model_config

        assert config.get("frozen") is True, (
            "ModelReducerExecutionResult must have frozen=True in model_config "
            "to ensure immutability and thread safety"
        )
