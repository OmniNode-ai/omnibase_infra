# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Comprehensive tests for custom __bool__ behavior across validation models.

This test module verifies the non-standard __bool__ behavior of Pydantic models
that override the default behavior. Standard Pydantic models always return True
when evaluated in a boolean context (bool(model) == True for any valid instance).
However, several ONEX models override __bool__ to enable idiomatic conditional checks.

**IMPORTANT**: This behavior differs from typical Pydantic where bool(model) is always True.

Models covered:
    - ModelReducerExecutionResult: True if has_intents (intents tuple non-empty)
    - ModelCategoryMatchResult: True if matched is True
    - ModelValidationOutcome: True if is_valid is True
    - ModelExecutionShapeValidationResult: True if passed is True
    - ModelDispatchOutputs: True if topics list is non-empty
    - ModelLifecycleResult: True if success is True

See Also:
    CLAUDE.md section "Custom `__bool__` for Result Models" for documentation standards.

.. versionadded:: 0.7.0
    Created as part of PR #92 review to add missing test coverage for __bool__.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from omnibase_core.enums import EnumNodeKind
from pydantic import BaseModel, ValidationError

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.enums.enum_node_output_type import EnumNodeOutputType
from omnibase_infra.models.dispatch.model_dispatch_outputs import ModelDispatchOutputs
from omnibase_infra.models.validation.model_category_match_result import (
    ModelCategoryMatchResult,
)
from omnibase_infra.models.validation.model_execution_shape_validation_result import (
    ModelExecutionShapeValidationResult,
)
from omnibase_infra.models.validation.model_validation_outcome import (
    ModelValidationOutcome,
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
from omnibase_infra.runtime.models.model_lifecycle_result import ModelLifecycleResult


def _make_consul_intent() -> ModelConsulRegistrationIntent:
    """Helper to create a valid ModelConsulRegistrationIntent for tests."""
    node_id = uuid4()
    correlation_id = uuid4()
    payload = ModelConsulIntentPayload(service_name="test-service")
    return ModelConsulRegistrationIntent(
        operation="register",
        node_id=node_id,
        correlation_id=correlation_id,
        payload=payload,
    )


def _make_postgres_intent() -> ModelPostgresUpsertIntent:
    """Helper to create a valid ModelPostgresUpsertIntent for tests."""
    node_id = uuid4()
    correlation_id = uuid4()
    payload = ModelPostgresIntentPayload(
        node_id=node_id,
        node_type=EnumNodeKind.EFFECT,
        correlation_id=correlation_id,
        timestamp="2025-01-01T00:00:00Z",
    )
    return ModelPostgresUpsertIntent(
        operation="upsert",
        node_id=node_id,
        correlation_id=correlation_id,
        payload=payload,
    )


class TestModelReducerExecutionResultBool:
    """Tests for ModelReducerExecutionResult.__bool__ non-standard behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when intents are present.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with no intents evaluates to False in boolean context!

    The design enables idiomatic "if result:" checks for work-to-do scenarios:
        if result:
            # Process intents - there is work to do
        else:
            # No intents - nothing to process
    """

    def test_bool_true_when_has_intents(self) -> None:
        """Result evaluates to True when intents tuple is non-empty.

        This is the 'truthy' case: there is work to be done.
        """
        state = ModelReducerState.initial()
        intent = _make_consul_intent()
        result = ModelReducerExecutionResult(state=state, intents=(intent,))

        assert bool(result) is True
        assert result.has_intents is True
        assert result  # Direct conditional check

    def test_bool_false_when_no_intents(self) -> None:
        """Result evaluates to False when intents tuple is empty.

        WARNING: This is non-standard Pydantic behavior!
        A valid model instance returns False because there's no work to do.
        """
        state = ModelReducerState.initial()
        result = ModelReducerExecutionResult(state=state, intents=())

        # This is the critical test - valid model, but bool is False
        assert bool(result) is False
        assert result.has_intents is False
        assert not result  # Direct conditional check

    def test_bool_false_for_empty_factory(self) -> None:
        """ModelReducerExecutionResult.empty() evaluates to False.

        The empty() factory creates a valid result with no intents.
        """
        result = ModelReducerExecutionResult.empty()

        assert bool(result) is False
        assert result.state is not None  # Valid model
        assert result.intents == ()

    def test_bool_false_for_no_change_factory(self) -> None:
        """ModelReducerExecutionResult.no_change() evaluates to False.

        The no_change() factory preserves state but has no intents.
        """
        state = ModelReducerState(pending_registrations=5)
        result = ModelReducerExecutionResult.no_change(state)

        assert bool(result) is False
        assert result.state.pending_registrations == 5  # State preserved
        assert result.intents == ()

    def test_bool_true_for_with_intents_factory(self) -> None:
        """ModelReducerExecutionResult.with_intents() evaluates to True.

        The with_intents() factory creates a result with work to do.
        """
        state = ModelReducerState.initial()
        consul_intent = _make_consul_intent()
        postgres_intent = _make_postgres_intent()
        result = ModelReducerExecutionResult.with_intents(
            state=state,
            intents=[consul_intent, postgres_intent],
        )

        assert bool(result) is True
        assert result.intent_count == 2

    def test_bool_with_multiple_intents(self) -> None:
        """Multiple intents all result in True evaluation."""
        state = ModelReducerState.initial()
        intents = tuple(_make_consul_intent() for _ in range(5))
        result = ModelReducerExecutionResult(state=state, intents=intents)

        assert bool(result) is True
        assert result.intent_count == 5

    def test_conditional_pattern_usage(self) -> None:
        """Demonstrate the idiomatic conditional usage pattern.

        This test shows the intended usage pattern that the custom __bool__
        enables - clean conditional checks for work presence.
        """
        state = ModelReducerState.initial()

        # Pattern 1: Result with work
        work_result = ModelReducerExecutionResult.with_intents(
            state=state,
            intents=[_make_consul_intent()],
        )

        work_done = False
        if work_result:
            work_done = True
        assert work_done is True

        # Pattern 2: Result without work
        no_work_result = ModelReducerExecutionResult.empty()

        skipped = False
        if not no_work_result:
            skipped = True
        assert skipped is True

    def test_has_intents_matches_bool(self) -> None:
        """has_intents property should match __bool__ return value.

        This explicit test verifies the contract: __bool__ returns has_intents.
        Both truthy and falsy cases must maintain this invariant.
        """
        # Falsy case: empty result
        empty_result = ModelReducerExecutionResult.empty()
        assert empty_result.has_intents == bool(empty_result)
        assert empty_result.has_intents is False

        # Truthy case: result with intents
        state = ModelReducerState.initial()
        with_intents = ModelReducerExecutionResult.with_intents(
            state=state,
            intents=[_make_consul_intent()],
        )
        assert with_intents.has_intents == bool(with_intents)
        assert with_intents.has_intents is True

        # Additional edge case: no_change factory
        no_change = ModelReducerExecutionResult.no_change(state)
        assert no_change.has_intents == bool(no_change)
        assert no_change.has_intents is False


class TestModelCategoryMatchResultFactoryMethods:
    """Dedicated tests for ModelCategoryMatchResult factory methods.

    This test class verifies that each factory method creates the correct state,
    validates parameters appropriately, and produces models with expected property
    values. These tests are separate from __bool__ behavior tests to ensure
    comprehensive coverage of factory semantics.

    Factory Methods:
        - matched_with_category(category): Creates match with specific category
        - matched_without_category(): Creates match without specific category
        - not_matched(): Creates non-match result

    .. versionadded:: 0.7.0
        Added as part of PR #92 review to add dedicated factory method coverage.
    """

    def test_matched_with_category_creates_correct_state_event(self) -> None:
        """matched_with_category creates result with matched=True and correct category.

        Verifies state for EnumMessageCategory.EVENT.
        """
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        assert result.matched is True
        assert result.category == EnumMessageCategory.EVENT
        assert result.has_category is True
        assert result.is_message_category is True
        assert result.is_projection is False

    def test_matched_with_category_creates_correct_state_command(self) -> None:
        """matched_with_category creates result with matched=True and correct category.

        Verifies state for EnumMessageCategory.COMMAND.
        """
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.COMMAND
        )

        assert result.matched is True
        assert result.category == EnumMessageCategory.COMMAND
        assert result.has_category is True
        assert result.is_message_category is True
        assert result.is_projection is False

    def test_matched_with_category_creates_correct_state_intent(self) -> None:
        """matched_with_category creates result with matched=True and correct category.

        Verifies state for EnumMessageCategory.INTENT.
        """
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.INTENT
        )

        assert result.matched is True
        assert result.category == EnumMessageCategory.INTENT
        assert result.has_category is True
        assert result.is_message_category is True
        assert result.is_projection is False

    def test_matched_with_category_creates_correct_state_projection(self) -> None:
        """matched_with_category creates result with matched=True and correct category.

        Verifies state for EnumNodeOutputType.PROJECTION - the only non-message
        category that can be matched.
        """
        result = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.PROJECTION
        )

        assert result.matched is True
        assert result.category == EnumNodeOutputType.PROJECTION
        assert result.has_category is True
        assert (
            result.is_message_category is False
        )  # PROJECTION is not a message category
        assert result.is_projection is True

    def test_matched_without_category_creates_correct_state(self) -> None:
        """matched_without_category creates result with matched=True but category=None.

        This factory is for cases where a match is detected but the specific
        category cannot be determined (e.g., generic @message_type decorator).
        """
        result = ModelCategoryMatchResult.matched_without_category()

        assert result.matched is True
        assert result.category is None
        assert result.has_category is False
        assert result.is_message_category is False
        assert result.is_projection is False

    def test_not_matched_creates_correct_state(self) -> None:
        """not_matched creates result with matched=False and category=None.

        This factory is for cases where no match pattern was found.
        """
        result = ModelCategoryMatchResult.not_matched()

        assert result.matched is False
        assert result.category is None
        assert result.has_category is False
        assert result.is_message_category is False
        assert result.is_projection is False

    def test_factory_results_are_frozen(self) -> None:
        """All factory results are immutable (frozen model).

        The model uses ConfigDict(frozen=True), so attempting to modify
        any attribute should raise an error.
        """
        results = [
            ModelCategoryMatchResult.matched_with_category(EnumMessageCategory.EVENT),
            ModelCategoryMatchResult.matched_without_category(),
            ModelCategoryMatchResult.not_matched(),
        ]

        for result in results:
            with pytest.raises(ValidationError):
                result.matched = not result.matched  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("category", "expected_is_message", "expected_is_projection"),
        [
            (EnumMessageCategory.EVENT, True, False),
            (EnumMessageCategory.COMMAND, True, False),
            (EnumMessageCategory.INTENT, True, False),
            (EnumNodeOutputType.PROJECTION, False, True),
        ],
    )
    def test_matched_with_category_property_behavior(
        self,
        category: EnumMessageCategory | EnumNodeOutputType,
        expected_is_message: bool,
        expected_is_projection: bool,
    ) -> None:
        """Verify is_message_category and is_projection for all category types."""
        result = ModelCategoryMatchResult.matched_with_category(category)

        assert result.is_message_category is expected_is_message
        assert result.is_projection is expected_is_projection

    def test_matched_with_category_returns_new_instance_each_call(self) -> None:
        """Each factory call returns a new instance (no instance caching)."""
        result1 = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )
        result2 = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        # Same values but different instances
        assert result1 == result2  # Value equality
        assert result1 is not result2  # Instance inequality

    def test_matched_without_category_returns_new_instance_each_call(self) -> None:
        """Each factory call returns a new instance (no instance caching)."""
        result1 = ModelCategoryMatchResult.matched_without_category()
        result2 = ModelCategoryMatchResult.matched_without_category()

        assert result1 == result2
        assert result1 is not result2

    def test_not_matched_returns_new_instance_each_call(self) -> None:
        """Each factory call returns a new instance (no instance caching)."""
        result1 = ModelCategoryMatchResult.not_matched()
        result2 = ModelCategoryMatchResult.not_matched()

        assert result1 == result2
        assert result1 is not result2

    def test_factory_methods_return_correct_type(self) -> None:
        """All factory methods return ModelCategoryMatchResult instances."""
        assert isinstance(
            ModelCategoryMatchResult.matched_with_category(EnumMessageCategory.EVENT),
            ModelCategoryMatchResult,
        )
        assert isinstance(
            ModelCategoryMatchResult.matched_without_category(),
            ModelCategoryMatchResult,
        )
        assert isinstance(
            ModelCategoryMatchResult.not_matched(),
            ModelCategoryMatchResult,
        )

    def test_direct_construction_vs_factory_equivalence(self) -> None:
        """Direct construction produces equivalent results to factory methods."""
        # matched_with_category equivalence
        factory_result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.COMMAND
        )
        direct_result = ModelCategoryMatchResult(
            matched=True, category=EnumMessageCategory.COMMAND
        )
        assert factory_result == direct_result

        # matched_without_category equivalence
        factory_without = ModelCategoryMatchResult.matched_without_category()
        direct_without = ModelCategoryMatchResult(matched=True, category=None)
        assert factory_without == direct_without

        # not_matched equivalence
        factory_not = ModelCategoryMatchResult.not_matched()
        direct_not = ModelCategoryMatchResult(matched=False, category=None)
        assert factory_not == direct_not


class TestModelCategoryMatchResultBool:
    """Tests for ModelCategoryMatchResult.__bool__ behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when matched is True.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with matched=False evaluates to False in boolean context!

    The design enables idiomatic category matching checks:
        if result:
            # Category was matched
            category = result.category
        else:
            # No match found
            pass
    """

    def test_bool_true_when_matched_with_category(self) -> None:
        """Result evaluates to True when matched with a specific category."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )

        assert bool(result) is True
        assert result.matched is True
        assert result.category == EnumMessageCategory.EVENT
        assert result  # Direct conditional

    def test_bool_true_when_matched_without_category(self) -> None:
        """Result evaluates to True when matched, even without specific category.

        Some decorators indicate a match but don't specify which category.
        """
        result = ModelCategoryMatchResult.matched_without_category()

        assert bool(result) is True
        assert result.matched is True
        assert result.category is None
        assert result.has_category is False

    def test_bool_false_when_not_matched(self) -> None:
        """Result evaluates to False when not matched.

        WARNING: This is non-standard Pydantic behavior!
        A valid model with matched=False returns False.
        """
        result = ModelCategoryMatchResult.not_matched()

        assert bool(result) is False
        assert result.matched is False
        assert result.category is None
        assert not result  # Direct conditional

    @pytest.mark.parametrize(
        "category",
        [
            EnumMessageCategory.EVENT,
            EnumMessageCategory.COMMAND,
            EnumMessageCategory.INTENT,
            EnumNodeOutputType.PROJECTION,
        ],
    )
    def test_bool_true_for_all_category_types(
        self, category: EnumMessageCategory | EnumNodeOutputType
    ) -> None:
        """All category types result in True when matched."""
        result = ModelCategoryMatchResult.matched_with_category(category)

        assert bool(result) is True
        assert result.matched is True
        assert result.category == category

    def test_conditional_pattern_usage(self) -> None:
        """Demonstrate the idiomatic conditional usage pattern."""
        # Pattern 1: Matched result
        matched_result = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.COMMAND
        )

        found_category = None
        if matched_result:
            found_category = matched_result.category
        assert found_category == EnumMessageCategory.COMMAND

        # Pattern 2: Not matched result
        not_matched_result = ModelCategoryMatchResult.not_matched()

        handled = False
        if not_matched_result:
            handled = True
        assert handled is False  # Was not handled because not matched

    def test_matched_matches_bool(self) -> None:
        """matched property should match __bool__ return value.

        This explicit test verifies the contract: __bool__ returns matched.
        Both truthy and falsy cases must maintain this invariant.
        """
        # Falsy case: not matched
        not_matched = ModelCategoryMatchResult.not_matched()
        assert not_matched.matched == bool(not_matched)
        assert not_matched.matched is False

        # Truthy case: matched with category
        matched_with = ModelCategoryMatchResult.matched_with_category(
            EnumMessageCategory.EVENT
        )
        assert matched_with.matched == bool(matched_with)
        assert matched_with.matched is True

        # Truthy case: matched without category
        matched_without = ModelCategoryMatchResult.matched_without_category()
        assert matched_without.matched == bool(matched_without)
        assert matched_without.matched is True


class TestModelValidationOutcomeBool:
    """Tests for ModelValidationOutcome.__bool__ behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when is_valid is True.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with is_valid=False evaluates to False in boolean context!

    The __bool__ returns is_valid, enabling idiomatic validation checks:
        if outcome:
            # Validation passed
        else:
            # Validation failed
            print(outcome.error_message)
    """

    def test_bool_true_when_valid(self) -> None:
        """Outcome evaluates to True when validation passed."""
        outcome = ModelValidationOutcome.success()

        assert bool(outcome) is True
        assert outcome.is_valid is True
        assert outcome  # Direct conditional

    def test_bool_false_when_invalid(self) -> None:
        """Outcome evaluates to False when validation failed."""
        outcome = ModelValidationOutcome.failure("Invalid input")

        assert bool(outcome) is False
        assert outcome.is_valid is False
        assert not outcome  # Direct conditional

    def test_conditional_with_raise_pattern(self) -> None:
        """Demonstrate combined conditional and raise_if_invalid pattern."""
        success = ModelValidationOutcome.success()
        failure = ModelValidationOutcome.failure("Bad data")

        # Pattern 1: Check then raise
        if not failure:
            with pytest.raises(ValueError, match="Bad data"):
                failure.raise_if_invalid()

        # Pattern 2: Success passes through
        if success:
            success.raise_if_invalid()  # Should not raise


class TestModelExecutionShapeValidationResultBool:
    """Tests for ModelExecutionShapeValidationResult.__bool__ behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when passed is True.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with passed=False evaluates to False in boolean context!

    The __bool__ returns passed, enabling idiomatic CI gate checks:
        if result:
            # Validation passed, can proceed
        else:
            # Has blocking violations
            for line in result.format_for_ci():
                print(line)
    """

    def test_bool_true_when_passed(self) -> None:
        """Result evaluates to True when validation passed."""
        result = ModelExecutionShapeValidationResult.success()

        assert bool(result) is True
        assert result.passed is True
        assert result  # Direct conditional

    def test_bool_false_when_failed(self) -> None:
        """Result evaluates to False when validation failed."""
        result = ModelExecutionShapeValidationResult(
            passed=False,
            violations=[],  # Could have violations
        )

        assert bool(result) is False
        assert result.passed is False
        assert not result  # Direct conditional


class TestModelDispatchOutputsBool:
    """Tests for ModelDispatchOutputs.__bool__ behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when topics list is non-empty.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with empty topics evaluates to False in boolean context!

    The __bool__ returns True if topics list is non-empty.
    This enables checking if any output was produced:
        if outputs:
            # There are topics to process
        else:
            # No output was produced
    """

    def test_bool_true_when_has_topics(self) -> None:
        """Outputs evaluates to True when topics list is non-empty."""
        outputs = ModelDispatchOutputs(topics=["dev.user.events.v1"])

        assert bool(outputs) is True
        assert len(outputs) == 1
        assert outputs  # Direct conditional

    def test_bool_false_when_no_topics(self) -> None:
        """Outputs evaluates to False when topics list is empty."""
        outputs = ModelDispatchOutputs(topics=[])

        assert bool(outputs) is False
        assert len(outputs) == 0
        assert not outputs  # Direct conditional

    def test_bool_false_for_default_construction(self) -> None:
        """Default construction (no topics) evaluates to False."""
        outputs = ModelDispatchOutputs()

        assert bool(outputs) is False
        assert len(outputs) == 0

    def test_bool_with_multiple_topics(self) -> None:
        """Multiple topics result in True evaluation."""
        outputs = ModelDispatchOutputs(
            topics=[
                "dev.user.events.v1",
                "dev.notification.commands.v1",
                "dev.audit.events.v1",
            ]
        )

        assert bool(outputs) is True
        assert len(outputs) == 3


class TestModelLifecycleResultBool:
    """Tests for ModelLifecycleResult.__bool__ behavior.

    Warning:
        This model overrides __bool__ to return True ONLY when success is True.
        This differs from standard Pydantic where bool(model) is always True.
        A valid model with success=False evaluates to False in boolean context!

    The __bool__ returns success, enabling idiomatic lifecycle checks:
        if result:
            # Shutdown/operation succeeded
        else:
            # Failed - check error_message
            print(result.error_message)
    """

    def test_bool_true_when_succeeded(self) -> None:
        """Result evaluates to True when operation succeeded."""
        result = ModelLifecycleResult.succeeded("kafka")

        assert bool(result) is True
        assert result.success is True
        assert result  # Direct conditional

    def test_bool_false_when_failed(self) -> None:
        """Result evaluates to False when operation failed."""
        result = ModelLifecycleResult.failed("database", "Connection timeout")

        assert bool(result) is False
        assert result.success is False
        assert not result  # Direct conditional

    def test_conditional_error_handling_pattern(self) -> None:
        """Demonstrate the idiomatic error handling pattern."""
        success = ModelLifecycleResult.succeeded("kafka")
        failure = ModelLifecycleResult.failed("db", "Connection refused")

        # Pattern 1: Check success
        if success:
            pass  # Continue normally
        else:
            pytest.fail("Should have succeeded")

        # Pattern 2: Check failure
        error_handled = False
        if not failure:
            error_handled = True
            assert failure.error_message == "Connection refused"
        assert error_handled is True


class TestStandardPydanticBoolComparison:
    """Demonstrate the difference between standard Pydantic and custom __bool__.

    These tests explicitly show how ONEX models differ from standard Pydantic
    behavior where bool(model) is always True for any valid instance.
    """

    def test_standard_pydantic_always_true(self) -> None:
        """Standard Pydantic models always evaluate to True.

        This test shows the standard behavior that ONEX models intentionally
        override for more idiomatic conditional checks.
        """

        class StandardModel(BaseModel):
            value: bool = False

        model = StandardModel(value=False)

        # Standard Pydantic: bool(model) is always True
        assert bool(model) is True

    def test_onex_models_can_be_false(self) -> None:
        """ONEX result models can evaluate to False when appropriate.

        This is the key difference from standard Pydantic behavior.
        """
        # All these are valid model instances, but evaluate to False
        assert bool(ModelReducerExecutionResult.empty()) is False
        assert bool(ModelCategoryMatchResult.not_matched()) is False
        assert bool(ModelValidationOutcome.failure("error")) is False
        assert (
            bool(ModelExecutionShapeValidationResult(passed=False, violations=[]))
            is False
        )
        assert bool(ModelDispatchOutputs()) is False
        assert bool(ModelLifecycleResult.failed("x", "error")) is False


class TestEdgeCases:
    """Edge case tests for __bool__ behavior."""

    def test_reducer_result_single_intent_is_truthy(self) -> None:
        """Single intent is enough to be truthy."""
        state = ModelReducerState.initial()
        result = ModelReducerExecutionResult(
            state=state,
            intents=(_make_consul_intent(),),
        )
        assert bool(result) is True

    def test_category_match_projection_is_truthy(self) -> None:
        """PROJECTION category match is still truthy (matched=True)."""
        result = ModelCategoryMatchResult.matched_with_category(
            EnumNodeOutputType.PROJECTION
        )
        assert bool(result) is True
        assert result.is_projection is True

    def test_validation_outcome_empty_error_on_success(self) -> None:
        """Success outcome has empty error_message and is truthy."""
        outcome = ModelValidationOutcome.success()
        assert bool(outcome) is True
        assert outcome.error_message == ""
        assert outcome.has_error is False

    def test_lifecycle_result_is_success_vs_bool(self) -> None:
        """is_success() and bool() return the same value."""
        success = ModelLifecycleResult.succeeded("test")
        failure = ModelLifecycleResult.failed("test", "error")

        assert success.is_success() == bool(success) is True
        assert failure.is_success() == bool(failure) is False
