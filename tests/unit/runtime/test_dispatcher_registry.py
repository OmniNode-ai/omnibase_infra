# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for DispatcherRegistry.

Tests the dispatcher registry functionality including:
- Dispatcher registration and validation
- Execution shape validation at registration time
- Freeze pattern behavior
- Dispatcher lookup by category and message type
- Thread safety contract enforcement

Related:
    - OMN-934: Dispatcher registry for message dispatch engine
    - src/omnibase_infra/runtime/dispatcher_registry.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from omnibase_core.enums.enum_core_error_code import EnumCoreErrorCode
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.errors.model_onex_error import ModelOnexError

from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.registry_dispatcher import (
    DispatcherRegistry,
)

# Import shared conformance helper
from tests.conftest import assert_dispatcher_protocol_interface


class MockMessageDispatcher:
    """Mock dispatcher implementing ProtocolMessageDispatcher for testing."""

    def __init__(
        self,
        dispatcher_id: str,
        category: EnumMessageCategory,
        node_kind: EnumNodeKind,
        message_types: set[str] | None = None,
    ) -> None:
        self._dispatcher_id = dispatcher_id
        self._category = category
        self._node_kind = node_kind
        self._message_types = message_types or set()

    @property
    def dispatcher_id(self) -> str:
        return self._dispatcher_id

    @property
    def category(self) -> EnumMessageCategory:
        return self._category

    @property
    def message_types(self) -> set[str]:
        return self._message_types

    @property
    def node_kind(self) -> EnumNodeKind:
        return self._node_kind

    async def handle(self, envelope: object) -> ModelDispatchResult:
        return ModelDispatchResult(
            status=EnumDispatchStatus.SUCCESS,
            topic="test.events",
            dispatcher_id=self._dispatcher_id,
            started_at=datetime(2025, 1, 1, tzinfo=UTC),
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def dispatcher_registry() -> DispatcherRegistry:
    """Create a fresh DispatcherRegistry for tests."""
    return DispatcherRegistry()


@pytest.fixture
def event_reducer_dispatcher() -> MockMessageDispatcher:
    """Create a dispatcher for EVENT -> REDUCER (valid shape)."""
    return MockMessageDispatcher(
        dispatcher_id="event-reducer-dispatcher",
        category=EnumMessageCategory.EVENT,
        node_kind=EnumNodeKind.REDUCER,
        message_types={"UserCreated", "UserUpdated"},
    )


@pytest.fixture
def event_compute_dispatcher() -> MockMessageDispatcher:
    """Create a dispatcher for EVENT -> COMPUTE (valid shape)."""
    return MockMessageDispatcher(
        dispatcher_id="event-compute-dispatcher",
        category=EnumMessageCategory.EVENT,
        node_kind=EnumNodeKind.COMPUTE,
    )


@pytest.fixture
def command_orchestrator_dispatcher() -> MockMessageDispatcher:
    """Create a dispatcher for COMMAND -> ORCHESTRATOR (valid shape)."""
    return MockMessageDispatcher(
        dispatcher_id="command-orchestrator-dispatcher",
        category=EnumMessageCategory.COMMAND,
        node_kind=EnumNodeKind.ORCHESTRATOR,
        message_types={"CreateOrder", "CancelOrder"},
    )


@pytest.fixture
def command_effect_dispatcher() -> MockMessageDispatcher:
    """Create a dispatcher for COMMAND -> EFFECT (valid shape)."""
    return MockMessageDispatcher(
        dispatcher_id="command-effect-dispatcher",
        category=EnumMessageCategory.COMMAND,
        node_kind=EnumNodeKind.EFFECT,
    )


@pytest.fixture
def intent_orchestrator_dispatcher() -> MockMessageDispatcher:
    """Create a dispatcher for INTENT -> ORCHESTRATOR (valid shape)."""
    return MockMessageDispatcher(
        dispatcher_id="intent-orchestrator-dispatcher",
        category=EnumMessageCategory.INTENT,
        node_kind=EnumNodeKind.ORCHESTRATOR,
        message_types={"SendEmail", "NotifyUser"},
    )


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolMessageDispatcher:
    """Tests for ProtocolMessageDispatcher protocol.

    These tests demonstrate protocol validation patterns for dispatchers.
    Per ONEX conventions, protocol conformance is verified via duck typing
    by checking for required properties and methods.

    Validation Approaches:
        1. Duck typing: Check for required attributes/methods using hasattr()
           and callable(). This is the ONEX-preferred approach.

        2. DispatcherRegistry.register_dispatcher(): Comprehensive validation
           including property type checking, execution shape validation,
           and detailed error messages for debugging.
    """

    def test_mock_dispatcher_implements_protocol(
        self, event_reducer_dispatcher: MockMessageDispatcher
    ) -> None:
        """MockMessageDispatcher should implement ProtocolMessageDispatcher.

        Per ONEX conventions, protocol conformance is verified via duck typing
        by checking for required properties and methods.
        """
        # Use shared conformance helper for dispatcher protocol verification
        assert_dispatcher_protocol_interface(event_reducer_dispatcher)

    def test_duck_typing_rejects_non_dispatcher(self) -> None:
        """Duck typing should identify objects that don't implement the protocol.

        This demonstrates using duck typing to reject objects that don't have
        the required dispatcher interface.
        """

        # A plain object without dispatcher properties fails duck typing check
        class NotADispatcher:
            pass

        required_props = ["dispatcher_id", "category", "message_types", "node_kind"]
        not_a_dispatcher = NotADispatcher()
        has_all_props = all(hasattr(not_a_dispatcher, prop) for prop in required_props)
        assert not has_all_props, "NotADispatcher should not have all required props"

        # An object with some but not all properties also fails
        class PartialDispatcher:
            @property
            def dispatcher_id(self) -> str:
                return "partial"

            # Missing: category, message_types, node_kind, handle

        partial_dispatcher = PartialDispatcher()
        has_all_props = all(
            hasattr(partial_dispatcher, prop) for prop in required_props
        )
        assert not has_all_props, "PartialDispatcher should not have all required props"

    def test_duck_typing_for_protocol_validation_pattern(
        self, event_reducer_dispatcher: MockMessageDispatcher
    ) -> None:
        """Demonstrate the recommended pattern for protocol validation.

        Use duck typing for quick structural checks, then let
        DispatcherRegistry.register_dispatcher() perform comprehensive
        validation including execution shape checking.
        """
        # Pattern: Duck typing check before registration (ONEX convention)
        required_attrs = [
            "dispatcher_id",
            "category",
            "message_types",
            "node_kind",
            "handle",
        ]
        has_required = all(
            hasattr(event_reducer_dispatcher, attr) for attr in required_attrs
        )

        if has_required and callable(event_reducer_dispatcher.handle):
            # Proceed with registration - comprehensive validation happens here
            registry = DispatcherRegistry()
            registry.register_dispatcher(event_reducer_dispatcher)
            assert registry.dispatcher_count == 1
        else:
            # This branch would handle non-compliant objects
            pytest.fail("Dispatcher should implement ProtocolMessageDispatcher")

    def test_dispatcher_has_required_properties(
        self, event_reducer_dispatcher: MockMessageDispatcher
    ) -> None:
        """Dispatcher should have all required properties."""
        assert event_reducer_dispatcher.dispatcher_id == "event-reducer-dispatcher"
        assert event_reducer_dispatcher.category == EnumMessageCategory.EVENT
        assert event_reducer_dispatcher.node_kind == EnumNodeKind.REDUCER
        assert event_reducer_dispatcher.message_types == {"UserCreated", "UserUpdated"}

    @pytest.mark.asyncio
    async def test_dispatcher_handle_method(
        self, event_reducer_dispatcher: MockMessageDispatcher
    ) -> None:
        """Dispatcher should have async handle method that returns ModelDispatchResult."""
        result = await event_reducer_dispatcher.handle(MagicMock())
        assert isinstance(result, ModelDispatchResult)
        assert result.status == EnumDispatchStatus.SUCCESS


# =============================================================================
# Registration Tests
# =============================================================================


class TestDispatcherRegistration:
    """Tests for dispatcher registration."""

    def test_register_valid_dispatcher(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should register a valid dispatcher successfully."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        assert dispatcher_registry.dispatcher_count == 1

    def test_register_multiple_dispatchers(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
        event_compute_dispatcher: MockMessageDispatcher,
        command_orchestrator_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should register multiple dispatchers successfully."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.register_dispatcher(event_compute_dispatcher)
        dispatcher_registry.register_dispatcher(command_orchestrator_dispatcher)
        assert dispatcher_registry.dispatcher_count == 3

    def test_register_with_custom_message_types(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should allow overriding message_types at registration."""
        custom_types = {"CustomEvent1", "CustomEvent2"}
        dispatcher_registry.register_dispatcher(
            event_reducer_dispatcher,
            message_types=custom_types,
        )
        dispatcher_registry.freeze()

        # Should find dispatcher with custom types
        dispatchers = dispatcher_registry.get_dispatchers(
            EnumMessageCategory.EVENT,
            message_type="CustomEvent1",
        )
        assert len(dispatchers) == 1

        # Should NOT find with original types
        dispatchers = dispatcher_registry.get_dispatchers(
            EnumMessageCategory.EVENT,
            message_type="UserCreated",
        )
        assert len(dispatchers) == 0

    def test_register_none_dispatcher_raises(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """Should raise when registering None dispatcher."""
        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(None)  # type: ignore[arg-type]

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER
        assert "None" in str(exc_info.value)

    def test_register_duplicate_dispatcher_id_raises(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should raise when registering duplicate dispatcher_id."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)

        # Create another dispatcher with same ID but valid execution shape
        duplicate = MockMessageDispatcher(
            dispatcher_id="event-reducer-dispatcher",  # Same ID
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.COMPUTE,  # Valid shape: EVENT -> COMPUTE
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(duplicate)

        assert exc_info.value.error_code == EnumCoreErrorCode.DUPLICATE_REGISTRATION

    def test_register_dispatcher_missing_dispatcher_id_raises(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """Should raise when dispatcher lacks dispatcher_id property."""
        dispatcher = MagicMock(spec=[])  # No dispatcher_id attribute

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(dispatcher)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER
        assert "dispatcher_id" in str(exc_info.value)

    def test_register_dispatcher_missing_category_raises(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """Should raise when dispatcher lacks category property."""
        dispatcher = MagicMock()
        dispatcher.dispatcher_id = "test-dispatcher"
        del dispatcher.category

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(dispatcher)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_PARAMETER
        assert "category" in str(exc_info.value)


# =============================================================================
# Execution Shape Validation Tests
# =============================================================================


class TestExecutionShapeValidation:
    """Tests for execution shape validation at registration time."""

    def test_valid_shapes_event_to_reducer(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """EVENT -> REDUCER is a valid execution shape."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        assert dispatcher_registry.dispatcher_count == 1

    def test_valid_shapes_event_to_compute(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_compute_dispatcher: MockMessageDispatcher,
    ) -> None:
        """EVENT -> COMPUTE is a valid execution shape."""
        dispatcher_registry.register_dispatcher(event_compute_dispatcher)
        assert dispatcher_registry.dispatcher_count == 1

    def test_valid_shapes_command_to_orchestrator(
        self,
        dispatcher_registry: DispatcherRegistry,
        command_orchestrator_dispatcher: MockMessageDispatcher,
    ) -> None:
        """COMMAND -> ORCHESTRATOR is a valid execution shape."""
        dispatcher_registry.register_dispatcher(command_orchestrator_dispatcher)
        assert dispatcher_registry.dispatcher_count == 1

    def test_valid_shapes_command_to_effect(
        self,
        dispatcher_registry: DispatcherRegistry,
        command_effect_dispatcher: MockMessageDispatcher,
    ) -> None:
        """COMMAND -> EFFECT is a valid execution shape."""
        dispatcher_registry.register_dispatcher(command_effect_dispatcher)
        assert dispatcher_registry.dispatcher_count == 1

    def test_valid_shapes_intent_to_orchestrator(
        self,
        dispatcher_registry: DispatcherRegistry,
        intent_orchestrator_dispatcher: MockMessageDispatcher,
    ) -> None:
        """INTENT -> ORCHESTRATOR is a valid execution shape."""
        dispatcher_registry.register_dispatcher(intent_orchestrator_dispatcher)
        assert dispatcher_registry.dispatcher_count == 1

    def test_invalid_shape_command_to_reducer_raises(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """COMMAND -> REDUCER is an invalid execution shape."""
        invalid_dispatcher = MockMessageDispatcher(
            dispatcher_id="invalid-dispatcher",
            category=EnumMessageCategory.COMMAND,
            node_kind=EnumNodeKind.REDUCER,  # Invalid!
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(invalid_dispatcher)

        assert exc_info.value.error_code == EnumCoreErrorCode.VALIDATION_FAILED
        assert "invalid execution shape" in str(exc_info.value).lower()

    def test_invalid_shape_intent_to_reducer_raises(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """INTENT -> REDUCER is an invalid execution shape."""
        invalid_dispatcher = MockMessageDispatcher(
            dispatcher_id="invalid-dispatcher",
            category=EnumMessageCategory.INTENT,
            node_kind=EnumNodeKind.REDUCER,  # Invalid!
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(invalid_dispatcher)

        assert exc_info.value.error_code == EnumCoreErrorCode.VALIDATION_FAILED

    def test_invalid_shape_event_to_effect_raises(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """EVENT -> EFFECT is an invalid execution shape."""
        invalid_dispatcher = MockMessageDispatcher(
            dispatcher_id="invalid-dispatcher",
            category=EnumMessageCategory.EVENT,
            node_kind=EnumNodeKind.EFFECT,  # Invalid!
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(invalid_dispatcher)

        assert exc_info.value.error_code == EnumCoreErrorCode.VALIDATION_FAILED

    def test_invalid_shape_intent_to_effect_raises(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """INTENT -> EFFECT is an invalid execution shape."""
        invalid_dispatcher = MockMessageDispatcher(
            dispatcher_id="invalid-dispatcher",
            category=EnumMessageCategory.INTENT,
            node_kind=EnumNodeKind.EFFECT,  # Invalid!
        )

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(invalid_dispatcher)

        assert exc_info.value.error_code == EnumCoreErrorCode.VALIDATION_FAILED


# =============================================================================
# Freeze Pattern Tests
# =============================================================================


class TestFreezePattern:
    """Tests for freeze-after-init pattern."""

    def test_initial_state_not_frozen(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """Registry should start unfrozen."""
        assert dispatcher_registry.is_frozen is False

    def test_freeze_sets_frozen_flag(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """freeze() should set is_frozen to True."""
        dispatcher_registry.freeze()
        assert dispatcher_registry.is_frozen is True

    def test_freeze_is_idempotent(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """Calling freeze() multiple times should be safe."""
        dispatcher_registry.freeze()
        dispatcher_registry.freeze()  # Should not raise
        dispatcher_registry.freeze()  # Should not raise
        assert dispatcher_registry.is_frozen is True

    def test_register_after_freeze_raises(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should raise when registering after freeze."""
        dispatcher_registry.freeze()

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.register_dispatcher(event_reducer_dispatcher)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE
        assert "frozen" in str(exc_info.value).lower()

    def test_unregister_after_freeze_raises(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should raise when unregistering after freeze."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.freeze()

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.unregister_dispatcher("event-reducer-dispatcher")

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE

    def test_get_dispatchers_before_freeze_raises(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should raise when getting dispatchers before freeze."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.get_dispatchers(EnumMessageCategory.EVENT)

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE
        assert "freeze" in str(exc_info.value).lower()

    def test_get_dispatcher_by_id_before_freeze_raises(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should raise when getting dispatcher by ID before freeze."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)

        with pytest.raises(ModelOnexError) as exc_info:
            dispatcher_registry.get_dispatcher_by_id("event-reducer-dispatcher")

        assert exc_info.value.error_code == EnumCoreErrorCode.INVALID_STATE


# =============================================================================
# Dispatcher Lookup Tests
# =============================================================================


class TestDispatcherLookup:
    """Tests for dispatcher lookup after freeze."""

    def test_get_dispatchers_by_category(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
        event_compute_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should return all dispatchers for a category."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.register_dispatcher(event_compute_dispatcher)
        dispatcher_registry.freeze()

        dispatchers = dispatcher_registry.get_dispatchers(EnumMessageCategory.EVENT)
        assert len(dispatchers) == 2

    def test_get_dispatchers_by_category_and_type(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
        event_compute_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should filter dispatchers by message type."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.register_dispatcher(event_compute_dispatcher)
        dispatcher_registry.freeze()

        # event_reducer_dispatcher has specific types, event_compute has none (all)
        dispatchers = dispatcher_registry.get_dispatchers(
            EnumMessageCategory.EVENT,
            message_type="UserCreated",
        )
        # Both should match: reducer has UserCreated, compute accepts all
        assert len(dispatchers) == 2

        # Only compute should match (reducer doesn't have UnknownEvent)
        dispatchers = dispatcher_registry.get_dispatchers(
            EnumMessageCategory.EVENT,
            message_type="UnknownEvent",
        )
        assert len(dispatchers) == 1
        assert dispatchers[0].dispatcher_id == "event-compute-dispatcher"

    def test_get_dispatchers_empty_category(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should return empty list for category with no dispatchers."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.freeze()

        dispatchers = dispatcher_registry.get_dispatchers(EnumMessageCategory.COMMAND)
        assert dispatchers == []

    def test_get_dispatcher_by_id_found(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should return dispatcher when found by ID."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.freeze()

        dispatcher = dispatcher_registry.get_dispatcher_by_id(
            "event-reducer-dispatcher"
        )
        assert dispatcher is event_reducer_dispatcher

    def test_get_dispatcher_by_id_not_found(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should return None when dispatcher ID not found."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.freeze()

        dispatcher = dispatcher_registry.get_dispatcher_by_id("nonexistent-dispatcher")
        assert dispatcher is None


# =============================================================================
# Unregistration Tests
# =============================================================================


class TestUnregistration:
    """Tests for dispatcher unregistration."""

    def test_unregister_existing_dispatcher(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Should unregister existing dispatcher and return True."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        assert dispatcher_registry.dispatcher_count == 1

        result = dispatcher_registry.unregister_dispatcher("event-reducer-dispatcher")
        assert result is True
        assert dispatcher_registry.dispatcher_count == 0

    def test_unregister_nonexistent_dispatcher(
        self, dispatcher_registry: DispatcherRegistry
    ) -> None:
        """Should return False when dispatcher not found."""
        result = dispatcher_registry.unregister_dispatcher("nonexistent-dispatcher")
        assert result is False

    def test_unregister_removes_from_category_index(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
        event_compute_dispatcher: MockMessageDispatcher,
    ) -> None:
        """Unregistered dispatcher should be removed from category index."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.register_dispatcher(event_compute_dispatcher)

        dispatcher_registry.unregister_dispatcher("event-reducer-dispatcher")
        dispatcher_registry.freeze()

        dispatchers = dispatcher_registry.get_dispatchers(EnumMessageCategory.EVENT)
        assert len(dispatchers) == 1
        assert dispatchers[0].dispatcher_id == "event-compute-dispatcher"


# =============================================================================
# String Representation Tests
# =============================================================================


class TestStringRepresentation:
    """Tests for __str__ and __repr__."""

    def test_str_representation(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """__str__ should return formatted summary."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)

        result = str(dispatcher_registry)
        assert "DispatcherRegistry" in result
        assert "dispatchers=1" in result
        assert "frozen=False" in result

    def test_str_representation_frozen(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """__str__ should show frozen state."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)
        dispatcher_registry.freeze()

        result = str(dispatcher_registry)
        assert "frozen=True" in result

    def test_repr_representation(
        self,
        dispatcher_registry: DispatcherRegistry,
        event_reducer_dispatcher: MockMessageDispatcher,
    ) -> None:
        """__repr__ should return detailed representation."""
        dispatcher_registry.register_dispatcher(event_reducer_dispatcher)

        result = repr(dispatcher_registry)
        assert "DispatcherRegistry(" in result
        assert "dispatchers=" in result
        assert "categories=" in result
        assert "event-reducer-dispatcher" in result
