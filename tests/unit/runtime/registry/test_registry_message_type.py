# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for MessageTypeRegistry."""

import pytest
from omnibase_core.models.errors.model_onex_error import ModelOnexError

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.registry.model_domain_constraint import (
    ModelDomainConstraint,
)
from omnibase_infra.runtime.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)
from omnibase_infra.runtime.registry.registry_message_type import (
    MessageTypeRegistry,
    MessageTypeRegistryError,
    extract_domain_from_topic,
)


class TestExtractDomainFromTopic:
    """Tests for extract_domain_from_topic utility."""

    def test_onex_kafka_format(self) -> None:
        """Test ONEX Kafka format topic parsing."""
        assert extract_domain_from_topic("onex.registration.events") == "registration"
        assert extract_domain_from_topic("onex.user.commands") == "user"
        assert extract_domain_from_topic("onex.order.intents") == "order"

    def test_environment_aware_format(self) -> None:
        """Test environment-aware format topic parsing."""
        assert extract_domain_from_topic("dev.user.events.v1") == "user"
        assert extract_domain_from_topic("prod.order.commands.v2") == "order"
        assert extract_domain_from_topic("staging.billing.intents.v1") == "billing"

    def test_empty_topic(self) -> None:
        """Test empty topic returns None."""
        assert extract_domain_from_topic("") is None
        assert extract_domain_from_topic(None) is None  # type: ignore[arg-type]

    def test_invalid_topic(self) -> None:
        """Test invalid topic format."""
        # Single segment - not enough to extract domain
        assert extract_domain_from_topic("registration") is None


class TestMessageTypeRegistryRegistration:
    """Tests for registration functionality."""

    def test_register_message_type(self) -> None:
        """Test basic message type registration."""
        registry = MessageTypeRegistry()
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        registry.register_message_type(entry)

        assert "UserCreated" in registry
        assert registry.entry_count == 1

    def test_register_simple(self) -> None:
        """Test convenience registration method."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="OrderCreated",
            handler_id="order-handler",
            category=EnumMessageCategory.EVENT,
            domain="order",
            description="Order creation event",
        )

        assert "OrderCreated" in registry
        assert registry.entry_count == 1

    def test_register_multiple_handlers_fan_out(self) -> None:
        """Test registering multiple handlers for same message type (fan-out)."""
        registry = MessageTypeRegistry()

        # First registration
        entry1 = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        registry.register_message_type(entry1)

        # Second registration with different handler
        entry2 = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("audit-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        registry.register_message_type(entry2)

        registry.freeze()

        # Should have both handlers
        handlers = registry.get_handlers(
            message_type="UserCreated",
            topic_category=EnumMessageCategory.EVENT,
            topic_domain="user",
        )
        assert len(handlers) == 2
        assert "user-handler" in handlers
        assert "audit-handler" in handlers

    def test_register_after_freeze_fails(self) -> None:
        """Test that registration fails after freeze."""
        registry = MessageTypeRegistry()
        registry.freeze()

        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )

        with pytest.raises(ModelOnexError) as exc_info:
            registry.register_message_type(entry)
        assert "frozen" in str(exc_info.value.message).lower()

    def test_register_none_entry_fails(self) -> None:
        """Test that registering None entry fails."""
        registry = MessageTypeRegistry()
        with pytest.raises(ModelOnexError):
            registry.register_message_type(None)  # type: ignore[arg-type]

    def test_register_conflicting_category_constraints_fails(self) -> None:
        """Test that conflicting category constraints raise error."""
        registry = MessageTypeRegistry()

        # First registration with EVENT only
        entry1 = ModelMessageTypeEntry(
            message_type="UserAction",
            handler_ids=("handler1",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        registry.register_message_type(entry1)

        # Second registration with COMMAND - should fail
        entry2 = ModelMessageTypeEntry(
            message_type="UserAction",
            handler_ids=("handler2",),
            allowed_categories=frozenset([EnumMessageCategory.COMMAND]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )

        with pytest.raises(MessageTypeRegistryError) as exc_info:
            registry.register_message_type(entry2)
        assert "Category constraint mismatch" in str(exc_info.value.message)

    def test_register_conflicting_domain_constraints_fails(self) -> None:
        """Test that conflicting domain constraints raise error."""
        registry = MessageTypeRegistry()

        # First registration with user domain
        entry1 = ModelMessageTypeEntry(
            message_type="SharedEvent",
            handler_ids=("handler1",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        registry.register_message_type(entry1)

        # Second registration with order domain - should fail
        entry2 = ModelMessageTypeEntry(
            message_type="SharedEvent",
            handler_ids=("handler2",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="order"),
        )

        with pytest.raises(MessageTypeRegistryError) as exc_info:
            registry.register_message_type(entry2)
        assert "Domain constraint mismatch" in str(exc_info.value.message)


class TestMessageTypeRegistryFreeze:
    """Tests for freeze functionality."""

    def test_freeze(self) -> None:
        """Test basic freeze functionality."""
        registry = MessageTypeRegistry()
        assert registry.is_frozen is False

        registry.freeze()
        assert registry.is_frozen is True

    def test_freeze_is_idempotent(self) -> None:
        """Test that freeze() can be called multiple times."""
        registry = MessageTypeRegistry()
        registry.freeze()
        registry.freeze()  # Should not raise
        assert registry.is_frozen is True


class TestMessageTypeRegistryQueries:
    """Tests for query functionality."""

    @pytest.fixture
    def populated_registry(self) -> MessageTypeRegistry:
        """Create a registry with test data."""
        registry = MessageTypeRegistry()

        # Register user domain events
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        registry.register_simple(
            message_type="UserUpdated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )

        # Register order domain events
        registry.register_simple(
            message_type="OrderCreated",
            handler_id="order-handler",
            category=EnumMessageCategory.EVENT,
            domain="order",
        )

        # Register command
        registry.register_simple(
            message_type="CreateUserCommand",
            handler_id="user-command-handler",
            category=EnumMessageCategory.COMMAND,
            domain="user",
        )

        registry.freeze()
        return registry

    def test_get_handlers_success(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test successful handler lookup."""
        handlers = populated_registry.get_handlers(
            message_type="UserCreated",
            topic_category=EnumMessageCategory.EVENT,
            topic_domain="user",
        )
        assert handlers == ["user-handler"]

    def test_get_handlers_not_found(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test handler lookup for unknown message type."""
        with pytest.raises(MessageTypeRegistryError) as exc_info:
            populated_registry.get_handlers(
                message_type="UnknownType",
                topic_category=EnumMessageCategory.EVENT,
                topic_domain="user",
            )
        assert "No handler mapping" in str(exc_info.value.message)
        assert "UnknownType" in str(exc_info.value.message)

    def test_get_handlers_category_mismatch(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test handler lookup with wrong category."""
        with pytest.raises(MessageTypeRegistryError) as exc_info:
            populated_registry.get_handlers(
                message_type="UserCreated",  # Registered as EVENT
                topic_category=EnumMessageCategory.COMMAND,  # Wrong category
                topic_domain="user",
            )
        assert "not allowed in category" in str(exc_info.value.message)

    def test_get_handlers_domain_mismatch(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test handler lookup with wrong domain."""
        with pytest.raises(MessageTypeRegistryError) as exc_info:
            populated_registry.get_handlers(
                message_type="UserCreated",  # Registered in user domain
                topic_category=EnumMessageCategory.EVENT,
                topic_domain="order",  # Wrong domain
            )
        assert "Domain mismatch" in str(exc_info.value.message)

    def test_get_handlers_before_freeze_fails(self) -> None:
        """Test that get_handlers fails before freeze."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )

        with pytest.raises(ModelOnexError) as exc_info:
            registry.get_handlers(
                message_type="UserCreated",
                topic_category=EnumMessageCategory.EVENT,
                topic_domain="user",
            )
        assert "freeze()" in str(exc_info.value.message)

    def test_get_handlers_unchecked(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test unchecked handler lookup."""
        handlers = populated_registry.get_handlers_unchecked("UserCreated")
        assert handlers == ["user-handler"]

        # Unknown type returns None
        assert populated_registry.get_handlers_unchecked("Unknown") is None

    def test_get_entry(self, populated_registry: MessageTypeRegistry) -> None:
        """Test getting full entry."""
        entry = populated_registry.get_entry("UserCreated")
        assert entry is not None
        assert entry.message_type == "UserCreated"
        assert entry.handler_ids == ("user-handler",)

    def test_get_entry_not_found(self, populated_registry: MessageTypeRegistry) -> None:
        """Test getting entry that doesn't exist."""
        entry = populated_registry.get_entry("Unknown")
        assert entry is None

    def test_has_message_type(self, populated_registry: MessageTypeRegistry) -> None:
        """Test checking if message type exists."""
        assert populated_registry.has_message_type("UserCreated") is True
        assert populated_registry.has_message_type("Unknown") is False

    def test_list_message_types(self, populated_registry: MessageTypeRegistry) -> None:
        """Test listing all message types."""
        types = populated_registry.list_message_types()
        assert "UserCreated" in types
        assert "UserUpdated" in types
        assert "OrderCreated" in types
        assert "CreateUserCommand" in types
        assert len(types) == 4

    def test_list_message_types_by_category(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test filtering message types by category."""
        events = populated_registry.list_message_types(
            category=EnumMessageCategory.EVENT
        )
        assert "UserCreated" in events
        assert "OrderCreated" in events
        assert "CreateUserCommand" not in events

        commands = populated_registry.list_message_types(
            category=EnumMessageCategory.COMMAND
        )
        assert "CreateUserCommand" in commands
        assert "UserCreated" not in commands

    def test_list_message_types_by_domain(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test filtering message types by domain."""
        user_types = populated_registry.list_message_types(domain="user")
        assert "UserCreated" in user_types
        assert "UserUpdated" in user_types
        assert "CreateUserCommand" in user_types
        assert "OrderCreated" not in user_types

        order_types = populated_registry.list_message_types(domain="order")
        assert "OrderCreated" in order_types
        assert "UserCreated" not in order_types

    def test_list_message_types_by_category_and_domain(
        self, populated_registry: MessageTypeRegistry
    ) -> None:
        """Test filtering message types by both category and domain."""
        user_events = populated_registry.list_message_types(
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        assert "UserCreated" in user_events
        assert "UserUpdated" in user_events
        assert "CreateUserCommand" not in user_events  # COMMAND, not EVENT
        assert "OrderCreated" not in user_events  # order domain

    def test_list_domains(self, populated_registry: MessageTypeRegistry) -> None:
        """Test listing all domains."""
        domains = populated_registry.list_domains()
        assert "user" in domains
        assert "order" in domains
        assert len(domains) == 2

    def test_list_handlers(self, populated_registry: MessageTypeRegistry) -> None:
        """Test listing all handler IDs."""
        handlers = populated_registry.list_handlers()
        assert "user-handler" in handlers
        assert "order-handler" in handlers
        assert "user-command-handler" in handlers


class TestMessageTypeRegistryValidation:
    """Tests for validation functionality."""

    def test_validate_startup_success(self) -> None:
        """Test successful startup validation."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        registry.freeze()

        # Validate with matching handler set
        errors = registry.validate_startup(available_handler_ids={"user-handler"})
        assert errors == []

    def test_validate_startup_missing_handler(self) -> None:
        """Test validation detects missing handlers."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        registry.freeze()

        # Validate with empty handler set
        errors = registry.validate_startup(available_handler_ids=set())
        assert len(errors) == 1
        assert "user-handler" in errors[0]
        assert "not registered with the dispatch engine" in errors[0]

    def test_validate_startup_without_handler_set(self) -> None:
        """Test validation without providing handler set."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        registry.freeze()

        # Should pass without handler set check
        errors = registry.validate_startup()
        assert errors == []

    def test_validate_topic_message_type_success(self) -> None:
        """Test topic-message type validation success."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        registry.freeze()

        is_valid, error = registry.validate_topic_message_type(
            topic="dev.user.events.v1",
            message_type="UserCreated",
        )
        assert is_valid is True
        assert error is None

    def test_validate_topic_message_type_category_mismatch(self) -> None:
        """Test topic-message type validation with category mismatch."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        registry.freeze()

        is_valid, error = registry.validate_topic_message_type(
            topic="dev.user.commands.v1",  # Commands, not events
            message_type="UserCreated",
        )
        assert is_valid is False
        assert error is not None
        assert "not allowed in category" in error

    def test_validate_topic_message_type_domain_mismatch(self) -> None:
        """Test topic-message type validation with domain mismatch."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="UserCreated",
            handler_id="user-handler",
            category=EnumMessageCategory.EVENT,
            domain="user",
        )
        registry.freeze()

        is_valid, error = registry.validate_topic_message_type(
            topic="dev.order.events.v1",  # order domain, not user
            message_type="UserCreated",
        )
        assert is_valid is False
        assert error is not None
        assert "Domain mismatch" in error


class TestMessageTypeRegistryDomainCrossDomain:
    """Tests for cross-domain consumption."""

    def test_cross_domain_allowed(self) -> None:
        """Test cross-domain consumption when explicitly allowed."""
        registry = MessageTypeRegistry()

        # Register notification handler that can consume from user domain
        entry = ModelMessageTypeEntry(
            message_type="UserNotification",
            handler_ids=("notification-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(
                owning_domain="notification",
                allowed_cross_domains=frozenset({"user"}),
            ),
        )
        registry.register_message_type(entry)
        registry.freeze()

        # Should succeed - cross-domain explicitly allowed
        handlers = registry.get_handlers(
            message_type="UserNotification",
            topic_category=EnumMessageCategory.EVENT,
            topic_domain="user",  # Different from owning_domain
        )
        assert handlers == ["notification-handler"]

    def test_cross_domain_blocked_by_default(self) -> None:
        """Test cross-domain consumption blocked by default."""
        registry = MessageTypeRegistry()

        # Register handler without cross-domain permissions
        entry = ModelMessageTypeEntry(
            message_type="UserEvent",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        registry.register_message_type(entry)
        registry.freeze()

        # Should fail - cross-domain not allowed
        with pytest.raises(MessageTypeRegistryError) as exc_info:
            registry.get_handlers(
                message_type="UserEvent",
                topic_category=EnumMessageCategory.EVENT,
                topic_domain="order",  # Different domain
            )
        assert "Domain mismatch" in str(exc_info.value.message)


class TestMessageTypeRegistryProperties:
    """Tests for registry properties."""

    def test_entry_count(self) -> None:
        """Test entry_count property."""
        registry = MessageTypeRegistry()
        assert registry.entry_count == 0

        registry.register_simple(
            message_type="Type1",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )
        assert registry.entry_count == 1

        registry.register_simple(
            message_type="Type2",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )
        assert registry.entry_count == 2

    def test_handler_count(self) -> None:
        """Test handler_count property."""
        registry = MessageTypeRegistry()
        assert registry.handler_count == 0

        registry.register_simple(
            message_type="Type1",
            handler_id="handler1",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )
        assert registry.handler_count == 1

        # Same handler for different type shouldn't increase count
        registry.register_simple(
            message_type="Type2",
            handler_id="handler1",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )
        assert registry.handler_count == 1

        # Different handler should increase count
        registry.register_simple(
            message_type="Type3",
            handler_id="handler2",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )
        assert registry.handler_count == 2

    def test_domain_count(self) -> None:
        """Test domain_count property."""
        registry = MessageTypeRegistry()
        assert registry.domain_count == 0

        registry.register_simple(
            message_type="Type1",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="domain1",
        )
        assert registry.domain_count == 1

        # Same domain shouldn't increase count
        registry.register_simple(
            message_type="Type2",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="domain1",
        )
        assert registry.domain_count == 1

        # Different domain should increase count
        registry.register_simple(
            message_type="Type3",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="domain2",
        )
        assert registry.domain_count == 2


class TestMessageTypeRegistryDunderMethods:
    """Tests for dunder methods."""

    def test_len(self) -> None:
        """Test __len__ method."""
        registry = MessageTypeRegistry()
        assert len(registry) == 0

        registry.register_simple(
            message_type="Type1",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )
        assert len(registry) == 1

    def test_contains(self) -> None:
        """Test __contains__ method."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="Type1",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )

        assert "Type1" in registry
        assert "Unknown" not in registry

    def test_str(self) -> None:
        """Test __str__ method."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="Type1",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )

        result = str(registry)
        assert "MessageTypeRegistry" in result
        assert "entries=1" in result
        assert "domains=1" in result

    def test_repr(self) -> None:
        """Test __repr__ method."""
        registry = MessageTypeRegistry()
        registry.register_simple(
            message_type="Type1",
            handler_id="handler",
            category=EnumMessageCategory.EVENT,
            domain="test",
        )

        result = repr(registry)
        assert "MessageTypeRegistry" in result
        assert "Type1" in result
