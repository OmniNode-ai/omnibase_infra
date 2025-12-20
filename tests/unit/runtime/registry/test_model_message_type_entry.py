# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ModelMessageTypeEntry."""

from datetime import UTC, datetime

import pytest

from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.registry.model_domain_constraint import (
    ModelDomainConstraint,
)
from omnibase_infra.runtime.registry.model_message_type_entry import (
    ModelMessageTypeEntry,
)


class TestModelMessageTypeEntry:
    """Tests for ModelMessageTypeEntry."""

    def test_create_basic_entry(self) -> None:
        """Test creating a basic entry with required fields."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        assert entry.message_type == "UserCreated"
        assert entry.handler_ids == ("user-handler",)
        assert entry.allowed_categories == frozenset([EnumMessageCategory.EVENT])
        assert entry.domain_constraint.owning_domain == "user"
        assert entry.enabled is True  # Default

    def test_create_entry_with_multiple_handlers(self) -> None:
        """Test creating entry with multiple handlers (fan-out)."""
        entry = ModelMessageTypeEntry(
            message_type="OrderCreated",
            handler_ids=("order-handler", "audit-handler", "notification-handler"),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="order"),
        )
        assert len(entry.handler_ids) == 3
        assert "order-handler" in entry.handler_ids
        assert "audit-handler" in entry.handler_ids
        assert "notification-handler" in entry.handler_ids

    def test_create_entry_with_multiple_categories(self) -> None:
        """Test creating entry allowing multiple categories."""
        entry = ModelMessageTypeEntry(
            message_type="UserAction",
            handler_ids=("action-handler",),
            allowed_categories=frozenset(
                [
                    EnumMessageCategory.EVENT,
                    EnumMessageCategory.COMMAND,
                ]
            ),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        assert EnumMessageCategory.EVENT in entry.allowed_categories
        assert EnumMessageCategory.COMMAND in entry.allowed_categories
        assert EnumMessageCategory.INTENT not in entry.allowed_categories

    def test_supports_category(self) -> None:
        """Test supports_category method."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        assert entry.supports_category(EnumMessageCategory.EVENT) is True
        assert entry.supports_category(EnumMessageCategory.COMMAND) is False
        assert entry.supports_category(EnumMessageCategory.INTENT) is False

    def test_validate_category_success(self) -> None:
        """Test validate_category success case."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        is_valid, error = entry.validate_category(EnumMessageCategory.EVENT)
        assert is_valid is True
        assert error is None

    def test_validate_category_failure(self) -> None:
        """Test validate_category failure case."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        is_valid, error = entry.validate_category(EnumMessageCategory.COMMAND)
        assert is_valid is False
        assert error is not None
        assert "UserCreated" in error
        assert "command" in error
        assert "event" in error

    def test_with_additional_handler(self) -> None:
        """Test adding additional handler to entry."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        updated = entry.with_additional_handler("audit-handler")

        # Original should be unchanged (immutable)
        assert entry.handler_ids == ("user-handler",)
        # Updated should have both handlers
        assert updated.handler_ids == ("user-handler", "audit-handler")

    def test_with_additional_handler_duplicate(self) -> None:
        """Test adding duplicate handler returns same entry."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        updated = entry.with_additional_handler("user-handler")

        # Should return same entry (no duplicate added)
        assert updated is entry
        assert updated.handler_ids == ("user-handler",)

    def test_with_enabled(self) -> None:
        """Test updating enabled status."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
            enabled=True,
        )
        disabled = entry.with_enabled(False)

        # Original unchanged
        assert entry.enabled is True
        # Updated should be disabled
        assert disabled.enabled is False

    def test_immutable(self) -> None:
        """Test that entry is immutable."""
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        with pytest.raises(Exception):
            entry.message_type = "Modified"  # type: ignore[misc]

    def test_registered_at_default(self) -> None:
        """Test that registered_at defaults to current time."""
        before = datetime.now(UTC)
        entry = ModelMessageTypeEntry(
            message_type="UserCreated",
            handler_ids=("user-handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        after = datetime.now(UTC)
        assert before <= entry.registered_at <= after


class TestModelMessageTypeEntryValidation:
    """Tests for Pydantic validation."""

    def test_message_type_required(self) -> None:
        """Test that message_type is required."""
        with pytest.raises(Exception):
            ModelMessageTypeEntry(
                handler_ids=("handler",),
                allowed_categories=frozenset([EnumMessageCategory.EVENT]),
                domain_constraint=ModelDomainConstraint(owning_domain="user"),
            )  # type: ignore[call-arg]

    def test_handler_ids_required(self) -> None:
        """Test that handler_ids is required."""
        with pytest.raises(Exception):
            ModelMessageTypeEntry(
                message_type="UserCreated",
                allowed_categories=frozenset([EnumMessageCategory.EVENT]),
                domain_constraint=ModelDomainConstraint(owning_domain="user"),
            )  # type: ignore[call-arg]

    def test_handler_ids_min_length(self) -> None:
        """Test that handler_ids requires at least one handler."""
        with pytest.raises(Exception):
            ModelMessageTypeEntry(
                message_type="UserCreated",
                handler_ids=(),  # Empty tuple
                allowed_categories=frozenset([EnumMessageCategory.EVENT]),
                domain_constraint=ModelDomainConstraint(owning_domain="user"),
            )

    def test_allowed_categories_required(self) -> None:
        """Test that allowed_categories is required."""
        with pytest.raises(Exception):
            ModelMessageTypeEntry(
                message_type="UserCreated",
                handler_ids=("handler",),
                domain_constraint=ModelDomainConstraint(owning_domain="user"),
            )  # type: ignore[call-arg]

    def test_domain_constraint_required(self) -> None:
        """Test that domain_constraint is required."""
        with pytest.raises(Exception):
            ModelMessageTypeEntry(
                message_type="UserCreated",
                handler_ids=("handler",),
                allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            )  # type: ignore[call-arg]

    def test_message_type_max_length(self) -> None:
        """Test message_type max length validation."""
        # Should succeed with valid length
        entry = ModelMessageTypeEntry(
            message_type="A" * 200,
            handler_ids=("handler",),
            allowed_categories=frozenset([EnumMessageCategory.EVENT]),
            domain_constraint=ModelDomainConstraint(owning_domain="user"),
        )
        assert len(entry.message_type) == 200

        # Should fail with too long
        with pytest.raises(Exception):
            ModelMessageTypeEntry(
                message_type="A" * 201,
                handler_ids=("handler",),
                allowed_categories=frozenset([EnumMessageCategory.EVENT]),
                domain_constraint=ModelDomainConstraint(owning_domain="user"),
            )

    def test_allowed_categories_empty_raises_validation_error(self) -> None:
        """Test that empty allowed_categories raises ValidationError.

        A message type with no allowed categories is invalid because it can
        never be routed - there is no valid topic category where it could appear.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            ModelMessageTypeEntry(
                message_type="InvalidType",
                handler_ids=("handler",),
                allowed_categories=frozenset(),  # Empty frozenset - INVALID
                domain_constraint=ModelDomainConstraint(owning_domain="user"),
            )

        # Verify the error message is clear and actionable
        error_str = str(exc_info.value)
        assert "allowed_categories" in error_str
        assert "empty" in error_str.lower() or "cannot be empty" in error_str.lower()
