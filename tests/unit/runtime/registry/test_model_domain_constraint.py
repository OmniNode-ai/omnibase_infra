# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Tests for ModelDomainConstraint."""

import pytest
from pydantic import ValidationError

from omnibase_infra.models.registry.model_domain_constraint import (
    ModelDomainConstraint,
)


class TestModelDomainConstraint:
    """Tests for ModelDomainConstraint."""

    def test_can_consume_from_own_domain(self) -> None:
        """Test that handler can always consume from own domain."""
        constraint = ModelDomainConstraint(owning_domain="registration")
        assert constraint.can_consume_from("registration") is True

    def test_cannot_consume_from_other_domain_by_default(self) -> None:
        """Test that handler cannot consume from other domains by default."""
        constraint = ModelDomainConstraint(owning_domain="registration")
        assert constraint.can_consume_from("user") is False
        assert constraint.can_consume_from("order") is False

    def test_can_consume_from_allowed_cross_domains(self) -> None:
        """Test that handler can consume from explicitly allowed domains."""
        constraint = ModelDomainConstraint(
            owning_domain="notification",
            allowed_cross_domains=frozenset({"user", "order"}),
        )
        assert constraint.can_consume_from("notification") is True
        assert constraint.can_consume_from("user") is True
        assert constraint.can_consume_from("order") is True
        assert constraint.can_consume_from("billing") is False

    def test_allow_all_domains(self) -> None:
        """Test that allow_all_domains permits any domain."""
        constraint = ModelDomainConstraint(
            owning_domain="audit",
            allow_all_domains=True,
        )
        assert constraint.can_consume_from("audit") is True
        assert constraint.can_consume_from("user") is True
        assert constraint.can_consume_from("order") is True
        assert constraint.can_consume_from("any_domain") is True

    def test_allow_all_domains_overrides_allowed_cross_domains(self) -> None:
        """Test that allow_all_domains overrides the allowlist."""
        constraint = ModelDomainConstraint(
            owning_domain="audit",
            allowed_cross_domains=frozenset({"user"}),  # Normally would block "order"
            allow_all_domains=True,
        )
        # allow_all_domains=True means all domains are allowed
        assert constraint.can_consume_from("order") is True

    def test_validate_consumption_success(self) -> None:
        """Test successful consumption validation."""
        constraint = ModelDomainConstraint(owning_domain="user")
        outcome = constraint.validate_consumption("user", "UserCreated")
        assert outcome.is_valid is True
        assert not outcome.has_error

    def test_validate_consumption_failure_with_explicit_opt_in(self) -> None:
        """Test consumption validation failure with explicit opt-in required."""
        constraint = ModelDomainConstraint(
            owning_domain="registration",
            require_explicit_opt_in=True,
        )
        outcome = constraint.validate_consumption("user", "UserCreated")
        assert outcome.is_valid is False
        assert outcome.has_error
        assert "Domain mismatch" in outcome.error_message
        assert "registration" in outcome.error_message
        assert "user" in outcome.error_message
        assert "UserCreated" in outcome.error_message
        assert "allowed_cross_domains" in outcome.error_message

    def test_validate_consumption_failure_without_explicit_opt_in(self) -> None:
        """Test consumption validation failure without explicit opt-in requirement."""
        constraint = ModelDomainConstraint(
            owning_domain="registration",
            require_explicit_opt_in=False,
        )
        outcome = constraint.validate_consumption("user", "UserCreated")
        assert outcome.is_valid is False
        assert outcome.has_error
        assert "Domain mismatch" in outcome.error_message
        # Should not mention opt-in instructions when require_explicit_opt_in=False
        assert "allowed_cross_domains" not in outcome.error_message

    def test_validate_consumption_with_cross_domain_allowed(self) -> None:
        """Test consumption validation with cross-domain allowed."""
        constraint = ModelDomainConstraint(
            owning_domain="notification",
            allowed_cross_domains=frozenset({"user"}),
        )
        outcome = constraint.validate_consumption("user", "UserCreated")
        assert outcome.is_valid is True
        assert not outcome.has_error

    def test_immutable(self) -> None:
        """Test that ModelDomainConstraint is immutable."""
        constraint = ModelDomainConstraint(owning_domain="user")
        with pytest.raises(ValidationError):
            constraint.owning_domain = "order"  # type: ignore[misc]

    def test_owning_domain_required(self) -> None:
        """Test that owning_domain is required."""
        with pytest.raises(ValidationError):
            ModelDomainConstraint()  # type: ignore[call-arg]

    def test_owning_domain_min_length(self) -> None:
        """Test owning_domain minimum length validation."""
        with pytest.raises(ValidationError):
            ModelDomainConstraint(owning_domain="")


class TestModelDomainConstraintDefaults:
    """Tests for default values."""

    def test_defaults(self) -> None:
        """Test default values are correct."""
        constraint = ModelDomainConstraint(owning_domain="test")
        assert constraint.allowed_cross_domains == frozenset()
        assert constraint.allow_all_domains is False
        assert constraint.require_explicit_opt_in is True
