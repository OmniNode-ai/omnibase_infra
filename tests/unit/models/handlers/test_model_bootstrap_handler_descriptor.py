# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelBootstrapHandlerDescriptor with required handler_class.

Tests the ModelBootstrapHandlerDescriptor functionality including:
- Required handler_class validation (fails if missing)
- Inheritance from ModelHandlerDescriptor
- Pattern validation for handler_class field
- Conversion to base ModelHandlerDescriptor
- Comparison with base descriptor behavior

Related:
    - OMN-1087: HandlerBootstrapSource descriptor-based validation
    - src/omnibase_infra/models/handlers/model_bootstrap_handler_descriptor.py
    - src/omnibase_infra/models/handlers/model_handler_descriptor.py

Expected Behavior:
    ModelBootstrapHandlerDescriptor extends ModelHandlerDescriptor with one key
    difference: handler_class is REQUIRED (not optional). This ensures bootstrap
    handlers always specify their implementation class since they have no
    contract file to derive the class from.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.handlers import (
    ModelBootstrapHandlerDescriptor,
    ModelHandlerDescriptor,
)

# =============================================================================
# Test Constants
# =============================================================================

# Valid handler_class values for testing
VALID_HANDLER_CLASS = "omnibase_infra.handlers.handler_consul.HandlerConsul"
VALID_HANDLER_CLASS_SHORT = "myapp.handlers.MyHandler"

# Invalid handler_class values for testing
INVALID_HANDLER_CLASS_NO_DOT = "HandlerConsul"
INVALID_HANDLER_CLASS_TOO_SHORT = "ab"  # Below min_length of 3
INVALID_HANDLER_CLASS_STARTS_WITH_NUMBER = "1module.Handler"

# Base descriptor fields (shared between all tests)
BASE_DESCRIPTOR_FIELDS = {
    "handler_id": "bootstrap.test",
    "name": "Test Handler",
    "version": "1.0.0",
    "handler_kind": "effect",
    "input_model": "omnibase_infra.models.types.JsonDict",
    "output_model": "omnibase_core.models.dispatch.ModelHandlerOutput",
}


# =============================================================================
# Required handler_class Validation Tests
# =============================================================================


class TestBootstrapHandlerDescriptorRequiredHandlerClass:
    """Tests for the required handler_class field behavior.

    These tests verify that ModelBootstrapHandlerDescriptor enforces
    handler_class as a required field, unlike the base ModelHandlerDescriptor
    where it is optional.
    """

    def test_handler_class_required_fails_when_missing(self) -> None:
        """Creating descriptor without handler_class should raise ValidationError.

        This is the key difference from ModelHandlerDescriptor where
        handler_class defaults to None.
        """
        with pytest.raises(ValidationError) as exc_info:
            ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                # handler_class intentionally omitted
            )

        # Verify the error mentions handler_class field
        errors = exc_info.value.errors()
        handler_class_errors = [e for e in errors if "handler_class" in str(e["loc"])]
        assert len(handler_class_errors) > 0, (
            "Expected validation error for missing handler_class"
        )

    def test_handler_class_required_fails_when_none(self) -> None:
        """Explicitly passing handler_class=None should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                handler_class=None,  # type: ignore[arg-type]  # intentional
            )

        errors = exc_info.value.errors()
        handler_class_errors = [e for e in errors if "handler_class" in str(e["loc"])]
        assert len(handler_class_errors) > 0

    def test_handler_class_succeeds_when_provided(self) -> None:
        """Creating descriptor with valid handler_class should succeed."""
        descriptor = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
        )

        assert descriptor.handler_class == VALID_HANDLER_CLASS

    def test_base_descriptor_allows_none_handler_class(self) -> None:
        """Base ModelHandlerDescriptor should allow handler_class=None.

        This test confirms the difference in behavior between the two classes.
        """
        # This should succeed - base descriptor has optional handler_class
        descriptor = ModelHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            # handler_class not provided - should default to None
        )

        assert descriptor.handler_class is None

    def test_base_descriptor_allows_explicit_none(self) -> None:
        """Base ModelHandlerDescriptor should allow explicit handler_class=None."""
        descriptor = ModelHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=None,
        )

        assert descriptor.handler_class is None


# =============================================================================
# handler_class Pattern Validation Tests
# =============================================================================


class TestBootstrapHandlerDescriptorPatternValidation:
    """Tests for handler_class pattern validation.

    Both ModelBootstrapHandlerDescriptor and ModelHandlerDescriptor share
    the same pattern constraint: the handler_class must be a valid
    fully qualified Python module path.
    """

    def test_valid_handler_class_patterns(self) -> None:
        """Various valid handler_class patterns should be accepted."""
        valid_patterns = [
            "omnibase_infra.handlers.handler_consul.HandlerConsul",
            "myapp.handlers.MyHandler",
            "a.b.C",
            "module_name.SubModule.ClassName",
            "_private.module.Handler",
        ]

        for pattern in valid_patterns:
            descriptor = ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                handler_class=pattern,
            )
            assert descriptor.handler_class == pattern, f"Failed for: {pattern}"

    def test_handler_class_without_dot_fails(self) -> None:
        """handler_class without a dot should fail validation (not a module path)."""
        with pytest.raises(ValidationError):
            ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                handler_class=INVALID_HANDLER_CLASS_NO_DOT,
            )

    def test_handler_class_too_short_fails(self) -> None:
        """handler_class shorter than min_length (3) should fail validation."""
        with pytest.raises(ValidationError):
            ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                handler_class="ab",  # Only 2 chars, below min_length of 3
            )

    def test_handler_class_starting_with_number_fails(self) -> None:
        """handler_class starting with a number should fail validation."""
        with pytest.raises(ValidationError):
            ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                handler_class=INVALID_HANDLER_CLASS_STARTS_WITH_NUMBER,
            )

    def test_handler_class_with_special_chars_fails(self) -> None:
        """handler_class with invalid characters should fail validation."""
        invalid_patterns = [
            "module.class-name",  # hyphen
            "module.class name",  # space
            "module.class@name",  # @
            "module/class.name",  # /
        ]

        for pattern in invalid_patterns:
            with pytest.raises(ValidationError):
                ModelBootstrapHandlerDescriptor(
                    **BASE_DESCRIPTOR_FIELDS,
                    handler_class=pattern,
                )


# =============================================================================
# Inheritance and Type Tests
# =============================================================================


class TestBootstrapHandlerDescriptorInheritance:
    """Tests for inheritance relationship with ModelHandlerDescriptor.

    ModelBootstrapHandlerDescriptor extends ModelHandlerDescriptor,
    so instances should be compatible with code expecting the base type.
    """

    def test_is_subclass_of_model_handler_descriptor(self) -> None:
        """ModelBootstrapHandlerDescriptor should be a subclass of base."""
        assert issubclass(
            ModelBootstrapHandlerDescriptor,
            ModelHandlerDescriptor,
        )

    def test_instance_is_model_handler_descriptor(self) -> None:
        """Instances should be recognized as ModelHandlerDescriptor."""
        descriptor = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
        )

        assert isinstance(descriptor, ModelHandlerDescriptor)
        assert isinstance(descriptor, ModelBootstrapHandlerDescriptor)

    def test_can_be_used_where_base_expected(self) -> None:
        """Bootstrap descriptor should work where base descriptor is expected.

        This simulates code that accepts ModelHandlerDescriptor and verifies
        bootstrap descriptors are compatible.
        """

        def process_descriptor(desc: ModelHandlerDescriptor) -> str:
            return desc.handler_id

        descriptor = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
        )

        result = process_descriptor(descriptor)
        assert result == "bootstrap.test"


# =============================================================================
# to_base_descriptor() Method Tests
# =============================================================================


class TestBootstrapHandlerDescriptorToBase:
    """Tests for the to_base_descriptor() conversion method."""

    def test_to_base_descriptor_returns_model_handler_descriptor(self) -> None:
        """to_base_descriptor() should return a ModelHandlerDescriptor instance."""
        bootstrap_desc = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
            description="Test description",
        )

        base_desc = bootstrap_desc.to_base_descriptor()

        assert type(base_desc) is ModelHandlerDescriptor
        assert not isinstance(base_desc, ModelBootstrapHandlerDescriptor)

    def test_to_base_descriptor_copies_all_fields(self) -> None:
        """to_base_descriptor() should copy all field values correctly."""
        bootstrap_desc = ModelBootstrapHandlerDescriptor(
            handler_id="bootstrap.consul",
            name="Consul Handler",
            version="2.1.3",
            handler_kind="effect",
            input_model="omnibase_infra.models.types.JsonDict",
            output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
            description="Test handler description",
            handler_class=VALID_HANDLER_CLASS,
            contract_path="/path/to/contract.yaml",
        )

        base_desc = bootstrap_desc.to_base_descriptor()

        assert base_desc.handler_id == bootstrap_desc.handler_id
        assert base_desc.name == bootstrap_desc.name
        assert base_desc.version == bootstrap_desc.version
        assert base_desc.handler_kind == bootstrap_desc.handler_kind
        assert base_desc.input_model == bootstrap_desc.input_model
        assert base_desc.output_model == bootstrap_desc.output_model
        assert base_desc.description == bootstrap_desc.description
        assert base_desc.handler_class == bootstrap_desc.handler_class
        assert base_desc.contract_path == bootstrap_desc.contract_path

    def test_to_base_descriptor_with_none_optional_fields(self) -> None:
        """to_base_descriptor() should handle None optional fields."""
        bootstrap_desc = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
            # description and contract_path not provided (defaults to None)
        )

        base_desc = bootstrap_desc.to_base_descriptor()

        assert base_desc.description is None
        assert base_desc.contract_path is None
        assert base_desc.handler_class == VALID_HANDLER_CLASS


# =============================================================================
# Frozen Model Tests
# =============================================================================


class TestBootstrapHandlerDescriptorFrozen:
    """Tests for frozen (immutable) model behavior."""

    def test_descriptor_is_frozen(self) -> None:
        """ModelBootstrapHandlerDescriptor should be immutable."""
        descriptor = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
        )

        with pytest.raises(ValidationError):
            descriptor.handler_class = "new.module.Handler"  # type: ignore[misc]

    def test_all_fields_are_immutable(self) -> None:
        """All descriptor fields should be immutable."""
        descriptor = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
        )

        immutable_fields = [
            ("handler_id", "new.id"),
            ("name", "New Name"),
            ("handler_kind", "compute"),
            ("input_model", "new.Input"),
            ("output_model", "new.Output"),
            ("handler_class", "new.Handler.Class"),
        ]

        for field_name, new_value in immutable_fields:
            with pytest.raises(ValidationError):
                setattr(descriptor, field_name, new_value)


# =============================================================================
# Version Field Tests
# =============================================================================


class TestBootstrapHandlerDescriptorVersion:
    """Tests for version field handling (inherited from base)."""

    def test_version_string_converted_to_model_semver(self) -> None:
        """String version should be converted to ModelSemVer."""
        descriptor = ModelBootstrapHandlerDescriptor(
            **BASE_DESCRIPTOR_FIELDS,
            handler_class=VALID_HANDLER_CLASS,
        )

        assert isinstance(descriptor.version, ModelSemVer)
        assert descriptor.version.major == 1
        assert descriptor.version.minor == 0
        assert descriptor.version.patch == 0

    def test_version_accepts_model_semver_directly(self) -> None:
        """ModelSemVer instance should be accepted directly."""
        semver = ModelSemVer(major=2, minor=3, patch=4)

        descriptor = ModelBootstrapHandlerDescriptor(
            handler_id="bootstrap.test",
            name="Test Handler",
            version=semver,
            handler_kind="effect",
            input_model="omnibase_infra.models.types.JsonDict",
            output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
            handler_class=VALID_HANDLER_CLASS,
        )

        assert descriptor.version.major == 2
        assert descriptor.version.minor == 3
        assert descriptor.version.patch == 4


# =============================================================================
# Integration with HandlerBootstrapSource Tests
# =============================================================================


class TestBootstrapHandlerDescriptorIntegration:
    """Integration tests with HandlerBootstrapSource.

    These tests verify that the descriptor works correctly when used
    by HandlerBootstrapSource for bootstrap handler validation.
    """

    @pytest.mark.asyncio
    async def test_bootstrap_source_uses_bootstrap_descriptor(self) -> None:
        """HandlerBootstrapSource should create ModelBootstrapHandlerDescriptor instances."""
        from omnibase_infra.runtime.handler_bootstrap_source import (
            HandlerBootstrapSource,
        )

        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        # All descriptors should be ModelBootstrapHandlerDescriptor instances
        for descriptor in result.descriptors:
            assert isinstance(descriptor, ModelBootstrapHandlerDescriptor)
            # And also ModelHandlerDescriptor due to inheritance
            assert isinstance(descriptor, ModelHandlerDescriptor)

    @pytest.mark.asyncio
    async def test_all_bootstrap_handlers_have_handler_class(self) -> None:
        """All bootstrap handlers should have non-None handler_class.

        This verifies the fail-fast behavior: if a bootstrap definition
        is missing handler_class, construction would fail during discovery.
        """
        from omnibase_infra.runtime.handler_bootstrap_source import (
            HandlerBootstrapSource,
        )

        source = HandlerBootstrapSource()
        result = await source.discover_handlers()

        for descriptor in result.descriptors:
            assert descriptor.handler_class is not None, (
                f"Handler {descriptor.handler_id} missing handler_class"
            )
            assert len(descriptor.handler_class) > 0


# =============================================================================
# Error Message Quality Tests
# =============================================================================


class TestBootstrapHandlerDescriptorErrorMessages:
    """Tests for validation error message quality.

    These tests ensure that error messages are clear and actionable
    when validation fails.
    """

    def test_missing_handler_class_error_message_is_clear(self) -> None:
        """Error for missing handler_class should clearly identify the issue."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                # handler_class intentionally omitted
            )

        error_str = str(exc_info.value)
        assert "handler_class" in error_str

    def test_invalid_pattern_error_message_is_clear(self) -> None:
        """Error for invalid pattern should clearly identify the issue."""
        with pytest.raises(ValidationError) as exc_info:
            ModelBootstrapHandlerDescriptor(
                **BASE_DESCRIPTOR_FIELDS,
                handler_class="invalid-no-dot",
            )

        error_str = str(exc_info.value)
        # Should mention pattern or string validation
        assert "handler_class" in error_str


__all__ = [
    "TestBootstrapHandlerDescriptorRequiredHandlerClass",
    "TestBootstrapHandlerDescriptorPatternValidation",
    "TestBootstrapHandlerDescriptorInheritance",
    "TestBootstrapHandlerDescriptorToBase",
    "TestBootstrapHandlerDescriptorFrozen",
    "TestBootstrapHandlerDescriptorVersion",
    "TestBootstrapHandlerDescriptorIntegration",
    "TestBootstrapHandlerDescriptorErrorMessages",
]
