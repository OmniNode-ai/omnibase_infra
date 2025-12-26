# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for ModelFailedComponent.

Tests validate:
- Model instantiation with valid data
- Field validation (min_length constraints)
- Strict mode behavior
- Extra fields forbidden (extra='forbid')
- Immutability (frozen=True)
- Custom __str__ method output
- from_attributes=True behavior

.. versionadded:: 1.0.0
    Initial test coverage for ModelFailedComponent.

Related Tickets:
    - OMN-1007: PR #92 review - Add isolated unit tests for ModelFailedComponent
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_failed_component import ModelFailedComponent


class TestModelFailedComponentConstruction:
    """Tests for ModelFailedComponent valid construction."""

    def test_basic_construction(self) -> None:
        """Test basic construction with required fields."""
        failed = ModelFailedComponent(
            component_name="KafkaEventBus",
            error_message="Connection timeout during shutdown",
        )
        assert failed.component_name == "KafkaEventBus"
        assert failed.error_message == "Connection timeout during shutdown"

    def test_construction_with_simple_values(self) -> None:
        """Test construction with simple single-character values."""
        failed = ModelFailedComponent(
            component_name="X",
            error_message="Y",
        )
        assert failed.component_name == "X"
        assert failed.error_message == "Y"

    def test_construction_with_long_values(self) -> None:
        """Test construction with long string values."""
        long_name = "A" * 1000
        long_message = "B" * 5000
        failed = ModelFailedComponent(
            component_name=long_name,
            error_message=long_message,
        )
        assert failed.component_name == long_name
        assert failed.error_message == long_message

    def test_construction_with_special_characters(self) -> None:
        """Test construction with special characters in values."""
        failed = ModelFailedComponent(
            component_name="Kafka::Event::Bus<T>",
            error_message="Error: Failed! @#$%^&*()_+{}|:<>?",
        )
        assert failed.component_name == "Kafka::Event::Bus<T>"
        assert failed.error_message == "Error: Failed! @#$%^&*()_+{}|:<>?"

    def test_construction_with_unicode(self) -> None:
        """Test construction with unicode characters."""
        failed = ModelFailedComponent(
            component_name="KafkaEventBus\u2605",
            error_message="Error: Failed with \u2764 and \u00e9",
        )
        assert failed.component_name == "KafkaEventBus\u2605"
        assert "\u00e9" in failed.error_message

    def test_construction_with_newlines(self) -> None:
        """Test construction with newlines in error message."""
        failed = ModelFailedComponent(
            component_name="MultiLineComponent",
            error_message="Line 1\nLine 2\nLine 3",
        )
        assert "\n" in failed.error_message
        assert failed.error_message.count("\n") == 2

    @pytest.mark.parametrize(
        ("component_name", "error_message"),
        [
            ("ConsulAdapter", "Service discovery failed"),
            ("VaultHandler", "Secret resolution timeout"),
            ("PostgresPool", "Connection pool exhausted"),
            ("RedisCache", "Cache invalidation error"),
        ],
        ids=[
            "consul_adapter",
            "vault_handler",
            "postgres_pool",
            "redis_cache",
        ],
    )
    def test_construction_with_various_component_types(
        self,
        component_name: str,
        error_message: str,
    ) -> None:
        """Test construction with various component type examples."""
        failed = ModelFailedComponent(
            component_name=component_name,
            error_message=error_message,
        )
        assert failed.component_name == component_name
        assert failed.error_message == error_message


class TestModelFailedComponentValidation:
    """Tests for ModelFailedComponent field validation."""

    def test_component_name_required(self) -> None:
        """Test that component_name is a required field."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                error_message="Some error",  # type: ignore[call-arg]
            )
        assert "component_name" in str(exc_info.value)

    def test_error_message_required(self) -> None:
        """Test that error_message is a required field."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="SomeComponent",  # type: ignore[call-arg]
            )
        assert "error_message" in str(exc_info.value)

    def test_component_name_min_length(self) -> None:
        """Test that component_name must have min_length=1."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="",
                error_message="Some error",
            )
        error_str = str(exc_info.value)
        assert "component_name" in error_str
        # Pydantic v2 uses 'String should have at least 1 character'
        assert "1" in error_str or "min_length" in error_str.lower()

    def test_error_message_min_length(self) -> None:
        """Test that error_message must have min_length=1."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="SomeComponent",
                error_message="",
            )
        error_str = str(exc_info.value)
        assert "error_message" in error_str
        assert "1" in error_str or "min_length" in error_str.lower()

    def test_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden (extra='forbid')."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="SomeComponent",
                error_message="Some error",
                unknown_field="unexpected",  # type: ignore[call-arg]
            )
        error_str = str(exc_info.value).lower()
        assert "unknown_field" in error_str or "extra" in error_str

    def test_strict_mode_rejects_non_string_component_name(self) -> None:
        """Test that strict mode rejects non-string component_name."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name=123,  # type: ignore[arg-type]
                error_message="Some error",
            )
        error_str = str(exc_info.value)
        assert "component_name" in error_str

    def test_strict_mode_rejects_non_string_error_message(self) -> None:
        """Test that strict mode rejects non-string error_message."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="SomeComponent",
                error_message=456,  # type: ignore[arg-type]
            )
        error_str = str(exc_info.value)
        assert "error_message" in error_str

    def test_strict_mode_rejects_bytes(self) -> None:
        """Test that strict mode rejects bytes for string fields."""
        with pytest.raises(ValidationError):
            ModelFailedComponent(
                component_name=b"BytesComponent",  # type: ignore[arg-type]
                error_message="Some error",
            )

    def test_strict_mode_rejects_none_for_component_name(self) -> None:
        """Test that None is rejected for component_name."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name=None,  # type: ignore[arg-type]
                error_message="Some error",
            )
        assert "component_name" in str(exc_info.value)

    def test_strict_mode_rejects_none_for_error_message(self) -> None:
        """Test that None is rejected for error_message."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent(
                component_name="SomeComponent",
                error_message=None,  # type: ignore[arg-type]
            )
        assert "error_message" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_type",
        [
            123,
            45.67,
            True,
            ["list"],
            {"dict": "value"},
            object(),
        ],
        ids=["int", "float", "bool", "list", "dict", "object"],
    )
    def test_strict_mode_rejects_various_types_for_component_name(
        self,
        invalid_type: object,
    ) -> None:
        """Test that strict mode rejects various non-string types."""
        with pytest.raises(ValidationError):
            ModelFailedComponent(
                component_name=invalid_type,  # type: ignore[arg-type]
                error_message="Some error",
            )


class TestModelFailedComponentImmutability:
    """Tests for ModelFailedComponent immutability (frozen=True)."""

    def test_component_name_is_immutable(self) -> None:
        """Test that component_name cannot be modified after creation."""
        failed = ModelFailedComponent(
            component_name="OriginalName",
            error_message="Original error",
        )
        with pytest.raises(ValidationError):
            failed.component_name = "NewName"  # type: ignore[misc]

    def test_error_message_is_immutable(self) -> None:
        """Test that error_message cannot be modified after creation."""
        failed = ModelFailedComponent(
            component_name="SomeComponent",
            error_message="Original error",
        )
        with pytest.raises(ValidationError):
            failed.error_message = "New error"  # type: ignore[misc]

    def test_frozen_model_is_hashable(self) -> None:
        """Test that frozen model is hashable."""
        failed = ModelFailedComponent(
            component_name="HashableComponent",
            error_message="Some error",
        )
        # Should not raise
        hash_value = hash(failed)
        assert isinstance(hash_value, int)

    def test_equal_instances_have_same_hash(self) -> None:
        """Test that equal instances have the same hash."""
        failed1 = ModelFailedComponent(
            component_name="SameComponent",
            error_message="Same error",
        )
        failed2 = ModelFailedComponent(
            component_name="SameComponent",
            error_message="Same error",
        )
        assert hash(failed1) == hash(failed2)

    def test_can_be_used_in_set(self) -> None:
        """Test that frozen model can be used in sets."""
        failed1 = ModelFailedComponent(
            component_name="Component1",
            error_message="Error 1",
        )
        failed2 = ModelFailedComponent(
            component_name="Component1",
            error_message="Error 1",
        )  # Duplicate
        failed3 = ModelFailedComponent(
            component_name="Component2",
            error_message="Error 2",
        )

        failed_set = {failed1, failed2, failed3}
        assert len(failed_set) == 2  # Deduplication

    def test_can_be_used_as_dict_key(self) -> None:
        """Test that frozen model can be used as dictionary key."""
        failed = ModelFailedComponent(
            component_name="DictKeyComponent",
            error_message="Some error",
        )
        cache: dict[ModelFailedComponent, str] = {failed: "cached_value"}
        assert cache[failed] == "cached_value"


class TestModelFailedComponentStrRepr:
    """Tests for ModelFailedComponent __str__ method."""

    def test_str_format(self) -> None:
        """Test that __str__ returns expected format."""
        failed = ModelFailedComponent(
            component_name="KafkaEventBus",
            error_message="Connection timeout during shutdown",
        )
        result = str(failed)
        assert result == "KafkaEventBus: Connection timeout during shutdown"

    def test_str_with_simple_values(self) -> None:
        """Test __str__ with simple values."""
        failed = ModelFailedComponent(
            component_name="A",
            error_message="B",
        )
        assert str(failed) == "A: B"

    def test_str_with_special_characters(self) -> None:
        """Test __str__ with special characters."""
        failed = ModelFailedComponent(
            component_name="Comp<T>",
            error_message="Error: !@#$%",
        )
        assert str(failed) == "Comp<T>: Error: !@#$%"

    def test_str_with_colon_in_component_name(self) -> None:
        """Test __str__ when component_name contains colon."""
        failed = ModelFailedComponent(
            component_name="Namespace::Component",
            error_message="Failed",
        )
        assert str(failed) == "Namespace::Component: Failed"

    def test_str_with_colon_in_error_message(self) -> None:
        """Test __str__ when error_message contains colon."""
        failed = ModelFailedComponent(
            component_name="Component",
            error_message="Error: nested: colons: here",
        )
        assert str(failed) == "Component: Error: nested: colons: here"

    def test_str_preserves_whitespace(self) -> None:
        """Test that __str__ preserves whitespace."""
        failed = ModelFailedComponent(
            component_name="  SpacedComponent  ",
            error_message="  Error with spaces  ",
        )
        assert str(failed) == "  SpacedComponent  :   Error with spaces  "

    def test_str_with_newlines(self) -> None:
        """Test __str__ with newlines in error message."""
        failed = ModelFailedComponent(
            component_name="MultiLineComponent",
            error_message="Line1\nLine2",
        )
        assert str(failed) == "MultiLineComponent: Line1\nLine2"

    @pytest.mark.parametrize(
        ("component_name", "error_message", "expected"),
        [
            ("A", "B", "A: B"),
            ("Kafka", "Timeout", "Kafka: Timeout"),
            ("X::Y", "Z", "X::Y: Z"),
        ],
        ids=["simple", "descriptive", "namespaced"],
    )
    def test_str_parametrized(
        self,
        component_name: str,
        error_message: str,
        expected: str,
    ) -> None:
        """Test __str__ with various input combinations."""
        failed = ModelFailedComponent(
            component_name=component_name,
            error_message=error_message,
        )
        assert str(failed) == expected


class TestModelFailedComponentFromAttributes:
    """Tests for ModelFailedComponent from_attributes=True behavior."""

    def test_from_attributes_with_dataclass(self) -> None:
        """Test creating model from a dataclass with matching attributes."""

        @dataclass
        class FailureData:
            component_name: str
            error_message: str

        data = FailureData(
            component_name="DataclassComponent",
            error_message="Dataclass error",
        )
        failed = ModelFailedComponent.model_validate(data)
        assert failed.component_name == "DataclassComponent"
        assert failed.error_message == "Dataclass error"

    def test_from_attributes_with_simple_object(self) -> None:
        """Test creating model from a simple object with attributes."""

        class SimpleObject:
            def __init__(self) -> None:
                self.component_name = "SimpleComponent"
                self.error_message = "Simple error"

        obj = SimpleObject()
        failed = ModelFailedComponent.model_validate(obj)
        assert failed.component_name == "SimpleComponent"
        assert failed.error_message == "Simple error"

    def test_from_attributes_with_namedtuple_like(self) -> None:
        """Test creating model from an object with named attributes."""

        class NamedTupleLike:
            __slots__ = ("component_name", "error_message")

            def __init__(self, name: str, message: str) -> None:
                self.component_name = name
                self.error_message = message

        obj = NamedTupleLike("SlottedComponent", "Slotted error")
        failed = ModelFailedComponent.model_validate(obj)
        assert failed.component_name == "SlottedComponent"
        assert failed.error_message == "Slotted error"

    def test_from_attributes_preserves_validation(self) -> None:
        """Test that from_attributes still validates min_length."""

        @dataclass
        class InvalidData:
            component_name: str
            error_message: str

        data = InvalidData(component_name="", error_message="Valid error")
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate(data)
        assert "component_name" in str(exc_info.value)


class TestModelFailedComponentSerialization:
    """Tests for ModelFailedComponent serialization."""

    def test_model_dump(self) -> None:
        """Test serialization to dict."""
        failed = ModelFailedComponent(
            component_name="SerializableComponent",
            error_message="Serializable error",
        )
        data = failed.model_dump()
        assert data == {
            "component_name": "SerializableComponent",
            "error_message": "Serializable error",
        }

    def test_model_dump_json(self) -> None:
        """Test JSON serialization."""
        failed = ModelFailedComponent(
            component_name="JsonComponent",
            error_message="Json error",
        )
        json_str = failed.model_dump_json()
        assert '"component_name":"JsonComponent"' in json_str
        assert '"error_message":"Json error"' in json_str

    def test_model_from_dict(self) -> None:
        """Test deserialization from dict."""
        data = {
            "component_name": "DictComponent",
            "error_message": "Dict error",
        }
        failed = ModelFailedComponent.model_validate(data)
        assert failed.component_name == "DictComponent"
        assert failed.error_message == "Dict error"

    def test_roundtrip_serialization(self) -> None:
        """Test roundtrip serialization/deserialization."""
        original = ModelFailedComponent(
            component_name="RoundtripComponent",
            error_message="Roundtrip error",
        )
        data = original.model_dump()
        restored = ModelFailedComponent.model_validate(data)
        assert original == restored


class TestModelFailedComponentEquality:
    """Tests for ModelFailedComponent equality comparison."""

    def test_same_values_are_equal(self) -> None:
        """Test that models with same values are equal."""
        failed1 = ModelFailedComponent(
            component_name="SameComponent",
            error_message="Same error",
        )
        failed2 = ModelFailedComponent(
            component_name="SameComponent",
            error_message="Same error",
        )
        assert failed1 == failed2

    def test_different_component_name_not_equal(self) -> None:
        """Test that different component_name makes models not equal."""
        failed1 = ModelFailedComponent(
            component_name="Component1",
            error_message="Same error",
        )
        failed2 = ModelFailedComponent(
            component_name="Component2",
            error_message="Same error",
        )
        assert failed1 != failed2

    def test_different_error_message_not_equal(self) -> None:
        """Test that different error_message makes models not equal."""
        failed1 = ModelFailedComponent(
            component_name="SameComponent",
            error_message="Error 1",
        )
        failed2 = ModelFailedComponent(
            component_name="SameComponent",
            error_message="Error 2",
        )
        assert failed1 != failed2

    def test_not_equal_to_non_model(self) -> None:
        """Test that model is not equal to non-model objects."""
        failed = ModelFailedComponent(
            component_name="Component",
            error_message="Error",
        )
        assert failed != "Component: Error"
        assert failed != {"component_name": "Component", "error_message": "Error"}
        assert failed != None


class TestModelFailedComponentEdgeCases:
    """Edge case tests for ModelFailedComponent."""

    def test_whitespace_only_component_name_valid(self) -> None:
        """Test that whitespace-only component_name is valid (min_length=1)."""
        failed = ModelFailedComponent(
            component_name=" ",
            error_message="Error",
        )
        assert failed.component_name == " "

    def test_whitespace_only_error_message_valid(self) -> None:
        """Test that whitespace-only error_message is valid (min_length=1)."""
        failed = ModelFailedComponent(
            component_name="Component",
            error_message="\t",
        )
        assert failed.error_message == "\t"

    def test_repr_contains_class_name(self) -> None:
        """Test that repr includes class name."""
        failed = ModelFailedComponent(
            component_name="ReprComponent",
            error_message="Repr error",
        )
        repr_str = repr(failed)
        assert "ModelFailedComponent" in repr_str
        assert "ReprComponent" in repr_str

    def test_copy_creates_equal_instance(self) -> None:
        """Test that model_copy creates an equal instance."""
        original = ModelFailedComponent(
            component_name="CopyComponent",
            error_message="Copy error",
        )
        copied = original.model_copy()
        assert original == copied
        assert original is not copied

    def test_copy_with_update(self) -> None:
        """Test that model_copy with update creates modified instance."""
        original = ModelFailedComponent(
            component_name="OriginalComponent",
            error_message="Original error",
        )
        modified = original.model_copy(update={"component_name": "ModifiedComponent"})
        assert modified.component_name == "ModifiedComponent"
        assert modified.error_message == "Original error"
        assert original.component_name == "OriginalComponent"
