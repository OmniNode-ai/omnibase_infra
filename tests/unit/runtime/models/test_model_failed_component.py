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
            ("HandlerVault", "Secret resolution timeout"),
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
        assert failed is not None


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


class TestModelFailedComponentJsonDeserialization:
    """Tests for ModelFailedComponent JSON deserialization using model_validate_json.

    These tests validate the model's behavior when parsing JSON strings directly,
    covering valid cases, malformed JSON, type mismatches, and constraint violations.

    Related Tickets:
        - PR #102 review: Add JSON deserialization tests using model_validate_json
    """

    def test_model_validate_json_valid(self) -> None:
        """Test JSON deserialization with valid JSON containing all fields."""
        json_str = (
            '{"component_name": "KafkaEventBus", "error_message": "Connection timeout"}'
        )
        failed = ModelFailedComponent.model_validate_json(json_str)
        assert failed.component_name == "KafkaEventBus"
        assert failed.error_message == "Connection timeout"

    def test_model_validate_json_roundtrip(self) -> None:
        """Test JSON roundtrip serialization/deserialization."""
        original = ModelFailedComponent(
            component_name="RoundtripComponent",
            error_message="Roundtrip error message",
        )
        json_str = original.model_dump_json()
        restored = ModelFailedComponent.model_validate_json(json_str)
        assert original == restored
        assert restored.component_name == "RoundtripComponent"
        assert restored.error_message == "Roundtrip error message"

    def test_model_validate_json_with_unicode(self) -> None:
        """Test JSON deserialization with unicode characters."""
        json_str = '{"component_name": "Component\\u2605", "error_message": "Error with \\u00e9"}'
        failed = ModelFailedComponent.model_validate_json(json_str)
        assert failed.component_name == "Component\u2605"
        assert "\u00e9" in failed.error_message

    def test_model_validate_json_with_escaped_characters(self) -> None:
        """Test JSON deserialization with escaped special characters."""
        json_str = '{"component_name": "Comp\\"Name\\"", "error_message": "Error with\\nNewline"}'
        failed = ModelFailedComponent.model_validate_json(json_str)
        assert '"' in failed.component_name
        assert "\n" in failed.error_message

    def test_model_validate_json_malformed_json(self) -> None:
        """Test that malformed JSON raises ValidationError."""
        malformed_json = '{"component_name": "Test", "error_message": }'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(malformed_json)
        # Pydantic wraps JSON parse errors in ValidationError
        assert exc_info.value.error_count() > 0

    def test_model_validate_json_incomplete_json(self) -> None:
        """Test that incomplete JSON raises ValidationError."""
        incomplete_json = '{"component_name": "Test"'
        with pytest.raises(ValidationError):
            ModelFailedComponent.model_validate_json(incomplete_json)

    def test_model_validate_json_empty_string(self) -> None:
        """Test that empty string raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelFailedComponent.model_validate_json("")

    def test_model_validate_json_not_object(self) -> None:
        """Test that non-object JSON raises ValidationError."""
        with pytest.raises(ValidationError):
            ModelFailedComponent.model_validate_json("[]")
        with pytest.raises(ValidationError):
            ModelFailedComponent.model_validate_json('"just a string"')
        with pytest.raises(ValidationError):
            ModelFailedComponent.model_validate_json("123")

    def test_model_validate_json_wrong_type_component_name_int(self) -> None:
        """Test that integer component_name raises ValidationError (strict mode)."""
        json_str = '{"component_name": 123, "error_message": "Error"}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        error_str = str(exc_info.value)
        assert "component_name" in error_str

    def test_model_validate_json_wrong_type_component_name_null(self) -> None:
        """Test that null component_name raises ValidationError."""
        json_str = '{"component_name": null, "error_message": "Error"}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        assert "component_name" in str(exc_info.value)

    def test_model_validate_json_wrong_type_error_message_bool(self) -> None:
        """Test that boolean error_message raises ValidationError (strict mode)."""
        json_str = '{"component_name": "Test", "error_message": true}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        error_str = str(exc_info.value)
        assert "error_message" in error_str

    def test_model_validate_json_wrong_type_error_message_array(self) -> None:
        """Test that array error_message raises ValidationError."""
        json_str = '{"component_name": "Test", "error_message": ["error1", "error2"]}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        assert "error_message" in str(exc_info.value)

    def test_model_validate_json_wrong_type_error_message_object(self) -> None:
        """Test that object error_message raises ValidationError."""
        json_str = '{"component_name": "Test", "error_message": {"detail": "error"}}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        assert "error_message" in str(exc_info.value)

    def test_model_validate_json_extra_fields_forbidden(self) -> None:
        """Test that extra fields in JSON raise ValidationError (extra='forbid')."""
        json_str = '{"component_name": "Test", "error_message": "Error", "extra_field": "value"}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        error_str = str(exc_info.value).lower()
        assert "extra_field" in error_str or "extra" in error_str

    def test_model_validate_json_multiple_extra_fields(self) -> None:
        """Test that multiple extra fields raise ValidationError."""
        json_str = (
            '{"component_name": "Test", "error_message": "Error", '
            '"field1": "a", "field2": "b"}'
        )
        with pytest.raises(ValidationError):
            ModelFailedComponent.model_validate_json(json_str)

    def test_model_validate_json_missing_component_name(self) -> None:
        """Test that missing component_name raises ValidationError."""
        json_str = '{"error_message": "Error"}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        assert "component_name" in str(exc_info.value)

    def test_model_validate_json_missing_error_message(self) -> None:
        """Test that missing error_message raises ValidationError."""
        json_str = '{"component_name": "Test"}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        assert "error_message" in str(exc_info.value)

    def test_model_validate_json_empty_object(self) -> None:
        """Test that empty JSON object raises ValidationError."""
        json_str = "{}"
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        # Should complain about missing fields
        error_str = str(exc_info.value)
        assert "component_name" in error_str or "error_message" in error_str

    def test_model_validate_json_empty_component_name(self) -> None:
        """Test that empty string component_name violates min_length constraint."""
        json_str = '{"component_name": "", "error_message": "Error"}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        error_str = str(exc_info.value)
        assert "component_name" in error_str

    def test_model_validate_json_empty_error_message(self) -> None:
        """Test that empty string error_message violates min_length constraint."""
        json_str = '{"component_name": "Test", "error_message": ""}'
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(json_str)
        error_str = str(exc_info.value)
        assert "error_message" in error_str

    def test_model_validate_json_whitespace_valid(self) -> None:
        """Test that whitespace-only strings satisfy min_length=1."""
        json_str = '{"component_name": " ", "error_message": "\\t"}'
        failed = ModelFailedComponent.model_validate_json(json_str)
        assert failed.component_name == " "
        assert failed.error_message == "\t"

    @pytest.mark.parametrize(
        ("json_str", "expected_name", "expected_msg"),
        [
            (
                '{"component_name": "A", "error_message": "B"}',
                "A",
                "B",
            ),
            (
                '{"component_name": "Kafka::Bus", "error_message": "Timeout: 30s"}',
                "Kafka::Bus",
                "Timeout: 30s",
            ),
            (
                '{"component_name": "Component<T>", "error_message": "Failed!"}',
                "Component<T>",
                "Failed!",
            ),
        ],
        ids=["simple", "with_colons", "with_special_chars"],
    )
    def test_model_validate_json_parametrized(
        self,
        json_str: str,
        expected_name: str,
        expected_msg: str,
    ) -> None:
        """Test JSON deserialization with various valid inputs."""
        failed = ModelFailedComponent.model_validate_json(json_str)
        assert failed.component_name == expected_name
        assert failed.error_message == expected_msg

    @pytest.mark.parametrize(
        "invalid_json",
        [
            '{"component_name": 1, "error_message": "Error"}',  # int
            '{"component_name": 1.5, "error_message": "Error"}',  # float
            '{"component_name": true, "error_message": "Error"}',  # bool
            '{"component_name": [], "error_message": "Error"}',  # array
            '{"component_name": {}, "error_message": "Error"}',  # object
        ],
        ids=["int", "float", "bool", "array", "object"],
    )
    def test_model_validate_json_strict_rejects_non_string_types(
        self,
        invalid_json: str,
    ) -> None:
        """Test that strict mode rejects non-string types via JSON."""
        with pytest.raises(ValidationError) as exc_info:
            ModelFailedComponent.model_validate_json(invalid_json)
        assert "component_name" in str(exc_info.value)
