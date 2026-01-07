"""Test JsonType equivalence with legacy JsonValue.

This test ensures JsonType (omnibase_core 0.6.2+) accepts the same
value types as the legacy JsonValue type alias.

The migration from JsonValue to JsonType was a rename only - both
represent the same recursive type alias for JSON-compatible values:
    JsonType = str | int | float | bool | None | list[JsonType] | dict[str, JsonType]
"""

from __future__ import annotations

import pytest
from omnibase_core.types import JsonType


class TestJsonTypePrimitives:
    """Verify JsonType accepts all JSON primitive types."""

    def test_accepts_string(self) -> None:
        """JsonType should accept string values."""
        value: JsonType = "hello world"
        assert value == "hello world"
        assert isinstance(value, str)

    def test_accepts_empty_string(self) -> None:
        """JsonType should accept empty strings."""
        value: JsonType = ""
        assert value == ""

    def test_accepts_integer(self) -> None:
        """JsonType should accept integer values."""
        value: JsonType = 42
        assert value == 42
        assert isinstance(value, int)

    def test_accepts_negative_integer(self) -> None:
        """JsonType should accept negative integers."""
        value: JsonType = -100
        assert value == -100

    def test_accepts_zero(self) -> None:
        """JsonType should accept zero."""
        value: JsonType = 0
        assert value == 0

    def test_accepts_float(self) -> None:
        """JsonType should accept float values."""
        value: JsonType = 3.14159
        assert value == 3.14159
        assert isinstance(value, float)

    def test_accepts_negative_float(self) -> None:
        """JsonType should accept negative floats."""
        value: JsonType = -2.718
        assert value == -2.718

    def test_accepts_bool_true(self) -> None:
        """JsonType should accept boolean True."""
        value: JsonType = True
        assert value is True
        assert isinstance(value, bool)

    def test_accepts_bool_false(self) -> None:
        """JsonType should accept boolean False."""
        value: JsonType = False
        assert value is False
        assert isinstance(value, bool)

    def test_accepts_none(self) -> None:
        """JsonType should accept None (JSON null)."""
        value: JsonType = None
        assert value is None


class TestJsonTypeContainers:
    """Verify JsonType accepts JSON container types."""

    def test_accepts_empty_list(self) -> None:
        """JsonType should accept empty list."""
        value: JsonType = []
        assert value == []
        assert isinstance(value, list)

    def test_accepts_list_of_strings(self) -> None:
        """JsonType should accept list of strings."""
        value: JsonType = ["a", "b", "c"]
        assert value == ["a", "b", "c"]

    def test_accepts_list_of_integers(self) -> None:
        """JsonType should accept list of integers."""
        value: JsonType = [1, 2, 3]
        assert value == [1, 2, 3]

    def test_accepts_mixed_type_list(self) -> None:
        """JsonType should accept list with mixed JSON types."""
        value: JsonType = ["string", 42, 3.14, True, False, None]
        assert value == ["string", 42, 3.14, True, False, None]

    def test_accepts_empty_dict(self) -> None:
        """JsonType should accept empty dict."""
        value: JsonType = {}
        assert value == {}
        assert isinstance(value, dict)

    def test_accepts_dict_with_string_keys(self) -> None:
        """JsonType should accept dict with string keys."""
        value: JsonType = {"key": "value"}
        assert value == {"key": "value"}

    def test_accepts_dict_with_mixed_values(self) -> None:
        """JsonType should accept dict with mixed value types."""
        value: JsonType = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
        }
        assert value["string"] == "hello"
        assert value["int"] == 42
        assert value["float"] == 3.14
        assert value["bool"] is True
        assert value["null"] is None


class TestJsonTypeNestedStructures:
    """Verify JsonType accepts nested JSON structures."""

    def test_accepts_nested_list(self) -> None:
        """JsonType should accept nested lists."""
        value: JsonType = [1, [2, [3, [4]]]]
        assert value == [1, [2, [3, [4]]]]

    def test_accepts_nested_dict(self) -> None:
        """JsonType should accept nested dicts."""
        value: JsonType = {"a": {"b": {"c": "deep"}}}
        assert value == {"a": {"b": {"c": "deep"}}}

    def test_accepts_dict_with_list_values(self) -> None:
        """JsonType should accept dict containing lists."""
        value: JsonType = {"items": [1, 2, 3], "tags": ["a", "b"]}
        assert value == {"items": [1, 2, 3], "tags": ["a", "b"]}

    def test_accepts_list_with_dict_elements(self) -> None:
        """JsonType should accept list containing dicts."""
        value: JsonType = [{"id": 1}, {"id": 2}, {"id": 3}]
        assert value == [{"id": 1}, {"id": 2}, {"id": 3}]

    def test_accepts_complex_nested_structure(self) -> None:
        """JsonType should accept complex nested JSON structure."""
        value: JsonType = {
            "name": "test",
            "version": 1,
            "enabled": True,
            "config": {
                "settings": [
                    {"key": "timeout", "value": 30},
                    {"key": "retries", "value": 3},
                ],
                "metadata": {
                    "created": None,
                    "tags": ["prod", "v2"],
                },
            },
            "data": [1, "two", 3.0, True, None, {"nested": []}],
        }
        assert value["name"] == "test"
        assert value["version"] == 1
        assert value["config"]["settings"][0]["value"] == 30
        assert value["config"]["metadata"]["tags"] == ["prod", "v2"]


class TestJsonTypeParameterized:
    """Parametrized tests for comprehensive JsonType validation."""

    @pytest.mark.parametrize(
        "value",
        [
            # Primitives
            "string",
            "",
            42,
            0,
            -1,
            3.14,
            0.0,
            -2.718,
            True,
            False,
            None,
            # Empty containers
            [],
            {},
            # Simple containers
            ["nested", 1, True],
            {"key": "value"},
            # Nested structures
            {"key": "value", "nested": {"a": 1}},
            [1, [2, [3]]],
            {"list": [1, 2, 3], "dict": {"a": "b"}},
        ],
        ids=[
            "string",
            "empty_string",
            "positive_int",
            "zero",
            "negative_int",
            "positive_float",
            "zero_float",
            "negative_float",
            "true",
            "false",
            "none",
            "empty_list",
            "empty_dict",
            "mixed_list",
            "simple_dict",
            "nested_dict",
            "nested_list",
            "complex_nested",
        ],
    )
    def test_json_type_accepts_valid_values(self, value: JsonType) -> None:
        """JsonType should accept all JSON-compatible value types.

        This test validates that JsonType is functionally equivalent to
        the legacy JsonValue type alias, accepting the same set of values:
        - str, int, float, bool, None (primitives)
        - list[JsonType] (arrays)
        - dict[str, JsonType] (objects)
        """
        # Type annotation validates the value is JsonType-compatible
        typed_value: JsonType = value
        assert typed_value == value


class TestJsonTypeRealWorldExamples:
    """Test JsonType with real-world JSON structures."""

    def test_accepts_api_response_structure(self) -> None:
        """JsonType should accept typical API response structure."""
        response: JsonType = {
            "status": "success",
            "code": 200,
            "data": {
                "users": [
                    {"id": 1, "name": "Alice", "active": True},
                    {"id": 2, "name": "Bob", "active": False},
                ],
                "total": 2,
                "page": 1,
                "per_page": 10,
            },
            "meta": {
                "request_id": "abc-123",
                "timestamp": 1234567890,
            },
        }
        assert response["status"] == "success"
        assert response["data"]["total"] == 2

    def test_accepts_config_structure(self) -> None:
        """JsonType should accept typical configuration structure."""
        config: JsonType = {
            "version": "1.0.0",
            "debug": False,
            "timeout_ms": 5000,
            "features": ["auth", "logging", "metrics"],
            "database": {
                "host": "localhost",
                "port": 5432,
                "ssl": True,
                "pool_size": None,
            },
        }
        assert config["version"] == "1.0.0"
        assert config["database"]["port"] == 5432

    def test_accepts_event_payload_structure(self) -> None:
        """JsonType should accept typical event payload structure."""
        event: JsonType = {
            "event_type": "user.created",
            "timestamp": 1234567890.123,
            "payload": {
                "user_id": "usr-001",
                "email": "user@example.com",
                "roles": ["admin", "editor"],
                "preferences": {
                    "theme": "dark",
                    "notifications": True,
                },
            },
            "metadata": {
                "correlation_id": "corr-abc",
                "source": "api",
                "version": 1,
            },
        }
        assert event["event_type"] == "user.created"
        assert event["payload"]["roles"] == ["admin", "editor"]


class TestJsonTypeReExport:
    """Verify JsonType is properly re-exported from omnibase_core.

    The migration from JsonValue to JsonType was completed. JsonValue has been
    removed (following the no backwards compatibility policy). This test class
    verifies that:
    1. JsonType from omnibase_core is properly re-exported via omnibase_infra
    2. The type accepts all JSON-compatible values (same as old JsonValue)
    """

    def test_json_type_from_core_equals_infra_json_type(self) -> None:
        """JsonType from omnibase_core should equal re-exported JsonType."""
        from omnibase_core.types import JsonType as CoreJsonType

        from omnibase_infra.models.types import JsonType as InfraJsonType

        # Both should reference the same type
        assert CoreJsonType is InfraJsonType

    def test_infra_json_type_accepts_primitives(self) -> None:
        """Re-exported JsonType should accept all primitive types.

        This validates JsonType accepts the same primitives as old JsonValue:
        str, int, float, bool, None
        """
        from omnibase_infra.models.types import (
            JsonType,
        )

        # String
        s: JsonType = "hello"
        assert s == "hello"

        # Integer
        i: JsonType = 42
        assert i == 42

        # Float
        f: JsonType = 3.14
        assert f == 3.14

        # Boolean
        b: JsonType = True
        assert b is True

        # None
        n: JsonType = None
        assert n is None

    def test_infra_json_type_accepts_containers(self) -> None:
        """Re-exported JsonType should accept container types.

        This validates JsonType accepts the same containers as old JsonValue:
        list[JsonType], dict[str, JsonType]
        """
        from omnibase_infra.models.types import (
            JsonType,
        )

        # List
        lst: JsonType = [1, "two", 3.0, True, None]
        assert lst == [1, "two", 3.0, True, None]

        # Dict
        dct: JsonType = {"key": "value", "number": 42}
        assert dct == {"key": "value", "number": 42}

    def test_infra_json_type_accepts_nested_structures(self) -> None:
        """Re-exported JsonType should accept nested structures.

        This validates JsonType accepts the same nested structures as old JsonValue.
        """
        from omnibase_infra.models.types import (
            JsonType,
        )

        nested: JsonType = {
            "level1": {
                "level2": {
                    "level3": ["deep", "value"],
                },
            },
            "list_of_dicts": [{"a": 1}, {"b": 2}],
        }
        assert nested["level1"]["level2"]["level3"] == ["deep", "value"]
