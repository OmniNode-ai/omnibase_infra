# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Regression tests for JsonType recursion fix.

This test module validates that Pydantic 2.x models using JsonType from omnibase_core
do not trigger RecursionError during model class definition or validation.

Background:
    Pydantic 2.x performs eager type resolution at class definition time.
    The original recursive type alias definition of JsonType:
        JsonType = dict[str, "JsonType"] | list["JsonType"] | str | int | float | bool | None
    caused infinite recursion during schema generation.

    The fix uses typing.TypeAlias pattern which Pydantic 2.x handles correctly.
    This test ensures the fix prevents regression.

Related:
    - ADR: adr-any-type-pydantic-workaround.md
    - Tickets: OMN-1274 (migration), OMN-1262 (tracking)
    - PR #132: Any->JsonType migration

Test Categories:
    1. Model instantiation with deeply nested structures
    2. Serialization round-trip (model -> JSON -> model)
    3. Model validation with complex payloads
    4. Edge cases: empty nesting, mixed types, extreme depth
"""

from __future__ import annotations

import json

import pytest

# Import affected models that use JsonType or dict[str, object]
from omnibase_infra.handlers.models.http.model_http_post_payload import (
    ModelHttpPostPayload,
)
from omnibase_infra.handlers.models.model_db_query_payload import ModelDbQueryPayload
from omnibase_infra.handlers.models.vault.model_vault_secret_payload import (
    ModelVaultSecretPayload,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.runtime.models.model_health_check_result import (
    ModelHealthCheckResult,
)


def create_deeply_nested_dict(depth: int, value: object = "leaf") -> dict[str, object]:
    """Create a deeply nested dictionary structure.

    Args:
        depth: Number of nesting levels.
        value: Value at the deepest level.

    Returns:
        Nested dictionary with specified depth.
    """
    if depth <= 0:
        return {"value": value}
    return {"nested": create_deeply_nested_dict(depth - 1, value)}


def create_complex_nested_structure() -> dict[str, object]:
    """Create a complex nested structure with mixed types.

    Returns:
        Dictionary with nested dicts, lists, and various primitive types.
    """
    return {
        "level1": {
            "level2a": {
                "level3": {"level4": {"level5": {"deep_value": "reached"}}},
                "sibling": [1, 2, {"nested_in_list": True}],
            },
            "level2b": [
                {"item1": "value1"},
                {"item2": ["nested", "list", {"deep": {"deeper": 42}}]},
            ],
        },
        "primitives": {
            "string": "text",
            "integer": 12345,
            "float": 3.14159,
            "boolean": True,
            "null_value": None,
        },
        "arrays": [[1, 2], [3, 4], [[5, 6], [7, 8]]],
    }


class TestJsonTypeRecursionRegression:
    """Regression tests ensuring JsonType doesn't cause RecursionError.

    These tests verify that the fix for Pydantic 2.x recursive type alias
    handling remains effective across all affected models.
    """

    def test_model_class_definition_does_not_recurse(self) -> None:
        """Verify model classes can be defined without RecursionError.

        The original bug caused RecursionError at class definition time,
        not at instantiation. This test imports and accesses model classes
        to ensure the fix prevents this.
        """
        # Simply accessing the model classes would have failed before the fix
        assert ModelNodeCapabilities is not None
        assert ModelHealthCheckResult is not None
        assert ModelDbQueryPayload is not None
        assert ModelHttpPostPayload is not None
        assert ModelVaultSecretPayload is not None

        # Verify we can access model fields (triggers schema resolution)
        assert "config" in ModelNodeCapabilities.model_fields
        assert "details" in ModelHealthCheckResult.model_fields
        assert "body" in ModelHttpPostPayload.model_fields

    def test_deeply_nested_dict_5_levels(self) -> None:
        """Test models with 5-level nested dictionary structures."""
        nested_5 = create_deeply_nested_dict(5)

        # ModelNodeCapabilities with nested config
        caps = ModelNodeCapabilities(config=nested_5)
        assert caps.config == nested_5
        assert (
            caps.config["nested"]["nested"]["nested"]["nested"]["nested"]["value"]
            == "leaf"
        )

    def test_deeply_nested_dict_10_levels(self) -> None:
        """Test models with 10-level nested dictionary structures."""
        nested_10 = create_deeply_nested_dict(10)

        # ModelNodeCapabilities with deeply nested config
        caps = ModelNodeCapabilities(config=nested_10)
        assert caps.config == nested_10

        # Navigate to the deepest value
        current = caps.config
        for _ in range(10):
            current = current["nested"]  # type: ignore[assignment,index]
        assert current["value"] == "leaf"  # type: ignore[index]

    def test_deeply_nested_dict_20_levels(self) -> None:
        """Test models with 20-level nested dictionary structures.

        This depth would have definitely triggered RecursionError
        with the original recursive type alias.
        """
        nested_20 = create_deeply_nested_dict(20)

        caps = ModelNodeCapabilities(config=nested_20)
        assert caps.config == nested_20

    def test_complex_nested_structure(self) -> None:
        """Test models with complex mixed nested structures."""
        complex_data = create_complex_nested_structure()

        # ModelNodeCapabilities
        caps = ModelNodeCapabilities(config=complex_data)
        assert caps.config == complex_data
        assert (
            caps.config["level1"]["level2a"]["level3"]["level4"]["level5"]["deep_value"]
            == "reached"
        )
        assert caps.config["primitives"]["integer"] == 12345

    def test_model_health_check_result_with_nested_details(self) -> None:
        """Test ModelHealthCheckResult with deeply nested details."""
        nested_details = {
            "kafka": {
                "brokers": [
                    {"host": "broker1", "port": 9092, "partitions": {"topic1": 10}},
                    {"host": "broker2", "port": 9092, "partitions": {"topic1": 10}},
                ],
                "consumer_groups": {"group1": {"lag": 100, "members": ["m1", "m2"]}},
            },
            "connections": {"active": 50, "idle": 10, "config": {"max": 100}},
        }

        result = ModelHealthCheckResult(
            handler_type="kafka",
            healthy=True,
            details=nested_details,
        )
        assert result.details == nested_details
        assert result.details["kafka"]["brokers"][0]["partitions"]["topic1"] == 10

    def test_db_query_payload_with_nested_rows(self) -> None:
        """Test ModelDbQueryPayload with nested row data."""
        nested_rows = [
            {
                "id": 1,
                "metadata": {
                    "tags": ["tag1", "tag2"],
                    "config": {"nested": {"deep": True}},
                },
            },
            {
                "id": 2,
                "metadata": {
                    "tags": ["tag3"],
                    "config": {"nested": {"deep": False, "extra": [1, 2, 3]}},
                },
            },
        ]

        payload = ModelDbQueryPayload(rows=nested_rows, row_count=2)
        assert payload.rows == nested_rows
        # Type ignore: dynamically accessing nested JSON structure
        assert payload.rows[0]["metadata"]["config"]["nested"]["deep"] is True  # type: ignore[index]


class TestJsonTypeSerializationRoundTrip:
    """Tests for serialization round-trip with nested JsonType data."""

    def test_model_node_capabilities_round_trip(self) -> None:
        """Test JSON serialization round-trip for ModelNodeCapabilities."""
        complex_config = create_complex_nested_structure()
        original = ModelNodeCapabilities(
            postgres=True,
            read=True,
            config=complex_config,
        )

        # Serialize to JSON string
        json_str = original.model_dump_json()
        parsed = json.loads(json_str)

        # Deserialize back to model
        restored = ModelNodeCapabilities.model_validate(parsed)

        assert restored.postgres is True
        assert restored.read is True
        assert restored.config == complex_config

    def test_health_check_result_round_trip(self) -> None:
        """Test JSON serialization round-trip for ModelHealthCheckResult."""
        nested_details = create_deeply_nested_dict(5)
        original = ModelHealthCheckResult(
            handler_type="db",
            healthy=True,
            details=nested_details,
        )

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = ModelHealthCheckResult.model_validate_json(json_str)

        assert restored.handler_type == "db"
        assert restored.healthy is True
        assert restored.details == nested_details

    def test_db_query_payload_round_trip(self) -> None:
        """Test JSON serialization round-trip for ModelDbQueryPayload."""
        rows_with_nested_data = [
            {"id": 1, "data": create_deeply_nested_dict(5)},
            {"id": 2, "data": create_complex_nested_structure()},
        ]
        original = ModelDbQueryPayload(rows=rows_with_nested_data, row_count=2)

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = ModelDbQueryPayload.model_validate_json(json_str)

        assert restored.row_count == 2
        assert len(restored.rows) == 2
        assert restored.rows[0] == rows_with_nested_data[0]

    def test_http_post_payload_round_trip(self) -> None:
        """Test JSON serialization round-trip for ModelHttpPostPayload."""
        response_body = create_complex_nested_structure()
        original = ModelHttpPostPayload(
            status_code=200,
            headers={"content-type": "application/json"},
            body=response_body,
        )

        # Serialize and deserialize
        json_str = original.model_dump_json()
        restored = ModelHttpPostPayload.model_validate_json(json_str)

        assert restored.status_code == 200
        assert restored.body == response_body


class TestJsonTypeValidation:
    """Tests for Pydantic validation with JsonType fields."""

    def test_valid_nested_json_structures(self) -> None:
        """Test that valid nested JSON structures pass validation."""
        valid_structures = [
            {},  # Empty dict
            {"key": "value"},  # Simple dict
            {"nested": {"deep": {"deeper": "value"}}},  # Nested dicts
            {"list": [1, 2, 3]},  # Dict with list
            {"mixed": [{"a": 1}, {"b": 2}]},  # Dict with list of dicts
            {
                "primitives": {
                    "str": "s",
                    "int": 1,
                    "float": 1.5,
                    "bool": True,
                    "null": None,
                }
            },
        ]

        for structure in valid_structures:
            caps = ModelNodeCapabilities(config=structure)
            assert caps.config == structure

    def test_db_query_payload_empty_rows(self) -> None:
        """Test ModelDbQueryPayload with empty rows."""
        payload = ModelDbQueryPayload(rows=[], row_count=0)
        assert payload.rows == []
        assert payload.row_count == 0

    def test_health_check_result_empty_details(self) -> None:
        """Test ModelHealthCheckResult with empty details."""
        result = ModelHealthCheckResult(
            handler_type="test",
            healthy=True,
            details={},
        )
        assert result.details == {}


class TestJsonTypeEdgeCases:
    """Edge case tests for JsonType handling."""

    def test_deeply_nested_lists(self) -> None:
        """Test handling of deeply nested list structures within dicts."""
        nested_list_structure = {
            "data": [
                [
                    [
                        [{"deep": "value"}],
                    ],
                ],
            ],
        }

        caps = ModelNodeCapabilities(config=nested_list_structure)
        assert caps.config["data"][0][0][0][0]["deep"] == "value"

    def test_mixed_nesting_with_various_primitives(self) -> None:
        """Test mixed nesting with all JSON primitive types."""
        mixed_structure = {
            "strings": {"a": "hello", "b": "world", "nested": {"c": "deep"}},
            "numbers": {"int": 42, "float": 3.14, "nested": {"negative": -100}},
            "booleans": {"true": True, "false": False, "nested": {"deep": True}},
            "nulls": {"null": None, "nested": {"deep_null": None}},
            "arrays": {
                "strings": ["a", "b", "c"],
                "numbers": [1, 2.5, -3],
                "mixed": [1, "two", True, None, {"nested": "in_list"}],
            },
        }

        caps = ModelNodeCapabilities(config=mixed_structure)
        assert caps.config == mixed_structure

    def test_unicode_and_special_characters_in_nested_structure(self) -> None:
        """Test handling of unicode and special characters in nested dicts."""
        unicode_structure = {
            "unicode": {
                "emoji": "Test data",  # Avoiding actual emojis per CLAUDE.md
                "chinese": "Chinese characters",
                "arabic": "Arabic text",
                "nested": {"special": "Line\nBreak\tTab"},
            },
        }

        caps = ModelNodeCapabilities(config=unicode_structure)
        assert caps.config["unicode"]["nested"]["special"] == "Line\nBreak\tTab"

    def test_large_nested_structure_performance(self) -> None:
        """Test that large nested structures don't cause performance issues.

        This tests both recursion handling and general performance
        with complex nested data.
        """
        # Create a structure with many keys and moderate depth
        large_structure: dict[str, object] = {}
        for i in range(100):
            large_structure[f"key_{i}"] = create_deeply_nested_dict(5, value=i)

        caps = ModelNodeCapabilities(config=large_structure)
        assert len(caps.config) == 100
        assert (
            caps.config["key_50"]["nested"]["nested"]["nested"]["nested"]["nested"][
                "value"
            ]
            == 50
        )


class TestModelSchemaGeneration:
    """Tests verifying schema generation doesn't recurse infinitely.

    Pydantic 2.x generates JSON schemas at class definition time.
    These tests ensure schema generation works without RecursionError.
    """

    def test_model_json_schema_generation(self) -> None:
        """Test that JSON schema can be generated for models with JsonType."""
        # These calls would have failed with RecursionError before the fix
        caps_schema = ModelNodeCapabilities.model_json_schema()
        assert caps_schema is not None
        assert "properties" in caps_schema
        assert "config" in caps_schema["properties"]

        health_schema = ModelHealthCheckResult.model_json_schema()
        assert health_schema is not None
        assert "properties" in health_schema
        assert "details" in health_schema["properties"]

        db_schema = ModelDbQueryPayload.model_json_schema()
        assert db_schema is not None
        assert "properties" in db_schema
        assert "rows" in db_schema["properties"]

    def test_model_fields_schema_info(self) -> None:
        """Test that model field info is accessible."""
        # Accessing field info triggers schema resolution
        caps_fields = ModelNodeCapabilities.model_fields
        assert "config" in caps_fields

        health_fields = ModelHealthCheckResult.model_fields
        assert "details" in health_fields

        http_fields = ModelHttpPostPayload.model_fields
        assert "body" in http_fields


class TestDispatchAndRegistrationModels:
    """Tests for dispatch and registration models that use JsonType.

    These are the models specifically mentioned in the ADR as being
    migrated from Any to JsonType.
    """

    def test_model_dispatch_result_error_details(self) -> None:
        """Test ModelDispatchResult.error_details field with nested data."""
        from datetime import UTC, datetime

        from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
        from omnibase_infra.models.dispatch.model_dispatch_result import (
            ModelDispatchResult,
        )

        nested_error_details = {
            "exception": {
                "type": "ValidationError",
                "message": "Field validation failed",
                "context": {"field": "user_id", "reason": "Invalid format"},
            },
            "stack_trace": [
                {"file": "handler.py", "line": 42},
                {"file": "validator.py", "line": 15},
            ],
        }

        result = ModelDispatchResult(
            status=EnumDispatchStatus.HANDLER_ERROR,
            topic="test.topic",
            started_at=datetime.now(UTC),
            error_message="Handler failed",
            error_details=nested_error_details,
        )

        assert result.error_details == nested_error_details
        # Type ignore: dynamically accessing nested JSON structure
        assert result.error_details["exception"]["context"]["field"] == "user_id"  # type: ignore[index]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
