"""Test suite for EnumDatabaseOperationType."""

from enum import Enum

import pytest

from omninode_bridge.nodes.database_adapter_effect.v1_0_0.enums.enum_database_operation_type import (
    EnumDatabaseOperationType,
)


class TestEnumDatabaseOperationType:
    """Test suite for EnumDatabaseOperationType."""

    def test_enum_values(self):
        """Test that all enum values are correctly defined."""
        # Core CRUD Operations
        assert EnumDatabaseOperationType.INSERT == "insert"
        assert EnumDatabaseOperationType.UPDATE == "update"
        assert EnumDatabaseOperationType.DELETE == "delete"
        assert EnumDatabaseOperationType.QUERY == "query"
        assert EnumDatabaseOperationType.UPSERT == "upsert"

        # Batch Operations
        assert EnumDatabaseOperationType.BATCH_INSERT == "batch_insert"
        assert EnumDatabaseOperationType.BATCH_UPDATE == "batch_update"
        assert EnumDatabaseOperationType.BATCH_DELETE == "batch_delete"

        # Utility Operations
        assert EnumDatabaseOperationType.COUNT == "count"
        assert EnumDatabaseOperationType.EXISTS == "exists"
        assert EnumDatabaseOperationType.HEALTH_CHECK == "health_check"

    def test_enum_inheritance(self):
        """Test that enum inherits from str and Enum."""
        assert issubclass(EnumDatabaseOperationType, str)
        assert issubclass(EnumDatabaseOperationType, Enum)

    def test_enum_string_behavior(self):
        """Test that enum values behave like strings."""
        operation = EnumDatabaseOperationType.INSERT
        assert isinstance(operation, str)
        assert operation == "insert"
        assert len(operation) == 6

    def test_enum_serialization(self):
        """Test enum serialization to JSON."""
        operation = EnumDatabaseOperationType.QUERY
        serialized = operation.value
        assert serialized == "query"

        import json

        json_str = json.dumps(operation)
        assert json_str == '"query"'

    def test_enum_iteration(self):
        """Test that enum can be iterated over."""
        operations = list(EnumDatabaseOperationType)
        assert len(operations) == 11  # Total number of operations
        assert EnumDatabaseOperationType.INSERT in operations
        assert EnumDatabaseOperationType.HEALTH_CHECK in operations

    def test_enum_membership(self):
        """Test enum membership operations."""
        assert "insert" in EnumDatabaseOperationType
        assert "invalid_operation" not in EnumDatabaseOperationType
        assert EnumDatabaseOperationType.INSERT in EnumDatabaseOperationType

    def test_enum_comparison(self):
        """Test enum comparison operations."""
        op1 = EnumDatabaseOperationType.INSERT
        op2 = EnumDatabaseOperationType.INSERT
        assert op1 == op2
        assert op1 is op2

    def test_enum_invalid_value(self):
        """Test that invalid enum values raise ValueError."""
        with pytest.raises(ValueError):
            EnumDatabaseOperationType("invalid_operation")

    def test_all_values(self):
        """Test that all expected enum values are present."""
        expected_values = {
            # Core CRUD
            "insert",
            "update",
            "delete",
            "query",
            "upsert",
            # Batch
            "batch_insert",
            "batch_update",
            "batch_delete",
            # Utility
            "count",
            "exists",
            "health_check",
        }
        actual_values = {op.value for op in EnumDatabaseOperationType}
        assert actual_values == expected_values

    def test_is_read_operation(self):
        """Test the is_read_operation method."""
        # Read operations
        assert EnumDatabaseOperationType.QUERY.is_read_operation()
        assert EnumDatabaseOperationType.COUNT.is_read_operation()
        assert EnumDatabaseOperationType.EXISTS.is_read_operation()
        assert EnumDatabaseOperationType.HEALTH_CHECK.is_read_operation()

        # Write operations
        assert not EnumDatabaseOperationType.INSERT.is_read_operation()
        assert not EnumDatabaseOperationType.UPDATE.is_read_operation()
        assert not EnumDatabaseOperationType.DELETE.is_read_operation()
        assert not EnumDatabaseOperationType.UPSERT.is_read_operation()

        # Batch operations
        assert not EnumDatabaseOperationType.BATCH_INSERT.is_read_operation()
        assert not EnumDatabaseOperationType.BATCH_UPDATE.is_read_operation()
        assert not EnumDatabaseOperationType.BATCH_DELETE.is_read_operation()

    def test_is_batch_operation(self):
        """Test the is_batch_operation method."""
        # Batch operations
        assert EnumDatabaseOperationType.BATCH_INSERT.is_batch_operation()
        assert EnumDatabaseOperationType.BATCH_UPDATE.is_batch_operation()
        assert EnumDatabaseOperationType.BATCH_DELETE.is_batch_operation()

        # Single-record operations
        assert not EnumDatabaseOperationType.INSERT.is_batch_operation()
        assert not EnumDatabaseOperationType.UPDATE.is_batch_operation()
        assert not EnumDatabaseOperationType.DELETE.is_batch_operation()
        assert not EnumDatabaseOperationType.QUERY.is_batch_operation()
        assert not EnumDatabaseOperationType.UPSERT.is_batch_operation()

        # Utility operations
        assert not EnumDatabaseOperationType.COUNT.is_batch_operation()
        assert not EnumDatabaseOperationType.EXISTS.is_batch_operation()
        assert not EnumDatabaseOperationType.HEALTH_CHECK.is_batch_operation()

    def test_requires_entity_data(self):
        """Test the requires_entity_data method."""
        # Operations requiring entity_data
        assert EnumDatabaseOperationType.INSERT.requires_entity_data()
        assert EnumDatabaseOperationType.UPDATE.requires_entity_data()
        assert EnumDatabaseOperationType.UPSERT.requires_entity_data()

        # Operations not requiring entity_data
        assert not EnumDatabaseOperationType.DELETE.requires_entity_data()
        assert not EnumDatabaseOperationType.QUERY.requires_entity_data()
        assert not EnumDatabaseOperationType.COUNT.requires_entity_data()
        assert not EnumDatabaseOperationType.EXISTS.requires_entity_data()
        assert not EnumDatabaseOperationType.HEALTH_CHECK.requires_entity_data()

        # Batch operations use batch_data instead
        assert not EnumDatabaseOperationType.BATCH_INSERT.requires_entity_data()
        assert not EnumDatabaseOperationType.BATCH_UPDATE.requires_entity_data()
        assert not EnumDatabaseOperationType.BATCH_DELETE.requires_entity_data()

    def test_requires_query_filters(self):
        """Test the requires_query_filters method."""
        # Operations requiring query_filters
        assert EnumDatabaseOperationType.UPDATE.requires_query_filters()
        assert EnumDatabaseOperationType.DELETE.requires_query_filters()
        assert EnumDatabaseOperationType.UPSERT.requires_query_filters()
        assert EnumDatabaseOperationType.EXISTS.requires_query_filters()

        # Operations not requiring query_filters
        assert not EnumDatabaseOperationType.INSERT.requires_query_filters()
        assert not EnumDatabaseOperationType.QUERY.requires_query_filters()
        assert not EnumDatabaseOperationType.COUNT.requires_query_filters()
        assert not EnumDatabaseOperationType.HEALTH_CHECK.requires_query_filters()

        # Batch operations use batch_data with filters
        assert not EnumDatabaseOperationType.BATCH_INSERT.requires_query_filters()
        assert not EnumDatabaseOperationType.BATCH_UPDATE.requires_query_filters()
        assert not EnumDatabaseOperationType.BATCH_DELETE.requires_query_filters()

    def test_enum_docstring(self):
        """Test that enum has proper docstring."""
        assert EnumDatabaseOperationType.__doc__ is not None
        assert "Generic database operation types" in EnumDatabaseOperationType.__doc__
        assert "CRUD + batch operations" in EnumDatabaseOperationType.__doc__

    def test_operation_categories(self):
        """Test that operations are correctly categorized."""
        # Core CRUD
        core_crud = {
            EnumDatabaseOperationType.INSERT,
            EnumDatabaseOperationType.UPDATE,
            EnumDatabaseOperationType.DELETE,
            EnumDatabaseOperationType.QUERY,
            EnumDatabaseOperationType.UPSERT,
        }

        # Batch Operations
        batch_ops = {
            EnumDatabaseOperationType.BATCH_INSERT,
            EnumDatabaseOperationType.BATCH_UPDATE,
            EnumDatabaseOperationType.BATCH_DELETE,
        }

        # Utility Operations
        utility_ops = {
            EnumDatabaseOperationType.COUNT,
            EnumDatabaseOperationType.EXISTS,
            EnumDatabaseOperationType.HEALTH_CHECK,
        }

        # Verify no overlap
        assert len(core_crud & batch_ops) == 0
        assert len(core_crud & utility_ops) == 0
        assert len(batch_ops & utility_ops) == 0

        # Verify all operations are categorized
        all_ops = core_crud | batch_ops | utility_ops
        assert len(all_ops) == 11
        assert all_ops == set(EnumDatabaseOperationType)
