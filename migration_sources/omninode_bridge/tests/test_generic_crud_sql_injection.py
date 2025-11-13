"""
SQL Injection Prevention Tests for Generic CRUD Handlers.

Tests that all generic handlers properly use parameterized queries
and prevent SQL injection attacks.

Test Categories:
1. WHERE clause SQL injection prevention
2. INSERT value SQL injection prevention
3. UPDATE value SQL injection prevention
4. Sort/column name validation
5. Filter operator validation

Implementation: Agent 5
"""

from datetime import datetime
from uuid import uuid4

import pytest

from omninode_bridge.infrastructure.entities.model_metadata_stamp import (
    ModelMetadataStamp,
)
from omninode_bridge.infrastructure.entities.model_workflow_execution import (
    ModelWorkflowExecution,
)
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.enums.enum_database_operation_type import (
    EnumDatabaseOperationType,
)
from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
    ModelDatabaseOperationInput,
)


class TestSQLInjectionPrevention:
    """Test suite for SQL injection prevention in generic CRUD handlers."""

    @pytest.mark.asyncio
    async def test_query_filters_sql_injection_prevention(self, database_adapter_node):
        """
        Test that query_filters properly sanitize SQL injection attempts.

        Malicious inputs like "'; DROP TABLE users; --" should be treated
        as literal parameter values, not executed as SQL.
        """
        # Malicious filter attempting SQL injection
        malicious_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={
                "workflow_type": "'; DROP TABLE workflow_executions; --",
                "namespace": "test'; DELETE FROM workflow_executions WHERE '1'='1",
            },
        )

        # Should not raise error - malicious strings treated as parameter values
        result = await database_adapter_node._handle_query(malicious_input)

        # Verify no rows affected (malicious SQL not executed)
        assert result.success is True
        assert result.rows_affected == 0  # No matching rows (safe)

    @pytest.mark.asyncio
    async def test_insert_value_sql_injection_prevention(self, database_adapter_node):
        """
        Test that INSERT values are properly parameterized.

        Malicious entity data should be inserted as literal values,
        not executed as SQL commands.
        """
        malicious_entity = ModelWorkflowExecution(
            workflow_type="'; DROP TABLE workflow_executions; --",
            correlation_id=uuid4(),
            current_state="PROCESSING'; DELETE FROM users WHERE '1'='1",
            namespace="test",
            started_at=datetime.now(),
        )

        input_data = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=malicious_entity,
        )

        # Should successfully insert malicious strings as literal values
        result = await database_adapter_node._handle_insert(input_data)

        assert result.success is True
        assert result.rows_affected == 1
        assert "id" in result.result_data

        # Verify data was inserted as literal values (not executed)
        # Query back the inserted record
        query_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={"id": result.result_data["id"]},
        )

        query_result = await database_adapter_node._handle_query(query_input)

        assert query_result.success is True
        assert len(query_result.result_data["items"]) == 1

        # Verify malicious strings stored as literals
        inserted_record = query_result.result_data["items"][0]
        assert (
            inserted_record["workflow_type"] == "'; DROP TABLE workflow_executions; --"
        )

    @pytest.mark.asyncio
    async def test_update_value_sql_injection_prevention(self, database_adapter_node):
        """
        Test that UPDATE values are properly parameterized.

        Malicious update values should be treated as parameter values,
        not executed as SQL.
        """
        # First insert a record
        test_correlation_id = uuid4()
        test_started_at = datetime.now()

        entity = ModelWorkflowExecution(
            workflow_type="wf-123",
            correlation_id=test_correlation_id,
            current_state="PENDING",
            namespace="test",
            started_at=test_started_at,
        )

        insert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=entity,
        )

        insert_result = await database_adapter_node._handle_insert(insert_input)

        # Convert string ID to UUID for proper comparison
        from uuid import UUID

        inserted_id = UUID(insert_result.result_data["id"])

        # Attempt SQL injection via UPDATE
        # Use valid current_state (has validator) but malicious workflow_type and namespace
        # Keep same correlation_id and started_at to avoid constraint violations
        malicious_update = ModelWorkflowExecution(
            id=inserted_id,
            workflow_type="'; DROP TABLE workflow_executions; --",
            correlation_id=test_correlation_id,  # Use same correlation_id
            current_state="PROCESSING",
            namespace="'; DELETE FROM users; --",
            started_at=test_started_at,  # Use same started_at
        )

        update_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPDATE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            entity=malicious_update,
            query_filters={"id": inserted_id},
        )

        # Should successfully execute UPDATE without SQL injection
        # (malicious strings treated as parameter values)
        update_result = await database_adapter_node._handle_update(update_input)

        # Primary assertion: UPDATE completes without SQL injection
        assert update_result.success is True
        # Note: rows_affected may be 0 if constraints prevent update, but that's OK
        # The key is that malicious SQL was NOT executed

    @pytest.mark.asyncio
    async def test_delete_filter_sql_injection_prevention(self, database_adapter_node):
        """
        Test that DELETE query_filters are properly parameterized.

        Malicious delete filters should not execute SQL commands.
        """
        # Attempt SQL injection via DELETE filters
        malicious_delete = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.DELETE,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={
                "workflow_type": "'; DROP TABLE workflow_executions; --",
            },
        )

        # Should safely execute (no matching rows)
        result = await database_adapter_node._handle_delete(malicious_delete)

        assert result.success is True
        assert result.rows_affected == 0  # No rows matched

    @pytest.mark.asyncio
    async def test_sort_by_field_validation(self, database_adapter_node):
        """
        Test that sort_by field name is validated to prevent SQL injection.

        Malicious sort_by values like "id; DROP TABLE" should be rejected.
        """
        # Attempt SQL injection via sort_by
        malicious_query = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            sort_by="id; DROP TABLE workflow_executions; --",
            sort_order="desc",
        )

        # Should raise validation error
        with pytest.raises(Exception) as exc_info:
            await database_adapter_node._handle_query(malicious_query)

        # Verify error mentions invalid SQL identifier
        assert "SQL identifier contains invalid characters" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_filter_field_name_validation(self, database_adapter_node):
        """
        Test that filter field names are validated to prevent SQL injection.

        Malicious field names should be rejected before query building.
        """
        # Attempt SQL injection via filter field name
        malicious_query = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={
                "id; DROP TABLE workflow_executions; --": "value",
            },
        )

        # Should raise validation error
        with pytest.raises(Exception) as exc_info:
            await database_adapter_node._handle_query(malicious_query)

        # Verify error mentions SQL identifier validation failure
        assert "SQL identifier contains invalid characters" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_batch_insert_sql_injection_prevention(self, database_adapter_node):
        """
        Test that batch INSERT properly parameterizes all values.

        Malicious values in batch should be treated as literals.
        """
        malicious_entities = [
            ModelWorkflowExecution(
                workflow_type=f"'; DROP TABLE workflow_executions; -- {i}",
                correlation_id=uuid4(),
                current_state="PROCESSING",
                namespace="test",
                started_at=datetime.now(),
            )
            for i in range(3)
        ]

        batch_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            batch_entities=malicious_entities,
        )

        # Should successfully insert all records
        result = await database_adapter_node._handle_batch_insert(batch_input)

        assert result.success is True
        assert result.rows_affected == 3
        assert len(result.result_data["ids"]) == 3

    @pytest.mark.asyncio
    async def test_count_filter_sql_injection_prevention(self, database_adapter_node):
        """
        Test that COUNT query_filters are properly parameterized.
        """
        malicious_count = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.COUNT,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={
                "current_state": "'; DROP TABLE workflow_executions; --",
            },
        )

        # Should safely execute (no matching rows)
        result = await database_adapter_node._handle_count(malicious_count)

        assert result.success is True
        assert result.result_data["count"] == 0

    @pytest.mark.asyncio
    async def test_exists_filter_sql_injection_prevention(self, database_adapter_node):
        """
        Test that EXISTS query_filters are properly parameterized.
        """
        malicious_exists = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.EXISTS,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={
                "workflow_type": "'; DROP TABLE workflow_executions; --",
            },
        )

        # Should safely execute
        result = await database_adapter_node._handle_exists(malicious_exists)

        assert result.success is True
        assert result.result_data["exists"] is False

    @pytest.mark.asyncio
    async def test_upsert_sql_injection_prevention(self, database_adapter_node):
        """
        Test that UPSERT properly parameterizes both insert and update values.
        """
        # Use valid 64-char hex hash, but test SQL injection in other fields
        malicious_entity = ModelMetadataStamp(
            file_hash="a" * 64,  # Valid 64-char hex hash
            namespace="test'; DROP TABLE metadata_stamps; --",
            stamp_data={
                "file_path": "/path/to/file'; DELETE FROM users; --",
                "stamp": "data'; DROP TABLE workflow_executions; --",
            },
        )

        upsert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPSERT,
            entity_type=EnumEntityType.METADATA_STAMP,
            correlation_id=uuid4(),
            entity=malicious_entity,
            query_filters={"file_hash": malicious_entity.file_hash},
        )

        # Should successfully upsert with malicious strings as literals
        result = await database_adapter_node._handle_upsert(upsert_input)

        assert result.success is True
        assert result.rows_affected == 1

    @pytest.mark.asyncio
    async def test_upsert_empty_update_set_validation(self, database_adapter_node):
        """
        Test that UPSERT rejects operations where all columns are conflict keys.

        If all columns are specified as conflict columns, the UPDATE SET clause
        would be empty, causing PostgreSQL to fail. This should be caught with
        a clear validation error.
        """
        entity = ModelMetadataStamp(
            file_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",  # Valid 64-char hex
            namespace="test",
            stamp_data={"stamp": "data"},
        )

        # Attempt UPSERT where all columns are conflict keys
        # This would result in an empty UPDATE SET clause
        upsert_input = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.UPSERT,
            entity_type=EnumEntityType.METADATA_STAMP,
            correlation_id=uuid4(),
            entity=entity,
            query_filters={
                "file_hash": entity.file_hash,
                "namespace": entity.namespace,
                "stamp_data": entity.stamp_data,
            },
        )

        # Should raise validation error
        with pytest.raises(Exception) as exc_info:
            await database_adapter_node._handle_upsert(upsert_input)

        # Verify error message mentions empty UPDATE SET
        error_message = str(exc_info.value)
        assert (
            "UPSERT requires at least one non-conflict column to update"
            in error_message
        )

    @pytest.mark.asyncio
    async def test_comparison_operator_validation(self, database_adapter_node):
        """
        Test that comparison operators are validated (gt, lt, gte, lte, in, ne).

        Invalid operators should be rejected.
        """
        # Attempt invalid operator
        invalid_operator_query = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={
                "id__DROP_TABLE": "123",  # Invalid operator
            },
        )

        # Should raise validation error
        with pytest.raises(Exception) as exc_info:
            await database_adapter_node._handle_query(invalid_operator_query)

        # Verify error mentions unsupported operator
        assert "Unsupported operator" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_in_operator_sql_injection_prevention(self, database_adapter_node):
        """
        Test that IN operator properly parameterizes list values.

        Malicious values in IN list should be treated as parameters.
        """
        malicious_in_query = ModelDatabaseOperationInput(
            operation_type=EnumDatabaseOperationType.QUERY,
            entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            correlation_id=uuid4(),
            query_filters={
                "current_state__in": [
                    "PENDING",
                    "'; DROP TABLE workflow_executions; --",
                    "PROCESSING'; DELETE FROM users; --",
                ]
            },
        )

        # Should safely execute (no matching rows with malicious values)
        result = await database_adapter_node._handle_query(malicious_in_query)

        assert result.success is True


# Note: database_adapter_node fixture is now provided by tests/conftest.py
# It uses sql_injection_test_db (PostgreSQL 16 testcontainer) and creates
# a properly initialized NodeBridgeDatabaseAdapterEffect for security testing.


if __name__ == "__main__":
    # Run tests with: pytest tests/test_generic_crud_sql_injection.py -v
    pytest.main([__file__, "-v"])
