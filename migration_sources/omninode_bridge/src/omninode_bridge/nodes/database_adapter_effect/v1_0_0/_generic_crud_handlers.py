"""
Generic CRUD Handlers for Database Adapter Effect Node.

Implements 8 type-safe generic handlers using EntityRegistry for validation.
All handlers use parameterized queries for SQL injection prevention.

Handlers:
    1. _handle_insert - INSERT new record
    2. _handle_update - UPDATE existing records
    3. _handle_delete - DELETE records
    4. _handle_query - SELECT records with pagination
    5. _handle_upsert - INSERT ON CONFLICT DO UPDATE
    6. _handle_batch_insert - Batch INSERT
    7. _handle_count - COUNT records
    8. _handle_exists - EXISTS check

Implementation: Agent 5
"""

import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from omninode_bridge.security.validation import InputSanitizer
except ImportError:
    # Fallback for when validation is not available
    class InputSanitizer:  # type: ignore[no-redef]
        @staticmethod
        def validate_sql_identifier(value: str, max_length: int = 63) -> str:
            if not value or not isinstance(value, str):
                raise ValueError("SQL identifier must be a non-empty string")
            if len(value) > max_length:
                raise ValueError(
                    f"SQL identifier too long. Maximum length: {max_length}"
                )
            if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", value):
                raise ValueError(
                    "SQL identifier contains invalid characters or invalid start character"
                )
            return value


# Direct imports - omnibase_core is required
from omnibase_core import EnumCoreErrorCode, ModelOnexError

# Aliases for compatibility
OnexError = ModelOnexError


from omninode_bridge.infrastructure.entity_registry import EntityRegistry
from omninode_bridge.infrastructure.enum_entity_type import EnumEntityType

from .models.inputs.model_database_operation_input import ModelDatabaseOperationInput
from .models.outputs.model_database_operation_output import ModelDatabaseOperationOutput


class GenericCRUDHandlers:
    """
    Mixin class providing generic CRUD handlers with EntityRegistry validation.

    All handlers:
    - Use EntityRegistry for type-safe entity validation
    - Build parameterized SQL queries (no SQL injection)
    - Execute through circuit breaker for resilience
    - Include comprehensive error handling with OnexError
    - Track performance metrics

    Transaction Consistency Model:
    - Single operations (INSERT, UPDATE, DELETE, UPSERT): PostgreSQL autocommit
    - Batch operations (BATCH_INSERT): Explicit transaction wrapper

    See docs/TRANSACTION_CONSISTENCY_MODEL.md for comprehensive documentation
    including design rationale, performance trade-offs, and usage guidelines.
    """

    # Query complexity limits to prevent resource exhaustion
    MAX_LIMIT = 1000
    MAX_OFFSET = 10000
    MAX_BATCH_SIZE = 1000

    async def _handle_insert(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic INSERT handler with type validation.

        Transaction Model: Uses PostgreSQL autocommit for single-statement atomicity.
        Each INSERT is automatically wrapped in its own transaction by PostgreSQL,
        providing full ACID guarantees without explicit transaction management.

        Args:
            input_data: Operation input with entity field

        Returns:
            ModelDatabaseOperationOutput with generated ID

        Raises:
            OnexError: If entity validation fails or insert fails

        Note:
            For bulk inserts requiring all-or-nothing semantics, use _handle_batch_insert
            instead. See docs/TRANSACTION_CONSISTENCY_MODEL.md for details.

        Example:
            >>> from entities import ModelWorkflowExecution
            >>> from uuid import uuid4
            >>> workflow = ModelWorkflowExecution(
            ...     workflow_id="wf-123",
            ...     correlation_id=uuid4(),
            ...     status="processing",
            ...     namespace="test_app",
            ...     started_at=datetime.now(UTC)
            ... )
            >>> input_data = ModelDatabaseOperationInput(
            ...     operation_type=EnumDatabaseOperationType.INSERT,
            ...     entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            ...     correlation_id=uuid4(),
            ...     entity=workflow
            ... )
            >>> result = await self._handle_insert(input_data)
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Step 1: Validate entity type
            entity_type = self._validate_entity_type(entity_type_str)

            # Step 2: Validate entity field is provided
            if input_data.entity is None:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="entity field is required for INSERT operation",
                    context={
                        "operation_type": "insert",
                        "entity_type": entity_type_str,
                        "correlation_id": str(correlation_id),
                    },
                )

            # Step 3: Validate entity type matches expected model
            expected_model = EntityRegistry.get_model(entity_type)
            if not isinstance(input_data.entity, expected_model):
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"Entity type mismatch: expected {expected_model.__name__}, got {type(input_data.entity).__name__}",
                    context={
                        "expected_type": expected_model.__name__,
                        "actual_type": type(input_data.entity).__name__,
                        "entity_type": entity_type_str,
                        "correlation_id": str(correlation_id),
                    },
                )

            # Step 4: Serialize entity to dict for database
            entity_dict = EntityRegistry.serialize_entity(input_data.entity)

            # Step 5: Get table name from registry and validate it
            table_name = EntityRegistry.get_table_name(entity_type)
            validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

            # Step 6: Build parameterized INSERT query with validated identifiers
            columns = list(entity_dict.keys())
            # Validate all column names to prevent SQL injection
            validated_columns = [
                InputSanitizer.validate_sql_identifier(col) for col in columns
            ]
            placeholders = [f"${i+1}" for i in range(len(columns))]
            values = [entity_dict[col] for col in columns]

            query = f"""
                INSERT INTO {validated_table_name} ({', '.join(validated_columns)})
                VALUES ({', '.join(placeholders)})
                RETURNING id
            """

            # Step 7: Execute through circuit breaker
            result_rows = await self._circuit_breaker.execute(
                self._query_executor.execute_query, query, *values
            )

            # Step 8: Extract generated ID
            generated_id = result_rows[0]["id"] if result_rows else None

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Log success
            if self._logger:
                self._logger.log_operation_complete(
                    correlation_id=correlation_id,
                    execution_time_ms=execution_time_ms,
                    rows_affected=1,
                    operation_type="insert",
                    additional_context={
                        "entity_type": entity_type_str,
                        "table_name": table_name,
                        "generated_id": str(generated_id),
                    },
                )

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="insert",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                result_data={"id": str(generated_id)},
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"INSERT operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    async def _handle_update(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic UPDATE handler with type validation.

        Args:
            input_data: Operation input with entity and query_filters

        Returns:
            ModelDatabaseOperationOutput with rows affected

        Raises:
            OnexError: If validation fails or update fails
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Step 1: Validate entity type
            entity_type = self._validate_entity_type(entity_type_str)

            # Step 2: Validate required fields
            if input_data.entity is None:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="entity field is required for UPDATE operation",
                    context={
                        "operation_type": "update",
                        "entity_type": entity_type_str,
                    },
                )

            if not input_data.query_filters:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="query_filters field is required for UPDATE operation",
                    context={
                        "operation_type": "update",
                        "entity_type": entity_type_str,
                    },
                )

            # Step 3: Validate entity type matches
            expected_model = EntityRegistry.get_model(entity_type)
            if not isinstance(input_data.entity, expected_model):
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"Entity type mismatch: expected {expected_model.__name__}",
                )

            # Step 4: Serialize entity
            entity_dict = EntityRegistry.serialize_entity(input_data.entity)
            table_name = EntityRegistry.get_table_name(entity_type)

            # Step 5: Build UPDATE query
            query, params = self._build_update_query(
                table_name, entity_dict, input_data.query_filters
            )

            # Step 6: Execute through circuit breaker
            result = await self._circuit_breaker.execute(
                self._query_executor.execute_query, query, *params
            )

            # Parse rows affected using robust DML result parser
            rows_affected = self._parse_dml_result(
                result, "UPDATE", correlation_id, entity_type_str
            )

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="update",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=rows_affected,
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"UPDATE operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    async def _handle_delete(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic DELETE handler.

        Args:
            input_data: Operation input with query_filters

        Returns:
            ModelDatabaseOperationOutput with rows affected

        Raises:
            OnexError: If validation fails or delete fails
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Validate entity type
            entity_type = self._validate_entity_type(entity_type_str)

            # Validate query_filters provided
            if not input_data.query_filters:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="query_filters field is required for DELETE operation",
                    context={
                        "operation_type": "delete",
                        "entity_type": entity_type_str,
                    },
                )

            # Get table name
            table_name = EntityRegistry.get_table_name(entity_type)

            # Build DELETE query
            query, params = self._build_delete_query(
                table_name, input_data.query_filters
            )

            # Execute through circuit breaker
            result = await self._circuit_breaker.execute(
                self._query_executor.execute_query, query, *params
            )

            # Parse rows affected using robust DML result parser
            rows_affected = self._parse_dml_result(
                result, "DELETE", correlation_id, entity_type_str
            )

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="delete",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=rows_affected,
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"DELETE operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    async def _handle_query(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic QUERY handler with pagination and sorting.

        JSONB Deserialization:
        This handler supports two modes for JSONB field deserialization controlled
        by input_data.strict_jsonb_validation:

        - True (default, fail-fast): Raises OnexError on corrupted JSONB data
        - False (graceful degradation): Logs warning and continues with raw string

        See docs/TRANSACTION_CONSISTENCY_MODEL.md for query consistency guarantees.

        Args:
            input_data: Operation input with optional query_filters, pagination, and
                       strict_jsonb_validation flag for JSONB error handling

        Returns:
            ModelDatabaseOperationOutput with list of entities

        Raises:
            OnexError: If validation fails, query fails, or JSONB deserialization
                      fails with strict_jsonb_validation=True

        Example:
            # Fail-fast mode (default)
            >>> result = await self._handle_query(ModelDatabaseOperationInput(
            ...     operation_type=EnumDatabaseOperationType.QUERY,
            ...     entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            ...     correlation_id=uuid4(),
            ...     strict_jsonb_validation=True  # Raise error on corrupted JSONB
            ... ))

            # Graceful degradation mode
            >>> result = await self._handle_query(ModelDatabaseOperationInput(
            ...     operation_type=EnumDatabaseOperationType.QUERY,
            ...     entity_type=EnumEntityType.WORKFLOW_EXECUTION,
            ...     correlation_id=uuid4(),
            ...     strict_jsonb_validation=False  # Continue with raw string
            ... ))
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Validate entity type
            entity_type = self._validate_entity_type(entity_type_str)

            # Get table name and model
            table_name = EntityRegistry.get_table_name(entity_type)
            entity_model = EntityRegistry.get_model(entity_type)

            # Build SELECT query
            query, params = self._build_select_query(
                table_name=table_name,
                query_filters=input_data.query_filters,
                sort_by=input_data.sort_by,
                sort_order=input_data.sort_order,
                limit=input_data.limit,
                offset=input_data.offset,
            )

            # Execute through circuit breaker
            rows = await self._circuit_breaker.execute(
                self._query_executor.execute_query, query, *params
            )

            # Schema-aware JSONB deserialization using EntityRegistry
            # Explicitly checks for json_schema_extra={"db_type": "jsonb"} markers
            # Supports both fail-fast and graceful degradation modes via strict_jsonb_validation
            import json

            parsed_rows = []
            for row in rows:
                row_dict = dict(row)
                # Use schema-aware JSONB detection via EntityRegistry
                for field_name, field_info in entity_model.model_fields.items():
                    if field_name in row_dict and EntityRegistry._is_jsonb_field(
                        field_info
                    ):
                        value = row_dict[field_name]
                        # Only deserialize JSONB fields that are strings
                        if isinstance(value, str):
                            try:
                                row_dict[field_name] = json.loads(value)
                            except (json.JSONDecodeError, TypeError) as e:
                                # Handle based on strict_jsonb_validation setting
                                if input_data.strict_jsonb_validation:
                                    # Fail-fast mode (default): Raise OnexError with context
                                    raise OnexError(
                                        error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                                        message=f"Failed to deserialize JSONB field '{field_name}': {e!s}",
                                        context={
                                            "entity_type": entity_type_str,
                                            "field_name": field_name,
                                            "correlation_id": str(correlation_id),
                                            "validation_mode": "strict",
                                        },
                                        original_error=e,
                                    )
                                else:
                                    # Graceful degradation mode: Log warning and continue with raw string
                                    if self._logger:
                                        self._logger.log_operation_warning(
                                            correlation_id=correlation_id,
                                            warning_message=f"Failed to deserialize JSONB field '{field_name}', continuing with raw string value",
                                            additional_context={
                                                "entity_type": entity_type_str,
                                                "field_name": field_name,
                                                "error": str(e),
                                                "validation_mode": "graceful",
                                            },
                                        )
                                    else:
                                        logger.warning(
                                            f"Failed to deserialize JSONB field '{field_name}' for {entity_type_str} "
                                            f"(correlation_id={correlation_id}), continuing with raw string value: {e}"
                                        )
                                    # Leave value as raw string for caller to handle
                parsed_rows.append(row_dict)

            # Deserialize to strongly-typed entities
            entities = [entity_model(**row) for row in parsed_rows]

            # Serialize for output - wrap list in dict for consistent result_data structure
            result_data = {"items": [entity.model_dump() for entity in entities]}

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="query",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=len(entities),
                result_data=result_data,
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"QUERY operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    async def _handle_upsert(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic UPSERT handler using PostgreSQL ON CONFLICT.

        Args:
            input_data: Operation input with entity and query_filters (conflict keys)

        Returns:
            ModelDatabaseOperationOutput with upserted record

        Raises:
            OnexError: If validation fails or upsert fails
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Validate
            entity_type = self._validate_entity_type(entity_type_str)

            if input_data.entity is None:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="entity field is required for UPSERT operation",
                )

            if not input_data.query_filters:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="query_filters field is required for UPSERT (conflict key)",
                )

            # Serialize entity
            entity_dict = EntityRegistry.serialize_entity(input_data.entity)
            table_name = EntityRegistry.get_table_name(entity_type)

            # Validate table name to prevent SQL injection
            validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

            # Build UPSERT query (INSERT ... ON CONFLICT ... DO UPDATE)
            conflict_columns = list(input_data.query_filters.keys())
            columns = list(entity_dict.keys())
            # Validate all column names to prevent SQL injection
            validated_columns = [
                InputSanitizer.validate_sql_identifier(col) for col in columns
            ]
            validated_conflict_columns = [
                InputSanitizer.validate_sql_identifier(col) for col in conflict_columns
            ]

            placeholders = [f"${i+1}" for i in range(len(columns))]
            values = [entity_dict[col] for col in columns]

            # Build UPDATE SET clause for conflict resolution
            update_set = ", ".join(
                [
                    f"{validated_col} = EXCLUDED.{validated_col}"
                    for validated_col in validated_columns
                    if validated_col not in validated_conflict_columns
                ]
            )

            # Validate UPDATE SET is not empty
            if not update_set:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="UPSERT requires at least one non-conflict column to update",
                    context={
                        "conflict_columns": conflict_columns,
                        "all_columns": columns,
                        "entity_type": entity_type_str,
                    },
                )

            query = f"""
                INSERT INTO {validated_table_name} ({', '.join(validated_columns)})
                VALUES ({', '.join(placeholders)})
                ON CONFLICT ({', '.join(validated_conflict_columns)})
                DO UPDATE SET {update_set}
                RETURNING id
            """

            # Execute
            result_rows = await self._circuit_breaker.execute(
                self._query_executor.execute_query, query, *values
            )

            generated_id = result_rows[0]["id"] if result_rows else None

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="upsert",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=1,
                result_data={"id": str(generated_id)},
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"UPSERT operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    async def _handle_batch_insert(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic BATCH_INSERT handler for multiple records.

        Transaction Model: Uses explicit transaction wrapper for all-or-nothing semantics.
        All rows are inserted within a single transaction via connection_manager.transaction().
        If any row fails, the entire batch is automatically rolled back, ensuring consistent state.

        Circuit Breaker Integration: Transaction execution is wrapped with circuit breaker
        for resilience. This ensures that database connectivity failures are detected and
        handled gracefully, preventing cascading failures across the system.

        Args:
            input_data: Operation input with batch_entities list

        Returns:
            ModelDatabaseOperationOutput with list of generated IDs

        Raises:
            OnexError: If validation fails or batch insert fails
            CircuitBreakerOpenError: If circuit breaker is in OPEN state

        Note:
            - Maximum batch size: MAX_BATCH_SIZE (1000 rows)
            - All rows succeed together or all fail together (atomic)
            - Performance: ~0.3-0.4ms per row (vs 1-2ms for individual inserts)
            - Circuit breaker monitors transaction health and prevents cascading failures
            - See docs/TRANSACTION_CONSISTENCY_MODEL.md for detailed transaction documentation

        Example:
            >>> from entities import ModelWorkflowStep
            >>> steps = [
            ...     ModelWorkflowStep(workflow_id=uuid4(), step_name="step1", ...),
            ...     ModelWorkflowStep(workflow_id=uuid4(), step_name="step2", ...)
            ... ]
            >>> result = await self._handle_batch_insert(ModelDatabaseOperationInput(
            ...     operation_type=EnumDatabaseOperationType.BATCH_INSERT,
            ...     entity_type=EnumEntityType.WORKFLOW_STEP,
            ...     correlation_id=uuid4(),
            ...     batch_entities=steps
            ... ))
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Validate
            entity_type = self._validate_entity_type(entity_type_str)

            if not input_data.batch_entities:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="batch_entities field is required for BATCH_INSERT operation",
                )

            # Validate batch size doesn't exceed maximum to prevent resource exhaustion
            batch_size = len(input_data.batch_entities)
            if batch_size > self.MAX_BATCH_SIZE:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"Batch size exceeds maximum allowed value of {self.MAX_BATCH_SIZE}",
                    context={
                        "batch_size": batch_size,
                        "max_batch_size": self.MAX_BATCH_SIZE,
                        "entity_type": entity_type_str,
                    },
                )

            # Validate all entities
            expected_model = EntityRegistry.get_model(entity_type)
            for idx, entity in enumerate(input_data.batch_entities):
                if not isinstance(entity, expected_model):
                    raise OnexError(
                        error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                        message=f"Entity at index {idx} type mismatch",
                    )

            # Serialize all entities
            entities_dicts = [
                EntityRegistry.serialize_entity(entity)
                for entity in input_data.batch_entities
            ]

            table_name = EntityRegistry.get_table_name(entity_type)

            # Validate table name to prevent SQL injection
            validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

            # Build batch INSERT query
            if not entities_dicts:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="batch_entities list is empty",
                )

            # Get columns from first entity (all should have same structure)
            columns = list(entities_dicts[0].keys())
            # Validate all column names to prevent SQL injection
            validated_columns = [
                InputSanitizer.validate_sql_identifier(col) for col in columns
            ]

            # Build placeholders for all rows
            values_list = []
            params = []
            param_counter = 1

            for entity_dict in entities_dicts:
                row_placeholders = []
                for col in columns:
                    row_placeholders.append(f"${param_counter}")
                    params.append(entity_dict[col])
                    param_counter += 1
                values_list.append(f"({', '.join(row_placeholders)})")

            query = f"""
                INSERT INTO {validated_table_name} ({', '.join(validated_columns)})
                VALUES {', '.join(values_list)}
                RETURNING id
            """

            # Define transaction operation for circuit breaker execution
            # This ensures circuit breaker resilience patterns apply to batch operations
            async def execute_batch_transaction():
                """Execute batch insert within transaction with circuit breaker protection."""
                async with self._connection_manager.transaction() as conn:
                    # Execute query using transaction connection for ACID compliance
                    return await conn.fetch(query, *params)

            # Execute batch insert through circuit breaker for resilience
            # Circuit breaker monitors transaction health and prevents cascading failures
            # See docs/TRANSACTION_CONSISTENCY_MODEL.md for transaction semantics
            result_rows = await self._circuit_breaker.execute(execute_batch_transaction)

            generated_ids = [str(row["id"]) for row in result_rows]

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="batch_insert",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=len(generated_ids),
                result_data={"ids": generated_ids},
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"BATCH_INSERT operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    async def _handle_count(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic COUNT handler.

        Args:
            input_data: Operation input with optional query_filters

        Returns:
            ModelDatabaseOperationOutput with count

        Raises:
            OnexError: If validation fails or count fails
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Validate
            entity_type = self._validate_entity_type(entity_type_str)
            table_name = EntityRegistry.get_table_name(entity_type)

            # Validate table name to prevent SQL injection
            validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

            # Build COUNT query
            where_clause, params = self._build_where_clause(input_data.query_filters)

            query = f"SELECT COUNT(*) as count FROM {validated_table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"

            # Execute through circuit breaker
            rows = await self._circuit_breaker.execute(
                self._query_executor.execute_query, query, *params
            )

            # Extract count value from first row
            count_value = rows[0]["count"] if rows else 0

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="count",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=0,  # No data modification
                result_data={"count": count_value},
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"COUNT operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    async def _handle_exists(
        self, input_data: ModelDatabaseOperationInput
    ) -> ModelDatabaseOperationOutput:
        """
        Generic EXISTS handler.

        Args:
            input_data: Operation input with query_filters

        Returns:
            ModelDatabaseOperationOutput with exists boolean

        Raises:
            OnexError: If validation fails or exists check fails
        """
        start_time = time.perf_counter()
        correlation_id = input_data.correlation_id
        entity_type_str = input_data.entity_type

        try:
            # Validate
            entity_type = self._validate_entity_type(entity_type_str)

            if not input_data.query_filters:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="query_filters field is required for EXISTS operation",
                )

            table_name = EntityRegistry.get_table_name(entity_type)

            # Validate table name to prevent SQL injection
            validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

            # Build EXISTS query
            where_clause, params = self._build_where_clause(input_data.query_filters)

            query = f"SELECT EXISTS(SELECT 1 FROM {validated_table_name} WHERE {where_clause})"

            # Execute through circuit breaker
            rows = await self._circuit_breaker.execute(
                self._query_executor.execute_query, query, *params
            )

            # Extract exists boolean value from first row
            exists_value = rows[0]["exists"] if rows else False

            execution_time_ms = int((time.perf_counter() - start_time) * 1000)

            return ModelDatabaseOperationOutput(
                success=True,
                operation_type="exists",
                correlation_id=correlation_id,
                execution_time_ms=execution_time_ms,
                rows_affected=0,  # No data modification
                result_data={"exists": bool(exists_value)},
            )

        except OnexError:
            raise
        except Exception as e:
            execution_time_ms = int((time.perf_counter() - start_time) * 1000)
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"EXISTS operation failed: {e!s}",
                context={
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                    "execution_time_ms": execution_time_ms,
                },
                original_error=e,
            )

    # === Helper Methods ===

    def _validate_entity_type(self, entity_type_str: str) -> EnumEntityType:
        """
        Validate and convert entity_type string to enum.

        Args:
            entity_type_str: Entity type string

        Returns:
            EnumEntityType enum value

        Raises:
            OnexError: If entity type is invalid or not registered
        """
        try:
            entity_type = EnumEntityType(entity_type_str)
        except ValueError:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Invalid entity_type: {entity_type_str}",
                context={
                    "entity_type": entity_type_str,
                    "valid_types": [e.value for e in EnumEntityType],
                },
            )

        # Check if entity type is registered
        if not EntityRegistry.is_registered(entity_type):
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message=f"Entity type not registered: {entity_type_str}",
                context={"entity_type": entity_type_str},
            )

        return entity_type

    def _build_update_query(
        self,
        table_name: str,
        entity_dict: dict[str, Any],
        query_filters: dict[str, Any],
    ) -> tuple[str, list[Any]]:
        """
        Build parameterized UPDATE query.

        Args:
            table_name: Target table name
            entity_dict: Fields to update
            query_filters: WHERE clause conditions

        Returns:
            Tuple of (query_string, parameters_list)
        """
        # Build SET clause
        set_clauses = []
        params = []
        param_counter = 1

        # Validate table name
        validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

        for col, value in entity_dict.items():
            # Validate column name to prevent SQL injection
            validated_col = InputSanitizer.validate_sql_identifier(col)
            set_clauses.append(f"{validated_col} = ${param_counter}")
            params.append(value)
            param_counter += 1

        # Build WHERE clause
        where_clause, where_params = self._build_where_clause(
            query_filters, start_param=param_counter
        )
        params.extend(where_params)

        query = f"UPDATE {validated_table_name} SET {', '.join(set_clauses)} WHERE {where_clause}"

        return query, params

    def _build_delete_query(
        self, table_name: str, query_filters: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        """
        Build parameterized DELETE query.

        Args:
            table_name: Target table name
            query_filters: WHERE clause conditions

        Returns:
            Tuple of (query_string, parameters_list)
        """
        # Validate table name to prevent SQL injection
        validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

        where_clause, params = self._build_where_clause(query_filters)
        query = f"DELETE FROM {validated_table_name} WHERE {where_clause}"
        return query, params

    def _build_select_query(
        self,
        table_name: str,
        query_filters: Optional[dict[str, Any]] = None,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> tuple[str, list[Any]]:
        """
        Build parameterized SELECT query with pagination and sorting.

        Args:
            table_name: Target table name
            query_filters: Optional WHERE clause conditions
            sort_by: Optional sort field
            sort_order: Sort order (asc/desc)
            limit: Optional result limit
            offset: Optional result offset

        Returns:
            Tuple of (query_string, parameters_list)
        """
        # Validate table name to prevent SQL injection
        validated_table_name = InputSanitizer.validate_sql_identifier(table_name)

        query = f"SELECT * FROM {validated_table_name}"
        params = []

        # WHERE clause
        if query_filters:
            where_clause, where_params = self._build_where_clause(query_filters)
            query += f" WHERE {where_clause}"
            params.extend(where_params)

        # ORDER BY clause
        if sort_by:
            # Validate sort_by is not SQL injection using proper SQL identifier validation
            validated_sort_by = InputSanitizer.validate_sql_identifier(sort_by)
            query += f" ORDER BY {validated_sort_by} {sort_order.upper()}"
        else:
            # Default sort by primary key
            query += " ORDER BY id DESC"

        # LIMIT and OFFSET with bounds checking to prevent SQL injection
        if limit is not None:
            # Explicitly reject booleans (which are int subclass in Python)
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"LIMIT must be a non-negative integer, got: {limit}",
                    context={
                        "limit": limit,
                        "table_name": table_name,
                    },
                )
            if limit > self.MAX_LIMIT:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"LIMIT exceeds maximum allowed value of {self.MAX_LIMIT}",
                    context={
                        "limit": limit,
                        "max_limit": self.MAX_LIMIT,
                    },
                )
            query += f" LIMIT {limit}"  # Safe after validation

        if offset is not None:
            # Explicitly reject booleans (which are int subclass in Python)
            if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"OFFSET must be a non-negative integer, got: {offset}",
                    context={
                        "offset": offset,
                        "table_name": table_name,
                    },
                )
            if offset > self.MAX_OFFSET:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"OFFSET exceeds maximum allowed value of {self.MAX_OFFSET}",
                    context={
                        "offset": offset,
                        "max_offset": self.MAX_OFFSET,
                    },
                )
            # Only add OFFSET clause if > 0 (OFFSET 0 is a no-op)
            if offset > 0:
                query += f" OFFSET {offset}"  # Safe after validation

        return query, params

    def _build_where_clause(
        self, query_filters: Optional[dict[str, Any]], start_param: int = 1
    ) -> tuple[str, list[Any]]:
        """
        Build parameterized WHERE clause from filters.

        Supports:
        - Simple equality: {"field": "value"}
        - Comparison operators: {"field__gt": 10, "field__lt": 20}
        - IN operator: {"field__in": [1, 2, 3]}

        Args:
            query_filters: Filter conditions
            start_param: Starting parameter number

        Returns:
            Tuple of (where_clause_string, parameters_list)
        """
        if not query_filters:
            return "TRUE", []

        conditions = []
        params = []
        param_counter = start_param

        for key, value in query_filters.items():
            # Parse operator from key
            if "__" in key:
                field, operator = key.rsplit("__", 1)
            else:
                field, operator = key, "eq"

            # Validate field name and capture validated result to prevent SQL injection
            validated_field = InputSanitizer.validate_sql_identifier(field)

            # Build condition based on operator using validated field name
            if operator == "eq":
                conditions.append(f"{validated_field} = ${param_counter}")
                params.append(value)
                param_counter += 1
            elif operator == "gt":
                conditions.append(f"{validated_field} > ${param_counter}")
                params.append(value)
                param_counter += 1
            elif operator == "gte":
                conditions.append(f"{validated_field} >= ${param_counter}")
                params.append(value)
                param_counter += 1
            elif operator == "lt":
                conditions.append(f"{validated_field} < ${param_counter}")
                params.append(value)
                param_counter += 1
            elif operator == "lte":
                conditions.append(f"{validated_field} <= ${param_counter}")
                params.append(value)
                param_counter += 1
            elif operator == "ne":
                conditions.append(f"{validated_field} != ${param_counter}")
                params.append(value)
                param_counter += 1
            elif operator == "in":
                if not isinstance(value, list):
                    raise OnexError(
                        error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                        message=f"IN operator requires list value for field: {field}",
                    )
                placeholders = [f"${param_counter + i}" for i in range(len(value))]
                conditions.append(f"{validated_field} IN ({', '.join(placeholders)})")
                params.extend(value)
                param_counter += len(value)
            else:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=f"Unsupported operator: {operator}",
                )

        where_clause = " AND ".join(conditions)
        return where_clause, params

    def _parse_dml_result(
        self,
        result: Any,
        expected_command: str,
        correlation_id: Any,
        entity_type_str: str,
    ) -> int:
        r"""
        Parse PostgreSQL DML result with robust regex validation.

        PostgreSQL DML operations without RETURNING clause return a string in
        "COMMAND N" format (e.g., "UPDATE 5", "DELETE 3", "INSERT 1").
        This method validates the format and extracts the row count with
        comprehensive error handling.

        Args:
            result: Result from execute_query (string or list)
            expected_command: Expected SQL command ("UPDATE", "DELETE", "INSERT")
            correlation_id: Correlation ID for error context
            entity_type_str: Entity type string for error context

        Returns:
            Number of rows affected by the operation

        Raises:
            OnexError: If result format is invalid or unexpected

        Implementation Notes:
            - Validates format using regex: ^(UPDATE|DELETE|INSERT)\s+(\d+)$
            - Handles both string and list result types
            - Provides detailed error context for debugging
            - Falls back gracefully for unexpected formats

        Example:
            >>> rows = self._parse_dml_result("UPDATE 5", "UPDATE", uuid4(), "workflow_execution")
            >>> print(rows)
            5
        """
        # If result is a list (RETURNING clause used), count rows
        if isinstance(result, list):
            return len(result)

        # If result is not a string, raise error with context
        if not isinstance(result, str):
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Unexpected DML result type: expected string or list, got {type(result).__name__}",
                context={
                    "expected_command": expected_command,
                    "result_type": type(result).__name__,
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                },
            )

        # Validate result format with regex: "COMMAND N" (e.g., "UPDATE 5")
        # Pattern: ^(UPDATE|DELETE|INSERT)\s+(\d+)$
        dml_pattern = re.compile(r"^(UPDATE|DELETE|INSERT)\s+(\d+)$")
        match = dml_pattern.match(result.strip())

        if not match:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Invalid DML result format: expected '{expected_command} N', got '{result}'",
                context={
                    "expected_command": expected_command,
                    "result_value": result,
                    "expected_format": f"{expected_command} N (e.g., '{expected_command} 5')",
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                },
            )

        # Extract command and count
        command, count_str = match.groups()

        # Verify command matches expected
        if command != expected_command:
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"DML command mismatch: expected '{expected_command}', got '{command}'",
                context={
                    "expected_command": expected_command,
                    "actual_command": command,
                    "result_value": result,
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                },
            )

        # Parse and return count
        try:
            return int(count_str)
        except ValueError as e:
            # Should never happen after regex validation, but handle for robustness
            raise OnexError(
                error_code=EnumCoreErrorCode.DATABASE_QUERY_ERROR,
                message=f"Failed to parse row count from DML result: '{count_str}'",
                context={
                    "expected_command": expected_command,
                    "count_string": count_str,
                    "result_value": result,
                    "entity_type": entity_type_str,
                    "correlation_id": str(correlation_id),
                },
                original_error=e,
            )
