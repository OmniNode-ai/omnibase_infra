"""
Database Security Validator for SQL injection detection and input validation.

This module provides comprehensive security validation for database operations
including SQL injection pattern detection, query complexity scoring, and
parameter validation.

Performance Targets:
- Validation overhead: < 5ms per operation
- Pattern matching: < 2ms (using pre-compiled regex)
- Complexity scoring: < 1ms

Security Coverage:
- 17+ SQL injection patterns detected (UNION, Boolean, Time-based, Error-based, etc.)
- Query size limits (configurable, default 10KB)
- Parameter count limits (max 100 parameters)
- Parameter size limits (max 1MB total)
- Correlation ID validation (UUID format, non-empty)
- Query complexity scoring with dual thresholds:
  * Warning threshold: > 20 (generates warning, allows execution)
  * Reject threshold: > 50 (blocks execution)
  * Formula: (JOINs * 2) + (Subqueries * 3) + (Aggregations * 1)

Primary Entry Point:
- validate_operation(operation_input) - Validates complete database operation

Individual Validators:
- validate_query(sql) - SQL injection and size validation
- validate_parameters(params) - Parameter count and size validation
- validate_correlation_id(correlation_id) - UUID format validation
- validate_query_size(sql, max_size) - Query size limit check
- validate_parameter_count(parameters, max_count) - Parameter count limit check
- validate_parameter_size(parameters, max_size) - Total parameter size check
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar
from uuid import UUID

# Direct imports - omnibase_core is required
from omnibase_core import EnumCoreErrorCode, ModelOnexError

# Aliases for compatibility
OnexError = ModelOnexError

# Import for type checking only to avoid circular imports
if TYPE_CHECKING:
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0.models.inputs.model_database_operation_input import (
        ModelDatabaseOperationInput,
    )


@dataclass
class ValidationResult:
    """
    Result of a security validation check.

    Attributes:
        valid: Whether the validation passed (all checks succeeded)
        errors: List of error messages for failed validations
        warnings: List of warning messages for potential issues
        details: Additional validation details for observability
    """

    valid: bool
    errors: list[str]
    warnings: list[str]
    details: dict[str, Any] | None = None


class DatabaseSecurityValidator:
    """
    Database security validator with SQL injection detection.

    This validator provides comprehensive security checks for database operations:
    - SQL injection pattern detection (10+ patterns)
    - Query size and complexity validation
    - Parameter count and size limits
    - Correlation ID validation
    - Query complexity scoring to prevent DoS attacks

    Performance characteristics:
    - Pre-compiled regex patterns for fast matching
    - Single-pass validation where possible
    - Minimal memory allocation

    Example:
        validator = DatabaseSecurityValidator(
            max_query_size=10240,  # 10KB
            max_parameter_count=100,
            max_parameter_size=1048576,  # 1MB
        )

        # Validate query
        result = validator.validate_query(sql_query)
        if not result.is_valid:
            raise OnexError(
                error_code=EnumCoreErrorCode.SECURITY_VIOLATION_ERROR,
                message=result.error_message
            )

        # Complete validation
        if not validator.is_query_safe(sql_query, params):
            raise OnexError(
                error_code=EnumCoreErrorCode.SECURITY_VIOLATION_ERROR,
                message="Query failed security validation"
            )
    """

    # SQL Injection Patterns (pre-compiled for performance)
    # These patterns detect common SQL injection techniques
    _SQL_INJECTION_PATTERNS: ClassVar[list[re.Pattern[str]]] = [
        # 1. UNION-based injection
        re.compile(r"union\s+(all\s+)?select", re.IGNORECASE),
        # 2. Boolean-based blind injection
        re.compile(r"(and|or)\s+1\s*=\s*1", re.IGNORECASE),
        re.compile(r"(and|or)\s+'1'\s*=\s*'1'", re.IGNORECASE),
        # 3. String concatenation attacks
        re.compile(r"'\s*or\s*'", re.IGNORECASE),
        # 4. Stacked queries - DROP TABLE
        re.compile(r";?\s*drop\s+(table|database|schema)", re.IGNORECASE),
        # 5. Stacked queries - DELETE
        re.compile(r";?\s*delete\s+from", re.IGNORECASE),
        # 6. Stacked queries - TRUNCATE
        re.compile(r";?\s*truncate\s+table", re.IGNORECASE),
        # 7. Time-based blind injection - WAITFOR (SQL Server)
        re.compile(r"waitfor\s+delay", re.IGNORECASE),
        # 8. Time-based blind injection - SLEEP (MySQL)
        re.compile(r"sleep\s*\(", re.IGNORECASE),
        # 9. Time-based blind injection - pg_sleep (PostgreSQL)
        re.compile(r"pg_sleep\s*\(", re.IGNORECASE),
        # 10. Error-based injection - CONVERT
        re.compile(r"(and|or)\s+\d+\s*=\s*convert", re.IGNORECASE),
        # 11. Out-of-band injection - LOAD_FILE (MySQL)
        re.compile(r"load_file\s*\(", re.IGNORECASE),
        # 12. Out-of-band injection - INTO OUTFILE
        re.compile(r"into\s+(out|dump)file", re.IGNORECASE),
        # 13. Command execution - xp_cmdshell (SQL Server)
        re.compile(r"xp_cmdshell", re.IGNORECASE),
        # 14. Comment-based injection - SQL line comments
        re.compile(r"--[^\n]*$", re.MULTILINE),
        # 15. Comment-based injection - C-style block comments
        re.compile(r"/\*.*?\*/", re.DOTALL),
        # 16. Information schema probing
        re.compile(r"information_schema\.", re.IGNORECASE),
        # 17. System catalog access
        re.compile(r"(pg_catalog\.|sys\.)", re.IGNORECASE),
    ]

    # Empty UUID constant (prevent usage)
    _EMPTY_UUID = UUID("00000000-0000-0000-0000-000000000000")

    def __init__(
        self,
        max_query_size: int = 10240,  # 10KB default
        max_parameter_count: int = 100,
        max_parameter_size: int = 1048576,  # 1MB default
        complexity_warning_threshold: int = 20,  # Warn above this score
        complexity_reject_threshold: int = 50,  # Reject above this score
        enable_strict_validation: bool = True,
    ):
        """
        Initialize database security validator.

        Args:
            max_query_size: Maximum query size in bytes (default: 10KB)
            max_parameter_count: Maximum number of parameters (default: 100)
            max_parameter_size: Maximum size per parameter in bytes (default: 1MB)
            complexity_warning_threshold: Complexity score for warnings (default: 20)
            complexity_reject_threshold: Complexity score for rejection (default: 50)
            enable_strict_validation: Enable strict validation mode
        """
        self.max_query_size = max_query_size
        self.max_parameter_count = max_parameter_count
        self.max_parameter_size = max_parameter_size
        self.complexity_warning_threshold = complexity_warning_threshold
        self.complexity_reject_threshold = complexity_reject_threshold
        self.enable_strict_validation = enable_strict_validation

    def validate_query(self, sql: str) -> ValidationResult:
        """
        Validate SQL query for security issues.

        Checks:
        - Query size limits
        - SQL injection patterns
        - Query complexity (with warning and reject thresholds)
        - Dangerous SQL commands

        Args:
            sql: SQL query to validate

        Returns:
            ValidationResult with errors and warnings lists

        Example:
            result = validator.validate_query("SELECT * FROM users WHERE id = $1")
            if not result.valid:
                for error in result.errors:
                    logger.error(f"Query validation failed: {error}")
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not sql or not isinstance(sql, str):
            errors.append("Query must be a non-empty string")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
            )

        # Check query size
        query_size = len(sql.encode("utf-8"))
        if query_size > self.max_query_size:
            errors.append(
                f"Query size ({query_size} bytes) exceeds maximum ({self.max_query_size} bytes)"
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                details={"query_size": query_size, "max_size": self.max_query_size},
            )

        # Check for SQL injection patterns
        for i, pattern in enumerate(self._SQL_INJECTION_PATTERNS, 1):
            if pattern.search(sql):
                errors.append(
                    f"SQL injection pattern #{i} detected: {pattern.pattern[:50]}..."
                )
                return ValidationResult(
                    valid=False,
                    errors=errors,
                    warnings=warnings,
                    details={
                        "pattern_index": i,
                        "pattern": pattern.pattern,
                        "query_preview": sql[:100],
                    },
                )

        # Calculate query complexity
        complexity = self.calculate_query_complexity(sql)

        # Check complexity thresholds
        if complexity > self.complexity_reject_threshold:
            errors.append(
                f"Query complexity ({complexity}) exceeds reject threshold ({self.complexity_reject_threshold})"
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                details={
                    "complexity": complexity,
                    "reject_threshold": self.complexity_reject_threshold,
                },
            )
        elif complexity > self.complexity_warning_threshold:
            warnings.append(
                f"Query complexity ({complexity}) exceeds warning threshold ({self.complexity_warning_threshold})"
            )

        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            details={"query_size": query_size, "complexity": complexity},
        )

    def validate_parameters(self, params: list) -> ValidationResult:
        """
        Validate query parameters for security issues.

        Checks:
        - Parameter count limits
        - Parameter size limits
        - Parameter type safety

        Args:
            params: List of query parameters

        Returns:
            ValidationResult with errors and warnings lists

        Example:
            result = validator.validate_parameters([user_id, email])
            if not result.valid:
                for error in result.errors:
                    logger.error(f"Parameter validation failed: {error}")
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(params, list):
            errors.append("Parameters must be a list")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
            )

        # Check parameter count
        param_count = len(params)
        if param_count > self.max_parameter_count:
            errors.append(
                f"Parameter count ({param_count}) exceeds maximum ({self.max_parameter_count})"
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                details={
                    "param_count": param_count,
                    "max_count": self.max_parameter_count,
                },
            )

        # Check individual parameter sizes
        total_param_size = 0
        for i, param in enumerate(params):
            if param is None:
                continue

            # Convert to string for size check
            param_str = str(param)
            param_size = len(param_str.encode("utf-8"))
            total_param_size += param_size

            if param_size > self.max_parameter_size:
                errors.append(
                    f"Parameter {i} size ({param_size} bytes) exceeds maximum ({self.max_parameter_size} bytes)"
                )
                return ValidationResult(
                    valid=False,
                    errors=errors,
                    warnings=warnings,
                    details={
                        "parameter_index": i,
                        "param_size": param_size,
                        "max_size": self.max_parameter_size,
                    },
                )

        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            details={"param_count": param_count, "total_param_size": total_param_size},
        )

    def validate_correlation_id(self, correlation_id: UUID) -> ValidationResult:
        """
        Validate correlation ID format and value.

        Checks:
        - UUID format validation
        - Empty UUID prevention
        - Non-null validation

        Args:
            correlation_id: Correlation ID to validate

        Returns:
            ValidationResult with errors and warnings lists

        Example:
            result = validator.validate_correlation_id(correlation_id)
            if not result.valid:
                for error in result.errors:
                    logger.error(f"Correlation ID validation failed: {error}")
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not correlation_id:
            errors.append("Correlation ID cannot be null or empty")
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
            )

        # Check for UUID type
        if not isinstance(correlation_id, UUID):
            errors.append(
                f"Correlation ID must be a UUID instance, got {type(correlation_id).__name__}"
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                details={"type": type(correlation_id).__name__},
            )

        # Check for empty UUID (all zeros)
        if correlation_id == self._EMPTY_UUID:
            errors.append(
                "Correlation ID cannot be empty UUID (00000000-0000-0000-0000-000000000000)"
            )
            return ValidationResult(
                valid=False,
                errors=errors,
                warnings=warnings,
                details={"correlation_id": str(correlation_id)},
            )

        return ValidationResult(
            valid=True,
            errors=errors,
            warnings=warnings,
            details={"correlation_id": str(correlation_id)},
        )

    def calculate_query_complexity(self, sql: str) -> int:
        """
        Calculate query complexity score to prevent DoS attacks.

        Uses simplified complexity formula as specified:
        Score = (JOINs * 2) + (Subqueries * 3) + (Aggregations * 1)

        Thresholds:
        - Warning: score > 20 (generates warning, allows execution)
        - Reject: score > 50 (blocks execution)

        Args:
            sql: SQL query to analyze

        Returns:
            Complexity score (higher = more complex)

        Example:
            complexity = validator.calculate_query_complexity(query)
            if complexity > 50:
                raise OnexError("Query complexity too high")
            elif complexity > 20:
                logger.warning("High complexity query", complexity=complexity)
        """
        if not sql:
            return 0

        sql_upper = sql.upper()

        # Count JOINs (weight: 2)
        # Count all JOIN keywords, but avoid double-counting compound joins
        join_count = sql_upper.count(" JOIN")
        complexity_joins = join_count * 2

        # Count subqueries (weight: 3)
        # Heuristic: count SELECT keywords after the first one
        select_count = sql_upper.count("SELECT")
        subquery_count = max(0, select_count - 1)
        complexity_subqueries = subquery_count * 3

        # Count aggregate functions (weight: 1)
        aggregate_count = (
            sql_upper.count("COUNT(")
            + sql_upper.count("SUM(")
            + sql_upper.count("AVG(")
            + sql_upper.count("MAX(")
            + sql_upper.count("MIN(")
            + sql_upper.count("GROUP_CONCAT(")
            + sql_upper.count("STRING_AGG(")
        )
        complexity_aggregations = aggregate_count * 1

        # Total complexity score
        total_complexity = (
            complexity_joins + complexity_subqueries + complexity_aggregations
        )

        return total_complexity

    def validate_operation(
        self, operation_input: "ModelDatabaseOperationInput"
    ) -> ValidationResult:
        """
        Validate a complete database operation input.

        This is the primary validation method that should be called before
        executing any database operation. It validates the correlation ID
        and performs operation-specific validation based on the operation type.

        Args:
            operation_input: Database operation input model to validate

        Returns:
            ValidationResult with aggregated errors and warnings

        Raises:
            OnexError: If validation fails in strict mode

        Example:
            from models.inputs.model_database_operation_input import ModelDatabaseOperationInput

            operation_input = ModelDatabaseOperationInput(
                operation_type="persist_workflow_execution",
                correlation_id=uuid4(),
                workflow_execution_data={...}
            )

            result = validator.validate_operation(operation_input)
            if not result.valid:
                for error in result.errors:
                    logger.error(f"Validation failed: {error}")
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="; ".join(result.errors)
                )
        """
        all_errors = []
        all_warnings = []

        # Step 1: Validate correlation ID
        correlation_result = self.validate_correlation_id(
            operation_input.correlation_id
        )
        all_errors.extend(correlation_result.errors)
        all_warnings.extend(correlation_result.warnings)

        if not correlation_result.valid:
            if self.enable_strict_validation:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message="; ".join(correlation_result.errors),
                )
            return ValidationResult(
                valid=False,
                errors=all_errors,
                warnings=all_warnings,
                details={"operation_type": operation_input.operation_type},
            )

        # Step 2: Operation-specific validation
        # For database adapter operations, we validate the entity data exists for operations that require it
        # Operations that require entity data: INSERT, UPDATE, UPSERT
        # Operations that use batch_entities: BATCH_INSERT, BATCH_UPDATE
        # Operations that don't need entity: DELETE, QUERY, COUNT, EXISTS, HEALTH_CHECK

        requires_entity = operation_input.operation_type in [
            "insert",
            "update",
            "upsert",
            # Legacy operation types (Phase 1 compatibility)
            "persist_workflow_execution",
            "persist_workflow_step",
            "persist_bridge_state",
            "persist_fsm_transition",
            "persist_metadata_stamp",
            "update_node_heartbeat",
        ]

        requires_batch_entities = operation_input.operation_type in [
            "batch_insert",
            "batch_update",
            "batch_delete",
        ]

        # Get entity data (either single entity or batch entities)
        operation_data = None
        if requires_entity and hasattr(operation_input, "entity"):
            operation_data = operation_input.entity
        elif requires_batch_entities and hasattr(operation_input, "batch_entities"):
            operation_data = operation_input.batch_entities

        # Check that the required data field is present for operations that need it
        if requires_entity and operation_data is None:
            all_errors.append(
                f"Missing entity data for operation type '{operation_input.operation_type}'"
            )
        elif requires_batch_entities and operation_data is None:
            all_errors.append(
                f"Missing batch_entities data for operation type '{operation_input.operation_type}'"
            )

        # Check for data size limits (prevent memory exhaustion)
        if operation_data is not None:
            # Convert to string representation and check size
            import json

            try:
                # Handle both single entities and lists of entities
                if isinstance(operation_data, list):
                    # For batch operations, validate total size
                    data_str = json.dumps(
                        [
                            (
                                entity.model_dump(mode="json")
                                if hasattr(entity, "model_dump")
                                else entity
                            )
                            for entity in operation_data
                        ]
                    )
                else:
                    # For single entities, convert to dict first with JSON mode for UUID serialization
                    entity_dict = (
                        operation_data.model_dump(mode="json")
                        if hasattr(operation_data, "model_dump")
                        else operation_data
                    )
                    data_str = json.dumps(entity_dict)

                data_size = len(data_str.encode("utf-8"))
                max_data_size = 1048576  # 1MB limit for operation data

                if data_size > max_data_size:
                    all_errors.append(
                        f"Operation data size ({data_size} bytes) exceeds maximum ({max_data_size} bytes)"
                    )
                elif data_size > max_data_size * 0.8:  # 80% threshold warning
                    all_warnings.append(
                        f"Operation data size ({data_size} bytes) is approaching limit ({max_data_size} bytes)"
                    )
            except (TypeError, ValueError, AttributeError) as e:
                all_errors.append(f"Invalid operation data format: {e}")

        # Return aggregated validation result
        is_valid = len(all_errors) == 0

        if not is_valid and self.enable_strict_validation:
            raise OnexError(
                error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                message="; ".join(all_errors),
            )

        return ValidationResult(
            valid=is_valid,
            errors=all_errors,
            warnings=all_warnings,
            details={
                "operation_type": operation_input.operation_type,
                "correlation_id": str(operation_input.correlation_id),
            },
        )

    # === Helper Validation Methods ===

    def validate_query_size(self, sql: str, max_size: int = 10240) -> ValidationResult:
        """
        Validate query size against maximum limit.

        Args:
            sql: SQL query to validate
            max_size: Maximum query size in bytes (default: 10KB)

        Returns:
            ValidationResult with size validation result

        Example:
            result = validator.validate_query_size(query, max_size=8192)
            if not result.valid:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=result.errors[0]
                )
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not sql or not isinstance(sql, str):
            errors.append("Query must be a non-empty string")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        query_size = len(sql.encode("utf-8"))
        if query_size > max_size:
            errors.append(
                f"Query size ({query_size} bytes) exceeds maximum ({max_size} bytes)"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={"query_size": query_size, "max_size": max_size},
        )

    def validate_parameter_count(
        self, parameters: list, max_count: int = 100
    ) -> ValidationResult:
        """
        Validate parameter count against maximum limit.

        Args:
            parameters: List of query parameters
            max_count: Maximum parameter count (default: 100)

        Returns:
            ValidationResult with parameter count validation result

        Example:
            result = validator.validate_parameter_count(params, max_count=50)
            if not result.valid:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=result.errors[0]
                )
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(parameters, list):
            errors.append("Parameters must be a list")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        param_count = len(parameters)
        if param_count > max_count:
            errors.append(
                f"Parameter count ({param_count}) exceeds maximum ({max_count})"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={"param_count": param_count, "max_count": max_count},
        )

    def validate_parameter_size(
        self, parameters: list, max_size: int = 1048576
    ) -> ValidationResult:
        """
        Validate total parameter size against maximum limit.

        Args:
            parameters: List of query parameters
            max_size: Maximum total parameter size in bytes (default: 1MB)

        Returns:
            ValidationResult with parameter size validation result

        Example:
            result = validator.validate_parameter_size(params, max_size=524288)
            if not result.valid:
                raise OnexError(
                    error_code=EnumCoreErrorCode.VALIDATION_ERROR,
                    message=result.errors[0]
                )
        """
        errors: list[str] = []
        warnings: list[str] = []

        if not isinstance(parameters, list):
            errors.append("Parameters must be a list")
            return ValidationResult(valid=False, errors=errors, warnings=warnings)

        # Calculate total parameter size
        total_size = 0
        for param in parameters:
            if param is None:
                continue
            param_str = str(param)
            total_size += len(param_str.encode("utf-8"))

        if total_size > max_size:
            errors.append(
                f"Total parameter size ({total_size} bytes) exceeds maximum ({max_size} bytes)"
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            details={"total_param_size": total_size, "max_size": max_size},
        )
