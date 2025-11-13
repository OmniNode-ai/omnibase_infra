"""
Comprehensive WHERE Clause Validation Tests.

Tests that WHERE clause field names are properly validated
to prevent SQL injection attacks through malicious field names.

Test Categories:
1. Validated field usage in all operators (eq, gt, gte, lt, lte, ne, in)
2. SQL injection via field names
3. Malicious field names rejection
4. Valid field names acceptance
5. Operator-specific validation
6. Complex filter combinations

Security Fix: WHERE clause field name validation to prevent SQL injection
Implementation: Agent 5
"""

import pytest

# Import OnexError from the actual implementation
# This ensures we catch the same exception type being raised
from omninode_bridge.nodes.database_adapter_effect.v1_0_0._generic_crud_handlers import (
    OnexError,
)


class TestWhereClauseFieldValidation:
    """Test suite for WHERE clause field name validation."""

    @pytest.mark.parametrize(
        "valid_field",
        [
            "id",
            "workflow_id",
            "correlation_id",
            "status",
            "namespace",
            "created_at",
            "updated_at",
            "workflow_type",
            "current_state",
            "step_order",
            "_private_field",
            "field_123",
            "CamelCaseField",
        ],
    )
    def test_where_clause_accepts_valid_field_names(
        self, valid_field, mock_crud_handler
    ):
        """
        Test that valid field names are accepted in WHERE clause.

        Valid field names are alphanumeric with underscores,
        not starting with a number.
        """
        where_clause, params = mock_crud_handler._build_where_clause(
            {valid_field: "test_value"}
        )

        # Validated field should appear in WHERE clause
        assert valid_field in where_clause
        assert "$1" in where_clause
        assert params == ["test_value"]

    @pytest.mark.parametrize(
        "malicious_field",
        [
            "id; DROP TABLE workflow_executions; --",
            "status' OR '1'='1",
            "namespace UNION SELECT * FROM users",
            "workflow_id--",
            "created_at/*",
            "updated_at*/",
            "field; DELETE FROM workflow_executions WHERE '1'='1",
            "id) OR 1=1--",
            "status' AND 1=2 UNION SELECT * FROM passwords--",
        ],
    )
    def test_where_clause_rejects_sql_injection_field_names(
        self, malicious_field, mock_crud_handler
    ):
        """
        Test that malicious field names with SQL injection attempts are rejected.

        Field names must be validated before being used in WHERE clauses
        to prevent SQL injection.
        """
        with pytest.raises((ValueError, OnexError)) as exc_info:
            mock_crud_handler._build_where_clause({malicious_field: "test_value"})

        # Should raise error about invalid field name
        error_msg = str(exc_info.value)
        assert (
            "invalid" in error_msg.lower()
            or "identifier" in error_msg.lower()
            or "character" in error_msg.lower()
        )

    @pytest.mark.parametrize(
        "operator",
        ["eq", "gt", "gte", "lt", "lte", "ne", "in"],
    )
    def test_validated_field_used_in_all_operators(self, operator, mock_crud_handler):
        """
        Test that validated field names are used in all comparison operators.

        All operators (eq, gt, gte, lt, lte, ne, in) must use the validated
        field name from InputSanitizer.validate_sql_identifier().
        """
        field_name = "workflow_id"

        if operator == "in":
            query_filters = {f"{field_name}__{operator}": ["value1", "value2"]}
        else:
            query_filters = {f"{field_name}__{operator}": "test_value"}

        where_clause, params = mock_crud_handler._build_where_clause(query_filters)

        # Validated field should appear in WHERE clause
        assert field_name in where_clause

        # Verify appropriate SQL operator
        if operator == "eq":
            assert "=" in where_clause and "!=" not in where_clause
        elif operator == "gt":
            assert ">" in where_clause and ">=" not in where_clause
        elif operator == "gte":
            assert ">=" in where_clause
        elif operator == "lt":
            assert "<" in where_clause and "<=" not in where_clause
        elif operator == "lte":
            assert "<=" in where_clause
        elif operator == "ne":
            assert "!=" in where_clause
        elif operator == "in":
            assert "IN" in where_clause

    def test_eq_operator_validation(self, mock_crud_handler):
        """Test equality operator uses validated field name."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {"status": "processing"}
        )

        assert "status = $1" in where_clause
        assert params == ["processing"]

    def test_gt_operator_validation(self, mock_crud_handler):
        """Test greater than operator uses validated field name."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {"step_order__gt": 5}
        )

        assert "step_order > $1" in where_clause
        assert params == [5]

    def test_gte_operator_validation(self, mock_crud_handler):
        """Test greater than or equal operator uses validated field name."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {"step_order__gte": 10}
        )

        assert "step_order >= $1" in where_clause
        assert params == [10]

    def test_lt_operator_validation(self, mock_crud_handler):
        """Test less than operator uses validated field name."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {"execution_time_ms__lt": 1000}
        )

        assert "execution_time_ms < $1" in where_clause
        assert params == [1000]

    def test_lte_operator_validation(self, mock_crud_handler):
        """Test less than or equal operator uses validated field name."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {"execution_time_ms__lte": 2000}
        )

        assert "execution_time_ms <= $1" in where_clause
        assert params == [2000]

    def test_ne_operator_validation(self, mock_crud_handler):
        """Test not equal operator uses validated field name."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {"status__ne": "failed"}
        )

        assert "status != $1" in where_clause
        assert params == ["failed"]

    def test_in_operator_validation(self, mock_crud_handler):
        """Test IN operator uses validated field name."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {"status__in": ["pending", "processing", "completed"]}
        )

        assert "status IN" in where_clause
        assert "$1" in where_clause and "$2" in where_clause and "$3" in where_clause
        assert params == ["pending", "processing", "completed"]

    @pytest.mark.parametrize(
        "invalid_field",
        [
            "table-name",  # Hyphen
            "field.name",  # Dot
            "field name",  # Space
            "field@name",  # At symbol
            "field#name",  # Hash
            "field$name",  # Dollar sign
            "field%name",  # Percent
            "field&name",  # Ampersand
            "field*name",  # Asterisk
            "field(name",  # Parenthesis
            "field)name",  # Parenthesis
            "field+name",  # Plus
            "field=name",  # Equals
            "field/name",  # Slash
            "field\\name",  # Backslash
            "field|name",  # Pipe
            "field<name",  # Less than
            "field>name",  # Greater than
            "field?name",  # Question mark
            "field!name",  # Exclamation
            "field~name",  # Tilde
            "field^name",  # Caret
            "field`name",  # Backtick
            "field'name",  # Single quote
            'field"name',  # Double quote
        ],
    )
    def test_where_clause_rejects_invalid_characters(
        self, invalid_field, mock_crud_handler
    ):
        """
        Test that field names with invalid characters are rejected.

        Only alphanumeric characters and underscores are allowed in field names.
        """
        with pytest.raises((ValueError, OnexError)):
            mock_crud_handler._build_where_clause({invalid_field: "value"})

    @pytest.mark.parametrize(
        "reserved_keyword",
        [
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "WHERE",
            "FROM",
            "JOIN",
            "UNION",
            "ORDER",
            "GROUP",
            "HAVING",
            "LIMIT",
            "OFFSET",
        ],
    )
    def test_where_clause_rejects_reserved_keywords(
        self, reserved_keyword, mock_crud_handler
    ):
        """
        Test that SQL reserved keywords are rejected as field names.

        Reserved keywords should not be used as field names to prevent
        SQL syntax errors and injection attacks.
        """
        with pytest.raises((ValueError, OnexError)):
            mock_crud_handler._build_where_clause({reserved_keyword: "value"})


class TestWhereClauseComplexFilters:
    """Test suite for complex WHERE clause scenarios."""

    def test_multiple_filters_with_validation(self, mock_crud_handler):
        """Test that multiple filters all use validated field names."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {
                "namespace": "test_app",
                "status": "processing",
                "workflow_type": "metadata_stamping",
            }
        )

        # All validated fields should appear
        assert "namespace" in where_clause
        assert "status" in where_clause
        assert "workflow_type" in where_clause

        # Should use AND to combine conditions
        assert where_clause.count("AND") == 2

        # All parameters should be present
        assert len(params) == 3
        assert "test_app" in params
        assert "processing" in params
        assert "metadata_stamping" in params

    def test_mixed_operators_with_validation(self, mock_crud_handler):
        """Test that mixed operators all use validated field names."""
        where_clause, params = mock_crud_handler._build_where_clause(
            {
                "namespace": "test_app",
                "step_order__gt": 5,
                "status__ne": "failed",
            }
        )

        assert "namespace = $1" in where_clause
        assert "step_order > $2" in where_clause
        assert "status != $3" in where_clause
        assert params == ["test_app", 5, "failed"]

    def test_in_operator_with_malicious_field_rejected(self, mock_crud_handler):
        """Test that IN operator rejects malicious field names."""
        with pytest.raises((ValueError, OnexError)):
            mock_crud_handler._build_where_clause(
                {"status; DROP TABLE--__in": ["value1", "value2"]}
            )

    def test_empty_filters_returns_true_condition(self, mock_crud_handler):
        """Test that empty filters return TRUE condition."""
        where_clause, params = mock_crud_handler._build_where_clause({})

        assert where_clause == "TRUE"
        assert params == []

    def test_none_filters_returns_true_condition(self, mock_crud_handler):
        """Test that None filters return TRUE condition."""
        where_clause, params = mock_crud_handler._build_where_clause(None)

        assert where_clause == "TRUE"
        assert params == []


class TestWhereClauseSQLInjectionVectors:
    """Test suite for SQL injection attack vectors in WHERE clause."""

    @pytest.mark.parametrize(
        "injection_vector",
        [
            # Classic SQL injection
            ("id' OR '1'='1", "value"),
            ("status' OR 1=1--", "value"),
            ("namespace' AND 1=2--", "value"),
            # Comment-based injection
            ("field--", "value"),
            ("field/*", "value"),
            ("field*/", "value"),
            # UNION injection
            ("id' UNION SELECT * FROM users--", "value"),
            ("status' UNION SELECT password FROM credentials--", "value"),
            # Subquery injection
            ("id' AND (SELECT COUNT(*) FROM users)>0--", "value"),
            ("status' OR EXISTS(SELECT * FROM passwords)--", "value"),
            # Time-based blind injection
            ("id'; WAITFOR DELAY '0:0:5'--", "value"),
            ("status'; SELECT pg_sleep(5)--", "value"),
            # Stacked queries
            ("id'; DROP TABLE workflow_executions; --", "value"),
            ("status'; DELETE FROM users WHERE '1'='1", "value"),
        ],
    )
    def test_sql_injection_vectors_rejected(self, injection_vector, mock_crud_handler):
        """
        Test that various SQL injection attack vectors are rejected.

        Comprehensive test of common SQL injection patterns that attackers
        might use to exploit WHERE clause field names.
        """
        field_name, field_value = injection_vector

        with pytest.raises((ValueError, OnexError)):
            mock_crud_handler._build_where_clause({field_name: field_value})


class TestOperatorValidation:
    """Test suite for operator validation in WHERE clause."""

    def test_unsupported_operator_rejected(self, mock_crud_handler):
        """Test that unsupported operators are rejected."""
        with pytest.raises((ValueError, OnexError)) as exc_info:
            mock_crud_handler._build_where_clause({"field__INVALID_OPERATOR": "value"})

        error_msg = str(exc_info.value)
        assert "Unsupported operator" in error_msg or "invalid" in error_msg.lower()

    def test_in_operator_requires_list_value(self, mock_crud_handler):
        """Test that IN operator requires list value, not single value."""
        with pytest.raises((ValueError, OnexError)) as exc_info:
            mock_crud_handler._build_where_clause(
                {"status__in": "not_a_list"}  # Should be list
            )

        error_msg = str(exc_info.value)
        assert "list" in error_msg.lower() or "IN operator" in error_msg

    def test_in_operator_with_empty_list(self, mock_crud_handler):
        """Test IN operator with empty list."""
        where_clause, params = mock_crud_handler._build_where_clause({"status__in": []})

        # Empty IN list should still validate field name
        assert "status" in where_clause
        assert "IN" in where_clause


# Fixtures


@pytest.fixture
def mock_crud_handler():
    """
    Create mock CRUD handler for testing.

    This provides a minimal handler implementation for testing
    WHERE clause building without requiring a database connection.
    """
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0._generic_crud_handlers import (
        GenericCRUDHandlers,
    )

    class MockCRUDHandler(GenericCRUDHandlers):
        """Mock handler for testing WHERE clause building."""

        def __init__(self):
            # No initialization needed for WHERE clause tests
            pass

    return MockCRUDHandler()


if __name__ == "__main__":
    # Run tests with: pytest tests/unit/security/test_where_clause_validation.py -v
    pytest.main([__file__, "-v"])
