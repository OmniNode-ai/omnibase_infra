"""
Comprehensive LIMIT/OFFSET Validation Tests.

Tests that LIMIT and OFFSET parameters are properly validated
to prevent SQL injection attacks and resource exhaustion.

Test Categories:
1. Negative values rejection
2. Zero values handling
3. MAX_LIMIT boundary testing (999, 1000, 1001)
4. MAX_OFFSET boundary testing (9999, 10000, 10001)
5. Non-integer values rejection
6. SQL injection attempts via LIMIT/OFFSET
7. Type safety validation
8. Edge cases (None, empty, malformed)

Security Fix: LIMIT/OFFSET validation to prevent SQL injection
Implementation: Agent 5
"""

import pytest

# Import OnexError from the actual implementation
# This ensures we catch the same exception type being raised
from omnibase_core import EnumCoreErrorCode, ModelOnexError

# Aliases for compatibility
OnexError = ModelOnexError
CoreErrorCode = EnumCoreErrorCode


class TestLimitValidation:
    """Test suite for LIMIT parameter validation."""

    @pytest.mark.parametrize(
        "limit_value",
        [
            -1,
            -10,
            -100,
            -999,
            -1000,
            -10000,
        ],
    )
    def test_limit_validation_rejects_negative_values(
        self, limit_value, mock_crud_handler
    ):
        """
        Test that negative LIMIT values are rejected.

        Negative LIMIT values could be exploited for SQL injection
        or cause unexpected database behavior.
        """
        # Test directly at _build_select_query level
        # (Pydantic model already validates >= 0)
        with pytest.raises(OnexError) as exc_info:
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                limit=limit_value,
            )

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        assert "LIMIT must be a non-negative integer" in exc_info.value.message
        assert "limit" in exc_info.value.context.get("additional_context", {}).get(
            "context", {}
        )

    def test_limit_validation_accepts_zero(self, mock_crud_handler):
        """
        Test that LIMIT 0 is accepted.

        LIMIT 0 is valid SQL and can be used for schema inspection
        without returning any rows.
        """
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=0,
        )

        assert "LIMIT 0" in query
        assert params == ["test"]

    @pytest.mark.parametrize(
        "limit_value",
        [
            1,
            10,
            100,
            500,
            999,  # Just under MAX_LIMIT
            1000,  # Exactly MAX_LIMIT
        ],
    )
    def test_limit_validation_accepts_valid_values(
        self, limit_value, mock_crud_handler
    ):
        """
        Test that valid LIMIT values are accepted.

        Valid values are non-negative integers up to and including MAX_LIMIT (1000).
        """
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=limit_value,
        )

        assert f"LIMIT {limit_value}" in query
        assert params == ["test"]

    @pytest.mark.parametrize(
        "limit_value",
        [
            1001,  # Just over MAX_LIMIT
            1500,
            2000,
            5000,
            10000,
            999999,
        ],
    )
    def test_limit_validation_rejects_excessive_values(
        self, limit_value, mock_crud_handler
    ):
        """
        Test that LIMIT values exceeding MAX_LIMIT (1000) are rejected.

        This prevents resource exhaustion attacks where an attacker
        requests extremely large result sets.
        """
        # Test directly at _build_select_query level
        with pytest.raises(OnexError) as exc_info:
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                limit=limit_value,
            )

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        assert (
            f"LIMIT exceeds maximum allowed value of {mock_crud_handler.MAX_LIMIT}"
            in exc_info.value.message
        )
        additional_context = exc_info.value.context.get("additional_context", {}).get(
            "context", {}
        )
        assert "limit" in additional_context
        assert additional_context["limit"] == limit_value
        assert additional_context["max_limit"] == 1000

    @pytest.mark.parametrize(
        "limit_value",
        [
            "10",  # String
            "1000",  # String number
            3.14,  # Float
            10.5,  # Float
            True,  # Boolean
            False,  # Boolean
            [],  # List
            {},  # Dict
            # None is handled separately by test_limit_none_value_is_accepted
        ],
    )
    def test_limit_validation_rejects_non_integer_types(
        self, limit_value, mock_crud_handler
    ):
        """
        Test that non-integer LIMIT values are rejected.

        Only integer types should be accepted to prevent type confusion
        and SQL injection attempts.
        """
        with pytest.raises(OnexError) as exc_info:
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                limit=limit_value,
            )

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        assert "LIMIT must be a non-negative integer" in exc_info.value.message

    @pytest.mark.parametrize(
        "malicious_limit",
        [
            "10; DROP TABLE workflow_executions; --",
            "1000 OR 1=1",
            "10 UNION SELECT * FROM users",
            "10' OR '1'='1",
            "10--",
            "10/*",
            "10; SELECT pg_sleep(5); --",
        ],
    )
    def test_limit_sql_injection_prevention(self, malicious_limit, mock_crud_handler):
        """
        Test that SQL injection attempts via LIMIT are rejected.

        LIMIT parameter must be validated before query building
        to prevent SQL injection attacks.
        """
        with pytest.raises((OnexError, ValueError, TypeError)):
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                limit=malicious_limit,
            )

    def test_limit_none_value_is_accepted(self, mock_crud_handler):
        """
        Test that LIMIT None (no limit) is accepted.

        None is a valid value indicating no LIMIT clause should be added.
        """
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=None,
        )

        # Should not contain LIMIT clause when None
        assert "LIMIT" not in query or ("LIMIT" in query and query.count("LIMIT") == 0)


class TestOffsetValidation:
    """Test suite for OFFSET parameter validation."""

    @pytest.mark.parametrize(
        "offset_value",
        [
            -1,
            -10,
            -100,
            -999,
            -1000,
            -10000,
        ],
    )
    def test_offset_validation_rejects_negative_values(
        self, offset_value, mock_crud_handler
    ):
        """
        Test that negative OFFSET values are rejected.

        Negative OFFSET values are invalid SQL and could be exploited
        for SQL injection.
        """
        # Test directly at _build_select_query level
        with pytest.raises(OnexError) as exc_info:
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                offset=offset_value,
            )

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        assert "OFFSET must be a non-negative integer" in exc_info.value.message
        assert "offset" in exc_info.value.context.get("additional_context", {}).get(
            "context", {}
        )

    def test_offset_validation_accepts_zero(self, mock_crud_handler):
        """
        Test that OFFSET 0 is accepted but not added to query.

        OFFSET 0 is semantically equivalent to no OFFSET, so it should
        be accepted but the OFFSET clause should be omitted.
        """
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            offset=0,
        )

        # OFFSET 0 should not add OFFSET clause (optimization)
        assert "OFFSET" not in query or "OFFSET 0" not in query

    @pytest.mark.parametrize(
        "offset_value",
        [
            1,
            10,
            100,
            500,
            1000,
            5000,
            9999,  # Just under MAX_OFFSET
            10000,  # Exactly MAX_OFFSET
        ],
    )
    def test_offset_validation_accepts_valid_values(
        self, offset_value, mock_crud_handler
    ):
        """
        Test that valid OFFSET values are accepted.

        Valid values are positive integers up to and including MAX_OFFSET (10000).
        """
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            offset=offset_value,
        )

        assert f"OFFSET {offset_value}" in query
        assert params == ["test"]

    @pytest.mark.parametrize(
        "offset_value",
        [
            10001,  # Just over MAX_OFFSET
            15000,
            20000,
            50000,
            100000,
            999999,
        ],
    )
    def test_offset_validation_rejects_excessive_values(
        self, offset_value, mock_crud_handler
    ):
        """
        Test that OFFSET values exceeding MAX_OFFSET (10000) are rejected.

        This prevents resource exhaustion attacks and performance degradation
        from extremely large offsets.
        """
        # Test directly at _build_select_query level
        with pytest.raises(OnexError) as exc_info:
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                offset=offset_value,
            )

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        assert (
            f"OFFSET exceeds maximum allowed value of {mock_crud_handler.MAX_OFFSET}"
            in exc_info.value.message
        )
        additional_context = exc_info.value.context.get("additional_context", {}).get(
            "context", {}
        )
        assert "offset" in additional_context
        assert additional_context["offset"] == offset_value
        assert additional_context["max_offset"] == 10000

    @pytest.mark.parametrize(
        "offset_value",
        [
            "10",  # String
            "10000",  # String number
            3.14,  # Float
            100.5,  # Float
            True,  # Boolean
            False,  # Boolean
            [],  # List
            {},  # Dict
            # None is handled separately by test_offset_none_value_is_accepted
        ],
    )
    def test_offset_validation_rejects_non_integer_types(
        self, offset_value, mock_crud_handler
    ):
        """
        Test that non-integer OFFSET values are rejected.

        Only integer types should be accepted to prevent type confusion
        and SQL injection attempts.
        """
        with pytest.raises(OnexError) as exc_info:
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                offset=offset_value,
            )

        assert exc_info.value.error_code == CoreErrorCode.VALIDATION_ERROR
        assert "OFFSET must be a non-negative integer" in exc_info.value.message

    @pytest.mark.parametrize(
        "malicious_offset",
        [
            "10; DROP TABLE workflow_executions; --",
            "1000 OR 1=1",
            "10 UNION SELECT * FROM users",
            "10' OR '1'='1",
            "10--",
            "10/*",
            "10; SELECT pg_sleep(5); --",
        ],
    )
    def test_offset_sql_injection_prevention(self, malicious_offset, mock_crud_handler):
        """
        Test that SQL injection attempts via OFFSET are rejected.

        OFFSET parameter must be validated before query building
        to prevent SQL injection attacks.
        """
        with pytest.raises((OnexError, ValueError, TypeError)):
            mock_crud_handler._build_select_query(
                table_name="workflow_executions",
                query_filters={"namespace": "test"},
                offset=malicious_offset,
            )

    def test_offset_none_value_is_accepted(self, mock_crud_handler):
        """
        Test that OFFSET None (no offset) is accepted.

        None is a valid value indicating no OFFSET clause should be added.
        """
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            offset=None,
        )

        # Should not contain OFFSET clause when None
        assert "OFFSET" not in query or (
            "OFFSET" in query and query.count("OFFSET") == 0
        )


class TestLimitOffsetCombinations:
    """Test suite for combined LIMIT/OFFSET validation."""

    def test_both_limit_and_offset_valid(self, mock_crud_handler):
        """Test that valid LIMIT and OFFSET can be used together."""
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=100,
            offset=50,
        )

        assert "LIMIT 100" in query
        assert "OFFSET 50" in query
        assert params == ["test"]

    def test_limit_without_offset(self, mock_crud_handler):
        """Test that LIMIT can be used without OFFSET."""
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=100,
            offset=None,
        )

        assert "LIMIT 100" in query
        assert "OFFSET" not in query

    def test_offset_without_limit(self, mock_crud_handler):
        """Test that OFFSET can be used without LIMIT."""
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=None,
            offset=50,
        )

        assert "OFFSET 50" in query
        # LIMIT should be omitted or handled by database default

    def test_boundary_limit_1000_with_offset(self, mock_crud_handler):
        """Test MAX_LIMIT (1000) with valid OFFSET."""
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=1000,
            offset=5000,
        )

        assert "LIMIT 1000" in query
        assert "OFFSET 5000" in query

    def test_boundary_offset_10000_with_limit(self, mock_crud_handler):
        """Test MAX_OFFSET (10000) with valid LIMIT."""
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=100,
            offset=10000,
        )

        assert "LIMIT 100" in query
        assert "OFFSET 10000" in query

    def test_both_at_maximum_boundaries(self, mock_crud_handler):
        """Test both LIMIT and OFFSET at their maximum allowed values."""
        query, params = mock_crud_handler._build_select_query(
            table_name="workflow_executions",
            query_filters={"namespace": "test"},
            limit=1000,
            offset=10000,
        )

        assert "LIMIT 1000" in query
        assert "OFFSET 10000" in query


# Fixtures


@pytest.fixture
def mock_crud_handler():
    """
    Create mock CRUD handler for testing.

    This provides a minimal handler implementation for testing
    query building without requiring a database connection.
    """
    from omninode_bridge.nodes.database_adapter_effect.v1_0_0._generic_crud_handlers import (
        GenericCRUDHandlers,
    )

    class MockCRUDHandler(GenericCRUDHandlers):
        """Mock handler for testing query building."""

        def __init__(self):
            # No initialization needed for query building tests
            pass

    return MockCRUDHandler()


if __name__ == "__main__":
    # Run tests with: pytest tests/unit/security/test_limit_offset_validation.py -v
    pytest.main([__file__, "-v"])
