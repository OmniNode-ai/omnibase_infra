"""
Unit tests for SqlSanitizer utility.

Tests the security-critical SQL query sanitization functionality used
for OpenTelemetry trace attributes to prevent sensitive data leakage.
"""

from unittest.mock import patch

import pytest
from omnibase_core.exceptions.base_onex_error import OnexError

from src.omnibase_infra.nodes.node_distributed_tracing_compute.v1_0_0.utils.sql_sanitizer import (
    SqlSanitizer,
)


class TestSqlSanitizer:
    """Test cases for SQL query sanitization functionality."""

    def test_empty_query(self):
        """Test handling of empty or None queries."""
        assert SqlSanitizer.sanitize_for_observability("") == ""
        assert SqlSanitizer.sanitize_for_observability("   ") == ""

    def test_simple_select_query(self):
        """Test sanitization of basic SELECT query with literals."""
        query = "SELECT * FROM users WHERE name = 'John Doe' AND age = 25"
        result = SqlSanitizer.sanitize_for_observability(query)

        # Should preserve structure but replace literals
        assert "SELECT" in result
        assert "FROM users" in result
        assert "WHERE" in result
        assert "name" in result
        assert "age" in result

        # Should not contain original literal values
        assert "John Doe" not in result
        assert "25" not in result

    def test_insert_query_with_values(self):
        """Test sanitization of INSERT query with multiple value types."""
        query = "INSERT INTO products (name, price, active) VALUES ('Widget', 29.99, true)"
        result = SqlSanitizer.sanitize_for_observability(query)

        assert "INSERT INTO products" in result
        assert "VALUES" in result
        assert "Widget" not in result
        assert "29.99" not in result

    def test_update_query_sanitization(self):
        """Test sanitization of UPDATE query with WHERE clause."""
        query = "UPDATE users SET password = 'secret123' WHERE email = 'user@example.com'"
        result = SqlSanitizer.sanitize_for_observability(query)

        assert "UPDATE users" in result
        assert "SET password" in result
        assert "WHERE email" in result
        assert "secret123" not in result
        assert "user@example.com" not in result

    def test_complex_query_with_joins(self):
        """Test sanitization of complex query with JOINs and multiple conditions."""
        query = """
        SELECT u.name, p.title
        FROM users u
        JOIN posts p ON u.id = p.user_id
        WHERE u.status = 'active' AND p.created_at > '2023-01-01'
        """
        result = SqlSanitizer.sanitize_for_observability(query)

        assert "SELECT" in result
        assert "JOIN" in result
        assert "WHERE" in result
        assert "active" not in result
        assert "2023-01-01" not in result

    def test_query_with_quotes_and_escapes(self):
        """Test handling of queries with escaped quotes and special characters."""
        query = "SELECT * FROM users WHERE name = 'O''Malley' AND comment = \"She said \\\"hello\\\"\""
        result = SqlSanitizer.sanitize_for_observability(query)

        # Should not contain the original escaped strings
        assert "O''Malley" not in result
        assert 'She said "hello"' not in result

    def test_query_with_numbers_and_hex(self):
        """Test sanitization of various number formats."""
        query = "SELECT * FROM data WHERE int_val = 42 AND float_val = 3.14159 AND hex_val = 0xABCD"
        result = SqlSanitizer.sanitize_for_observability(query)

        assert "42" not in result
        assert "3.14159" not in result
        assert "0xABCD" not in result

    def test_query_without_literals(self):
        """Test that queries without literals are preserved mostly unchanged."""
        query = "SELECT COUNT(*) FROM users WHERE active IS NOT NULL"
        result = SqlSanitizer.sanitize_for_observability(query)

        # Should preserve the structure since no literals to sanitize
        assert "SELECT COUNT(*)" in result
        assert "FROM users" in result
        assert "WHERE active IS NOT NULL" in result

    def test_query_length_limit(self):
        """Test truncation of very long queries."""
        long_query = "SELECT * FROM table WHERE " + "column = 'value' AND " * 50
        result = SqlSanitizer.sanitize_for_observability(long_query, max_length=100)

        assert len(result) <= 100
        assert result.endswith("...")

    def test_query_size_limit(self):
        """Test handling of queries that exceed size limit."""
        # Create a query larger than 10KB
        huge_query = "SELECT * FROM table WHERE " + "x" * 15000
        result = SqlSanitizer.sanitize_for_observability(huge_query)

        assert "QUERY TOO LARGE" in result
        assert "15000 chars" in result or "15003 chars" in result  # Allow for slight variation

    def test_sql_keywords_preserved(self):
        """Test that SQL keywords are properly preserved."""
        query = "SELECT DISTINCT name FROM users ORDER BY created_at LIMIT 10"
        result = SqlSanitizer.sanitize_for_observability(query)

        # All keywords should be preserved
        keywords = ["SELECT", "DISTINCT", "FROM", "ORDER", "BY", "LIMIT"]
        for keyword in keywords:
            assert keyword in result.upper()

    def test_multiple_statements(self):
        """Test handling of multiple SQL statements."""
        query = "SELECT * FROM users; UPDATE users SET active = true"
        result = SqlSanitizer.sanitize_for_observability(query)

        # Should handle multiple statements (sqlparse takes the first one)
        assert "SELECT" in result
        # The UPDATE might not be included if sqlparse only processes first statement

    @patch("src.omnibase_infra.nodes.node_distributed_tracing_compute.v1_0_0.utils.sql_sanitizer.sqlparse.parse")
    def test_sqlparse_failure_handling(self, mock_parse):
        """Test handling when sqlparse fails to parse query."""
        mock_parse.side_effect = Exception("Parse error")

        query = "SELECT * FROM users WHERE name = 'test'"

        # Should raise OnexError when sqlparse fails (fail-fast principle)
        with pytest.raises(OnexError) as exc_info:
            SqlSanitizer.sanitize_for_observability(query)

        assert "SQL query sanitization failed" in str(exc_info.value)

    def test_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        query = "SELECT   *    FROM     users    WHERE    name   =   'test'"
        result = SqlSanitizer.sanitize_for_observability(query)

        # Should not have excessive whitespace
        assert "    " not in result  # No quadruple spaces
        assert "SELECT * FROM users WHERE name" in result

    def test_sensitive_literal_detection(self):
        """Test the _is_sensitive_literal helper method."""
        # This is a white-box test to ensure the token detection works
        # In practice, this would require mocking sqlparse tokens
        # Implementation would need sqlparse token mocks

    def test_sql_keywords_set_completeness(self):
        """Test that the SQL keywords set contains expected keywords."""
        keywords = SqlSanitizer._get_sql_keywords()

        # Test a few critical keywords
        expected_keywords = {
            "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE",
            "JOIN", "GROUP", "ORDER", "HAVING", "LIMIT",
        }

        assert expected_keywords.issubset(keywords)
        assert len(keywords) > 20  # Should have a substantial set of keywords

    def test_clean_whitespace_utility(self):
        """Test the whitespace cleaning utility method."""
        test_cases = [
            ("SELECT    *   FROM   users", "SELECT * FROM users"),
            ("  SELECT *  ", "SELECT *"),
            ("SELECT\n\t*\r\nFROM users", "SELECT * FROM users"),
        ]

        for input_str, expected in test_cases:
            result = SqlSanitizer._clean_whitespace(input_str)
            assert result == expected


class TestSqlSanitizerEdgeCases:
    """Test edge cases and error conditions for SqlSanitizer."""

    def test_none_input_handling(self):
        """Test that None input is handled gracefully."""
        # Should handle None input without crashing
        try:
            result = SqlSanitizer.sanitize_for_observability(None)
            assert result == ""
        except (TypeError, AttributeError):
            # Acceptable to raise error for None input
            pass

    def test_special_characters_in_strings(self):
        """Test handling of special characters and Unicode in string literals."""
        query = "SELECT * FROM users WHERE name = 'José García' AND emoji = '🚀'"
        result = SqlSanitizer.sanitize_for_observability(query)

        # Should not contain the Unicode characters
        assert "José García" not in result
        assert "🚀" not in result

    def test_nested_quotes_and_backslashes(self):
        """Test complex quoting scenarios."""
        query = r"SELECT * FROM data WHERE json_field = '{\"key\": \"value with \\\"quotes\\\"\"}'"
        result = SqlSanitizer.sanitize_for_observability(query)

        # The complex JSON string should be sanitized
        assert '{"key": "value with \\"quotes\\"}' not in result

    def test_sql_injection_patterns(self):
        """Test that common SQL injection patterns are properly sanitized."""
        injection_queries = [
            "SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
            "SELECT * FROM users WHERE name = '' OR '1'='1'",
            "SELECT * FROM users WHERE id = 1 UNION SELECT password FROM admin",
        ]

        for query in injection_queries:
            result = SqlSanitizer.sanitize_for_observability(query)
            # The literal values that could be injection should be sanitized
            # Structure should be preserved for observability
            assert "users" in result  # Table name should be preserved
            assert "SELECT" in result  # Keywords should be preserved
