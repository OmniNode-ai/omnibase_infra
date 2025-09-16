"""SQL Query Sanitization for OpenTelemetry Trace Attributes.

Provides secure SQL query sanitization for observability purposes, replacing sensitive
literals with placeholders to prevent data leakage in traces while preserving query structure.
"""

import logging

# Required dependency - fail fast if unavailable (ONEX principle)
import sqlparse
from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from sqlparse.tokens import (
    Keyword,
    Literal,
    Name,
    Number,
)
from sqlparse.tokens import (
    String as StringToken,
)


class SqlSanitizer:
    """
    SQL query sanitizer for OpenTelemetry trace attributes.

    Removes sensitive data from SQL queries for observability while preserving
    query structure for debugging and performance analysis.
    """

    _logger = logging.getLogger(__name__)

    @staticmethod
    def sanitize_for_observability(query: str, max_length: int = 200) -> str:
        """
        Sanitize SQL query for inclusion in trace attributes.

        Replaces literal values (strings, numbers) with placeholder tokens to prevent
        sensitive data from being exposed in traces while preserving query structure
        for observability and debugging.

        Args:
            query: SQL query to sanitize
            max_length: Maximum length of sanitized query (default: 200)

        Returns:
            Sanitized SQL query with literals replaced by placeholders

        Raises:
            OnexError: If query sanitization fails critically
        """
        if not query or not query.strip():
            return ""

        # Validate input length to prevent resource exhaustion
        if len(query) > 10000:  # 10KB limit
            SqlSanitizer._logger.warning(f"Query exceeds size limit: {len(query)} chars")
            return f"-- QUERY TOO LARGE ({len(query)} chars) --"

        try:
            return SqlSanitizer._sanitize_with_sqlparse(query, max_length)

        except Exception as e:
            SqlSanitizer._logger.error(f"Query sanitization failed: {e}")
            # For observability, prefer showing a safe placeholder over failing
            return "-- SANITIZATION_FAILED --"

    @staticmethod
    def _sanitize_with_sqlparse(query: str, max_length: int) -> str:
        """
        Sanitize SQL query using sqlparse library for accurate parsing.

        Args:
            query: SQL query to sanitize
            max_length: Maximum length of result

        Returns:
            Sanitized query with literals replaced by placeholders
        """
        try:
            # Parse the SQL query
            parsed = sqlparse.parse(query)
            if not parsed:
                return "-- UNPARSEABLE_QUERY --"

            # Process the first parsed statement
            statement = parsed[0]
            sanitized_tokens = []

            for token in statement.flatten():
                if SqlSanitizer._is_sensitive_literal(token):
                    # Replace sensitive literals with placeholder
                    sanitized_tokens.append("?")
                elif token.ttype in (Keyword, Name) or token.value.upper() in SqlSanitizer._get_sql_keywords():
                    # Preserve keywords and identifiers (case insensitive)
                    sanitized_tokens.append(token.value)
                elif token.ttype is None and token.value.strip():
                    # Preserve operators, punctuation, and other structural elements
                    sanitized_tokens.append(token.value)

            # Reconstruct and clean up the query
            sanitized = " ".join(sanitized_tokens)
            sanitized = SqlSanitizer._clean_whitespace(sanitized)

            # Truncate if necessary
            if len(sanitized) > max_length:
                sanitized = sanitized[:max_length - 3] + "..."

            return sanitized

        except Exception as e:
            SqlSanitizer._logger.error(f"sqlparse sanitization failed: {e}")
            raise OnexError(
                message=f"SQL query sanitization failed: {e!s}",
                error_code=CoreErrorCode.PROCESSING_ERROR,
            ) from e


    @staticmethod
    def _is_sensitive_literal(token) -> bool:
        """
        Determine if a token contains sensitive literal data.

        Args:
            token: sqlparse token to evaluate

        Returns:
            True if token contains sensitive data that should be sanitized
        """
        # Check for various literal types that should be sanitized
        sensitive_types = [
            Literal.String.Single,      # 'string'
            Literal.String.Symbol,      # "string"
            Literal.Number.Integer,     # 123
            Literal.Number.Float,       # 123.45
            Literal.Number.Hexadecimal, # 0xABC
            Number.Integer,             # Alternative number tokens
            Number.Float,
            Number.Hexadecimal,
            StringToken.Single,         # Alternative string tokens
            StringToken.Symbol,
        ]

        return any(token.ttype == sensitive_type for sensitive_type in sensitive_types)

    @staticmethod
    def _get_sql_keywords() -> set:
        """
        Get common SQL keywords that should be preserved.

        Returns:
            Set of SQL keywords to preserve in sanitized queries
        """
        return {
            "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP",
            "ALTER", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "GROUP", "ORDER",
            "BY", "HAVING", "LIMIT", "OFFSET", "UNION", "INTERSECT", "EXCEPT", "AS",
            "DISTINCT", "ALL", "AND", "OR", "NOT", "IN", "EXISTS", "BETWEEN", "LIKE",
            "IS", "NULL", "TRUE", "FALSE", "CASE", "WHEN", "THEN", "ELSE", "END",
            "IF", "COALESCE", "NULLIF", "CAST", "CONVERT", "COUNT", "SUM", "AVG",
            "MIN", "MAX", "FIRST", "LAST", "TOP", "INTO", "VALUES", "SET", "TABLE",
            "INDEX", "VIEW", "PROCEDURE", "FUNCTION", "TRIGGER", "DATABASE", "SCHEMA",
            "GRANT", "REVOKE", "COMMIT", "ROLLBACK", "BEGIN", "TRANSACTION",
        }

    @staticmethod
    def _clean_whitespace(query: str) -> str:
        """
        Clean up excessive whitespace in sanitized queries.

        Args:
            query: Query with potential whitespace issues

        Returns:
            Query with normalized whitespace
        """
        import re
        # Replace multiple whitespace with single space
        cleaned = re.sub(r"\s+", " ", query.strip())
        return cleaned

