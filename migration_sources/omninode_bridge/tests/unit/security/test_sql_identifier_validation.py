"""
Comprehensive SQL Injection Tests for Identifier Validation.

Tests that the SQL identifier validation function properly prevents
SQL injection attacks through table names, column names, schema names,
and other database identifiers.

Test Categories:
1. Valid identifier acceptance
2. SQL injection with DROP TABLE statements
3. SQL injection with UNION SELECT attacks
4. SQL injection with comment characters (--)
5. SQL injection with semicolon command chaining
6. SQL injection with quotes and escape characters
7. Invalid characters and patterns
8. Edge cases (empty strings, very long identifiers)
9. Reserved keyword detection
10. Unicode and international character handling

Implementation: Security testing for SQL identifier validation
"""

import pytest

from omninode_bridge.security.validation import InputSanitizer


class TestSQLIdentifierValidation:
    """Comprehensive test suite for SQL identifier validation."""

    # === Valid Identifier Tests ===

    @pytest.mark.parametrize(
        "valid_identifier",
        [
            # Simple valid identifiers
            "users",
            "table_name",
            "column_1",
            "_private",
            "public",
            "schema_name",
            "data_2023",
            "user_id",
            "created_at",
            # Valid with numbers
            "table1",
            "col2",
            "test_123",
            "data_2024_01",
            # Valid with underscores
            "__private_table",
            "internal__data",
            "test__table__name",
            # Mixed case
            "Users",
            "TableName",
            "camelCase",
            "Mixed_Case",
            # Maximum length (63 characters)
            "a" * 63,
            "very_long_table_name_that_is_exactly_63_characters_long_xxxxx",
        ],
    )
    def test_valid_identifiers_accepted(self, valid_identifier):
        """Test that valid SQL identifiers are accepted."""
        result = InputSanitizer.validate_sql_identifier(valid_identifier)
        assert result == valid_identifier

    # === SQL Injection Attack Tests ===

    @pytest.mark.parametrize(
        "malicious_identifier",
        [
            # DROP TABLE attacks
            "users; DROP TABLE users; --",
            "table_name; DROP TABLE important_data; --",
            "'; DROP TABLE users; --",
            "users`; DROP TABLE users; --",
            "users'); DROP TABLE users; --",
            # UNION SELECT attacks
            "users UNION SELECT * FROM passwords",
            "table_name UNION SELECT username,password FROM users",
            "'; UNION SELECT * FROM sensitive_data --",
            "col UNION SELECT * FROM secrets --",
            # Comment-based attacks
            "users--",
            "table_name#",
            "column_name/*",
            "schema_name-- comment",
            "users/**/DROP TABLE",
            # Semicolon command chaining
            "users; SELECT pg_sleep(5)",
            "table_name; DELETE FROM records",
            "column; UPDATE users SET admin='true'",
            "schema; CREATE TABLE backdoor",
            # Nested SQL attacks
            "users') OR 1=1--",
            "table_name' OR '1'='1",
            "column_name' AND 1=1--",
            "schema_name' || 'DROP TABLE",
            # Conditional logic attacks
            "users WHERE 1=1",
            "table_name' AND 1=2--",
            "column_name' OR EXISTS(SELECT * FROM passwords)",
            # Function call attacks
            "users; SELECT version()",
            "table_name; SELECT current_user",
            "column_name; SELECT pg_sleep(10)",
            # Multi-statement attacks
            "users; BEGIN; DROP TABLE users; COMMIT",
            "table_name; START TRANSACTION; DELETE FROM data; COMMIT",
        ],
    )
    def test_sql_injection_attacks_rejected(self, malicious_identifier):
        """Test that SQL injection attempts are rejected."""
        with pytest.raises(
            ValueError,
            match="SQL identifier contains invalid characters|SQL identifier cannot be a reserved keyword",
        ):
            InputSanitizer.validate_sql_identifier(malicious_identifier)

    # === Quote and Escape Character Tests ===

    @pytest.mark.parametrize(
        "escape_identifier",
        [
            # Single quotes
            "'users'",
            "table'name",
            "col'umn",
            "'table_name'",
            # Double quotes
            '"users"',
            'table"name"',
            '"column"',
            # Backticks
            "`users`",
            "table`name",
            "`column_name`",
            # Mixed quotes
            "'`users`'",
            '"`table`"',
            "`'column'`",
            # Escape sequences
            "users\\",
            "table\\name",
            "col\\umn",
            # Null bytes and control characters
            "users\x00",
            "table\x01name",
            "column\x1b",
            # Unicode attempts
            "users\u2028",  # Line separator
            "table\u2029",  # Paragraph separator
            "col\u0000umn",
        ],
    )
    def test_quote_and_escape_characters_rejected(self, escape_identifier):
        """Test that quote and escape characters are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier(escape_identifier)

    # === Invalid Character Tests ===

    @pytest.mark.parametrize(
        "invalid_char_identifier",
        [
            # Spaces and whitespace
            "table name",
            "column name",
            " schema",
            "public ",
            "\ttab",
            "  leading_space",
            "trailing_space  ",
            # Special characters
            "table-name",  # Hyphen not allowed
            "table.name",  # Dot not allowed
            "table@name",  # At symbol not allowed
            "table#name",  # Hash not allowed
            "table$name",  # Dollar sign not allowed
            "table%name",  # Percent not allowed
            "table&name",  # Ampersand not allowed
            "table*name",  # Asterisk not allowed
            "table(name",  # Parenthesis not allowed
            "table)name",  # Parenthesis not allowed
            "table+name",  # Plus not allowed
            "table=name",  # Equals not allowed
            "table/name",  # Slash not allowed
            "table\\name",  # Backslash not allowed
            "table|name",  # Pipe not allowed
            "table<name",  # Less than not allowed
            "table>name",  # Greater than not allowed
            "table?name",  # Question mark not allowed
            "table!name",  # Exclamation not allowed
            "table~name",  # Tilde not allowed
            "table^name",  # Caret not allowed
            "table`name",  # Backtick not allowed
            # SQL operators
            "table+name",
            "table-name",
            "table/name",
            "table=name",
            # Path traversal attempts
            "../table",
            "..\\table",
            ".\\table",
            "./table",
            # URL/URI attempts
            "http://table",
            "ftp://column",
            "file:///schema",
            # JavaScript/HTML attempts
            "<script>",
            "javascript:",
            "onerror=",
            "onclick=",
        ],
    )
    def test_invalid_characters_rejected(self, invalid_char_identifier):
        """Test that identifiers with invalid characters are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier(invalid_char_identifier)

    # === Reserved Keyword Tests ===

    @pytest.mark.parametrize(
        "reserved_keyword",
        [
            # SQL reserved keywords (uppercase)
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "UNION",
            "JOIN",
            "WHERE",
            "HAVING",
            "GROUP",
            "ORDER",
            "LIMIT",
            "OFFSET",
            "OR",
            "AND",
            "NOT",
            "NULL",
            "TRUE",
            "FALSE",
            "INTO",
            "VALUES",
            "SET",
            "FROM",
            "TABLE",
            "VIEW",
            "INDEX",
            "CASCADE",
            "RESTRICT",
            "SHOW",
            "DESCRIBE",
            "EXPLAIN",
            "ANALYZE",
            "VACUUM",
            "REINDEX",
            "CLUSTER",
            "BEGIN",
            "COMMIT",
            "ROLLBACK",
            # SQL reserved keywords (lowercase)
            "select",
            "insert",
            "update",
            "delete",
            "drop",
            "create",
            "alter",
            "truncate",
            "union",
            "join",
            "where",
            "having",
            "group",
            "order",
            "limit",
            "offset",
            "or",
            "and",
            "not",
            "null",
            "true",
            "false",
            # SQL reserved keywords (mixed case)
            "Select",
            "Insert",
            "Update",
            "Delete",
            "Drop",
            "Create",
            "Alter",
            "Where",
            "Having",
            "Group",
            "Order",
            "From",
            "Table",
            # PostgreSQL-specific keywords
            "RETURNING",
            "WITH",
            "RECURSIVE",
            "ILIKE",
            "SIMILAR",
            "REGEX",
            "WINDOW",
            "OVER",
            "PARTITION",
            "RANGE",
            "ROWS",
            "PRECEDING",
            "FOLLOWING",
            "CURRENT",
            "ROW",
            "EXCLUDE",
            "INCLUDE",
            "TIES",
            # Data type keywords
            "INTEGER",
            "VARCHAR",
            "TEXT",
            "BOOLEAN",
            "DATE",
            "TIMESTAMP",
            "UUID",
            "JSON",
            "JSONB",
            "ARRAY",
            "SERIAL",
            "BIGINT",
            "SMALLINT",
            "DECIMAL",
            "NUMERIC",
            "REAL",
            "DOUBLE",
            "PRECISION",
            "INTERVAL",
            # Constraint keywords
            "PRIMARY",
            "FOREIGN",
            "KEY",
            "REFERENCES",
            "UNIQUE",
            "CHECK",
            "CONSTRAINT",
            "DEFAULT",
            "NOT",
            "NULL",
            "AUTO_INCREMENT",
        ],
    )
    def test_reserved_keywords_rejected(self, reserved_keyword):
        """Test that SQL reserved keywords are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier cannot be a reserved keyword"
        ):
            InputSanitizer.validate_sql_identifier(reserved_keyword)

    # === Edge Case Tests ===

    def test_empty_identifier_rejected(self):
        """Test that empty identifiers are rejected."""
        with pytest.raises(ValueError, match="SQL identifier cannot be empty"):
            InputSanitizer.validate_sql_identifier("")

    def test_whitespace_only_identifier_rejected(self):
        """Test that whitespace-only identifiers are rejected."""
        with pytest.raises(ValueError, match="SQL identifier cannot be empty"):
            InputSanitizer.validate_sql_identifier("   ")

    def test_none_identifier_rejected(self):
        """Test that None identifiers are rejected."""
        with pytest.raises(ValueError, match="SQL identifier cannot be None"):
            InputSanitizer.validate_sql_identifier(None)

    def test_non_string_identifier_rejected(self):
        """Test that non-string identifiers are rejected."""
        with pytest.raises(ValueError, match="SQL identifier must be a string"):
            InputSanitizer.validate_sql_identifier(123)

    def test_long_identifier_rejected(self):
        """Test that overly long identifiers are rejected."""
        long_identifier = "a" * 64  # Exceeds 63 character limit
        with pytest.raises(ValueError, match="SQL identifier too long"):
            InputSanitizer.validate_sql_identifier(long_identifier)

    def test_custom_max_length(self):
        """Test custom max length parameter."""
        # Should accept within limit
        result = InputSanitizer.validate_sql_identifier("test", max_length=10)
        assert result == "test"

        # Should reject over limit
        with pytest.raises(ValueError, match="SQL identifier too long"):
            InputSanitizer.validate_sql_identifier("too_long_name", max_length=10)

    def test_numeric_start_rejected(self):
        """Test that identifiers starting with numbers are rejected."""
        with pytest.raises(
            ValueError,
            match="SQL identifier contains invalid characters or invalid start character",
        ):
            InputSanitizer.validate_sql_identifier("1table")
        with pytest.raises(
            ValueError,
            match="SQL identifier contains invalid characters or invalid start character",
        ):
            InputSanitizer.validate_sql_identifier("2_users")

    def test_single_underscore_accepted(self):
        """Test that single underscore is accepted."""
        result = InputSanitizer.validate_sql_identifier("_")
        assert result == "_"

    def test_double_underscore_accepted(self):
        """Test that double underscore is accepted."""
        result = InputSanitizer.validate_sql_identifier("__")
        assert result == "__"

    def test_trailing_underscore_accepted(self):
        """Test that trailing underscore is accepted."""
        result = InputSanitizer.validate_sql_identifier("table_")
        assert result == "table_"

    def test_leading_underscore_accepted(self):
        """Test that leading underscore is accepted."""
        result = InputSanitizer.validate_sql_identifier("_table")
        assert result == "_table"

    def test_sql_keyword_substrings_accepted(self):
        """Test that identifiers containing SQL keywords as substrings are accepted.

        This is intentional - we only reject standalone SQL keywords, not identifiers
        that happen to contain SQL keywords as part of their name. For example:
        - "tableSELECT" is valid (contains "SELECT" but is not "SELECT")
        - "insertDate" is valid (contains "insert" but is not "INSERT")
        """
        valid_identifiers_with_keywords = [
            "tableSELECT",  # Contains SELECT
            "columnWHERE",  # Contains WHERE
            "schemaINSERT",  # Contains INSERT
            "tableUPDATE",  # Contains UPDATE
            "columnDELETE",  # Contains DELETE
            "insertDate",  # Contains insert
            "updateTime",  # Contains update
            "selectAll",  # Contains select
            "deleteFlag",  # Contains delete
            "dropColumn",  # Contains drop
        ]

        for identifier in valid_identifiers_with_keywords:
            result = InputSanitizer.validate_sql_identifier(identifier)
            assert result == identifier

    # === Unicode and International Character Tests ===

    def test_unicode_letters_rejected(self):
        """Test that Unicode letters are rejected (ASCII-only)."""
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier(
                "tαble"  # noqa: RUF001
            )  # Greek alpha
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier("täble")  # German umlaut
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier("tçble")  # Portuguese cedilla

    def test_emoji_rejected(self):
        """Test that emoji characters are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier("table🚀")

    def test_chinese_characters_rejected(self):
        """Test that Chinese characters are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier("表名")  # "table name" in Chinese

    def test_japanese_characters_rejected(self):
        """Test that Japanese characters are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier("テーブル")  # "table" in Japanese

    def test_cyrillic_characters_rejected(self):
        """Test that Cyrillic characters are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier contains invalid characters"
        ):
            InputSanitizer.validate_sql_identifier("таблица")  # "table" in Russian

    # === Case Sensitivity Tests ===

    @pytest.mark.parametrize(
        "case_variant",
        [
            "Users",
            "USERS",
            "users",
            "UsErS",
            "Table_Name",
            "table_name",
            "TABLE_NAME",
        ],
    )
    def test_case_variants_accepted(self, case_variant):
        """Test that various case variants are accepted when valid."""
        result = InputSanitizer.validate_sql_identifier(case_variant)
        assert result == case_variant

    @pytest.mark.parametrize(
        "case_variant",
        [
            "SELECT",
            "Select",
            "select",
            "INSERT",
            "Insert",
            "insert",
        ],
    )
    def test_case_variants_for_keywords_rejected(self, case_variant):
        """Test that case variants of keywords are rejected."""
        with pytest.raises(
            ValueError, match="SQL identifier cannot be a reserved keyword"
        ):
            InputSanitizer.validate_sql_identifier(case_variant)

    # === Complex Attack Pattern Tests ===

    def test_obfuscated_sql_injection_rejected(self):
        """Test that obfuscated SQL injection attempts are rejected."""
        # String concatenation attempts
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("users" + "/*comment*/" + "DROP")

        # Hex encoding attempts (if any)
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("0x7573657273")  # "users" in hex

        # SQL comments with obfuscation
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("users/**/DROP/**/TABLE")

        # Case obfuscation
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("SeLeCt")

        # Whitespace obfuscation
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("SELECT users")

    def test_time_based_blind_injection_rejected(self):
        """Test that time-based blind SQL injection attempts are rejected."""
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("users; WAITFOR DELAY '0:0:5'")
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("table; SELECT pg_sleep(5)")

    def test_boolean_based_blind_injection_rejected(self):
        """Test that boolean-based blind SQL injection attempts are rejected."""
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("users' AND 1=1")
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier("table' OR '1'='1")

    def test_stack_traces_attempts_rejected(self):
        """Test that attempts to cause stack traces are rejected."""
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier(
                "users' AND (SELECT COUNT(*) FROM information_schema.tables)>0"
            )
        with pytest.raises(ValueError):
            InputSanitizer.validate_sql_identifier(
                "table' OR 1 IN (SELECT 1 FROM dual)--"
            )


class TestSQLIdentifierValidationSecurity:
    """Security-focused tests for SQL identifier validation."""

    def test_defense_in_depth(self):
        """Test that multiple layers of validation catch different attack vectors."""
        attack_patterns = [
            "users; DROP TABLE users; --",
            "'; SELECT * FROM users --",
            "users/**/UNION/**/SELECT/**/*",
            "users' OR '1'='1",
            "users/*comment*/DROP",
            "users; WAITFOR DELAY '0:0:5'",
        ]

        for pattern in attack_patterns:
            with pytest.raises(ValueError):
                InputSanitizer.validate_sql_identifier(pattern)

    def test_performance_characteristics(self):
        """Test that validation doesn't introduce performance vulnerabilities."""
        import time

        # Test with various identifier lengths
        test_cases = [
            "short",
            "medium_length_table_name",
            "very_long_table_name_that_is_approaching_the_limit_xxxxxxx",
        ]

        for test_case in test_cases:
            start_time = time.perf_counter()
            InputSanitizer.validate_sql_identifier(test_case)
            elapsed = (time.perf_counter() - start_time) * 1000

            # Validation should be fast (< 1ms)
            assert elapsed < 1.0, f"Validation took {elapsed}ms for '{test_case}'"

    def test_error_message_safety(self):
        """Test that error messages don't expose sensitive information."""
        try:
            InputSanitizer.validate_sql_identifier("users; DROP TABLE users; --")
        except ValueError as e:
            error_message = str(e)
            # Should not reveal internal details
            assert "DROP TABLE" not in error_message
            assert "--" not in error_message
            assert ";" not in error_message
            # Should be generic security message
            assert (
                "invalid characters" in error_message
                or "reserved keyword" in error_message
            )


if __name__ == "__main__":
    # Run tests with: pytest tests/unit/security/test_sql_identifier_validation.py -v
    pytest.main([__file__, "-v"])
