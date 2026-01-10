# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
Unit tests for SQL injection prevention in projector models.

This test suite validates the SQL security hardening implemented in:
- ModelProjectorColumn: Column name validation, default value restrictions
- ModelProjectorIndex: Index name, column validation, table_name validation
- ModelProjectorSchema: Table name validation, schema_version sanitization
- util_sql_identifiers: Core identifier validation and quoting utilities

Test Organization:
    - TestIdentifierValidation: Validates identifier pattern enforcement
    - TestQuoteIdentifier: Tests identifier quoting for special characters
    - TestEscapeSqlString: Tests SQL string literal escaping
    - TestColumnSqlInjection: Column model SQL injection prevention
    - TestIndexSqlInjection: Index model SQL injection prevention
    - TestSchemaSqlInjection: Schema model SQL injection prevention
    - TestGeneratedSqlSafety: End-to-end SQL generation safety

Coverage Goals:
    - 100% coverage of security-critical validation paths
    - All SQL injection vectors tested and blocked
    - Trust boundary documentation verified

Related Tickets:
    - OMN-1168: ProjectorPluginLoader contract discovery loading
    - PR #138: SQL injection security hardening
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.projectors import (
    ModelProjectorColumn,
    ModelProjectorIndex,
    ModelProjectorSchema,
)
from omnibase_infra.models.projectors.util_sql_identifiers import (
    IDENT_PATTERN,
    escape_sql_string,
    is_valid_identifier,
    quote_identifier,
    validate_identifier,
)


@pytest.mark.unit
class TestIdentifierValidation:
    """Test PostgreSQL identifier validation patterns."""

    @pytest.mark.parametrize(
        "identifier",
        [
            "valid_name",
            "ValidName",
            "_underscore_start",
            "name123",
            "a",
            "A",
            "_",
            "snake_case_name",
            "CamelCase",
        ],
    )
    def test_valid_identifiers_accepted(self, identifier: str) -> None:
        """Test that valid PostgreSQL identifiers pass validation."""
        assert is_valid_identifier(identifier)
        assert IDENT_PATTERN.match(identifier)
        # Should not raise
        result = validate_identifier(identifier)
        assert result == identifier

    @pytest.mark.parametrize(
        ("identifier", "description"),
        [
            ("123start", "starts with digit"),
            ("name-with-dash", "contains dash"),
            ("name.with.dot", "contains dot"),
            ("name with space", "contains space"),
            ("name;drop", "contains semicolon (SQL injection)"),
            ("name'quote", "contains single quote"),
            ('name"quote', "contains double quote"),
            ("name--comment", "contains SQL comment"),
            ("name/*comment*/", "contains SQL block comment"),
            ("", "empty string"),
            ("   ", "whitespace only"),
            ("name\ntab", "contains newline"),
            ("name\ttab", "contains tab"),
        ],
    )
    def test_invalid_identifiers_rejected(
        self, identifier: str, description: str
    ) -> None:
        """Test that invalid identifiers are rejected ({description})."""
        assert not is_valid_identifier(identifier)
        with pytest.raises(ValueError, match="Invalid"):
            validate_identifier(identifier)


@pytest.mark.unit
class TestQuoteIdentifier:
    """Test PostgreSQL identifier quoting."""

    def test_simple_identifier_quoted(self) -> None:
        """Test that simple identifiers are wrapped in double quotes."""
        assert quote_identifier("table_name") == '"table_name"'

    def test_embedded_double_quote_escaped(self) -> None:
        """Test that embedded double quotes are doubled for escape."""
        assert quote_identifier('name"with"quotes') == '"name""with""quotes"'

    def test_single_double_quote(self) -> None:
        """Test a single embedded double quote."""
        assert quote_identifier('test"name') == '"test""name"'

    def test_empty_string_quoted(self) -> None:
        """Test empty string still gets quotes (though invalid identifier)."""
        assert quote_identifier("") == '""'


@pytest.mark.unit
class TestEscapeSqlString:
    """Test SQL string literal escaping."""

    def test_simple_string_unchanged(self) -> None:
        """Test that strings without quotes are unchanged."""
        assert escape_sql_string("simple text") == "simple text"

    def test_single_quote_doubled(self) -> None:
        """Test that single quotes are doubled for SQL escape."""
        assert escape_sql_string("user's name") == "user''s name"

    def test_multiple_quotes_doubled(self) -> None:
        """Test multiple single quotes are all escaped."""
        assert escape_sql_string("it's John's") == "it''s John''s"

    def test_empty_string(self) -> None:
        """Test empty string."""
        assert escape_sql_string("") == ""


@pytest.mark.unit
class TestVarcharLengthValidation:
    """Test varchar column length validation to prevent schema drift."""

    def test_varchar_without_length_rejected(self) -> None:
        """Test varchar columns without length are rejected."""
        with pytest.raises(ValidationError, match="must specify an explicit length"):
            ModelProjectorColumn(
                name="valid_column",
                column_type="varchar",
                nullable=True,
                # Missing length!
            )

    def test_varchar_with_length_accepted(self) -> None:
        """Test varchar columns with length are accepted."""
        column = ModelProjectorColumn(
            name="valid_column",
            column_type="varchar",
            length=255,
            nullable=True,
        )
        assert column.length == 255

    def test_varchar_length_min_boundary(self) -> None:
        """Test minimum varchar length (1) is accepted."""
        column = ModelProjectorColumn(
            name="tiny_col",
            column_type="varchar",
            length=1,
            nullable=True,
        )
        assert column.length == 1

    def test_varchar_length_max_boundary(self) -> None:
        """Test maximum varchar length (10485760 - PostgreSQL max) is accepted."""
        column = ModelProjectorColumn(
            name="huge_col",
            column_type="varchar",
            length=10485760,
            nullable=True,
        )
        assert column.length == 10485760

    def test_varchar_length_exceeds_max_rejected(self) -> None:
        """Test varchar length exceeding PostgreSQL max is rejected."""
        with pytest.raises(ValidationError, match="less than or equal to 10485760"):
            ModelProjectorColumn(
                name="too_huge",
                column_type="varchar",
                length=10485761,
                nullable=True,
            )

    def test_varchar_length_zero_rejected(self) -> None:
        """Test varchar length of 0 is rejected."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            ModelProjectorColumn(
                name="zero_col",
                column_type="varchar",
                length=0,
                nullable=True,
            )

    def test_non_varchar_types_dont_require_length(self) -> None:
        """Test that non-varchar types don't require length."""
        # These should all work without length
        for col_type in ["uuid", "text", "integer", "bigint", "timestamp", "jsonb"]:
            column = ModelProjectorColumn(
                name=f"{col_type}_col",
                column_type=col_type,  # type: ignore[arg-type]
                nullable=True,
            )
            assert column.column_type == col_type


@pytest.mark.unit
class TestColumnSqlInjection:
    """Test SQL injection prevention in ModelProjectorColumn."""

    def test_valid_column_name_accepted(self) -> None:
        """Test valid column names are accepted."""
        column = ModelProjectorColumn(
            name="valid_column",
            column_type="varchar",
            length=255,
            nullable=True,
        )
        assert column.name == "valid_column"

    @pytest.mark.parametrize(
        ("name", "attack_type"),
        [
            ("name; DROP TABLE users;--", "SQL injection via semicolon"),
            ("name' OR '1'='1", "SQL injection via single quote"),
            ('name" OR "1"="1', "SQL injection via double quote"),
            ("name--comment", "SQL comment injection"),
            ("name/**/DROP", "SQL block comment injection"),
            ("name\n; DROP TABLE", "Newline-based injection"),
        ],
    )
    def test_sql_injection_in_column_name_rejected(
        self, name: str, attack_type: str
    ) -> None:
        """Test SQL injection in column name is rejected ({attack_type})."""
        with pytest.raises(ValidationError, match="Invalid column name"):
            ModelProjectorColumn(
                name=name,
                column_type="varchar",
                length=255,
                nullable=True,
            )

    def test_newline_in_default_rejected(self) -> None:
        """Test newline in default value is rejected (prevents multi-statement)."""
        with pytest.raises(ValidationError, match="must not contain line breaks"):
            ModelProjectorColumn(
                name="valid_column",
                column_type="varchar",
                length=255,
                default="'test'\n; DROP TABLE users;",
            )

    def test_carriage_return_in_default_rejected(self) -> None:
        """Test carriage return in default value is rejected."""
        with pytest.raises(ValidationError, match="must not contain line breaks"):
            ModelProjectorColumn(
                name="valid_column",
                column_type="varchar",
                length=255,
                default="'test'\r\n; DROP TABLE",
            )

    def test_newline_in_description_rejected(self) -> None:
        """Test newline in description is rejected."""
        with pytest.raises(ValidationError, match="must not contain line breaks"):
            ModelProjectorColumn(
                name="valid_column",
                column_type="varchar",
                length=255,
                description="Valid start\n-- Injection attempt",
            )

    def test_column_name_quoted_in_sql_output(self) -> None:
        """Test that column name is properly quoted in SQL output."""
        column = ModelProjectorColumn(
            name="user_name",
            column_type="varchar",
            length=255,
            nullable=False,
        )
        sql = column.to_sql_definition()
        assert '"user_name"' in sql
        assert "user_name" not in sql.replace('"user_name"', "")


@pytest.mark.unit
class TestIndexSqlInjection:
    """Test SQL injection prevention in ModelProjectorIndex."""

    def test_valid_index_accepted(self) -> None:
        """Test valid index definition is accepted."""
        index = ModelProjectorIndex(
            name="idx_valid",
            columns=["col1", "col2"],
            index_type="btree",
        )
        assert index.name == "idx_valid"
        assert index.columns == ["col1", "col2"]

    @pytest.mark.parametrize(
        "name",
        [
            "idx; DROP TABLE users;--",
            "idx' OR '1'='1",
            "idx--comment",
        ],
    )
    def test_sql_injection_in_index_name_rejected(self, name: str) -> None:
        """Test SQL injection in index name is rejected."""
        with pytest.raises(ValidationError, match="Invalid index name"):
            ModelProjectorIndex(
                name=name,
                columns=["valid_col"],
            )

    @pytest.mark.parametrize(
        "column",
        [
            "col; DROP TABLE",
            "col'injection",
            "col--comment",
        ],
    )
    def test_sql_injection_in_column_rejected(self, column: str) -> None:
        """Test SQL injection in index column is rejected."""
        with pytest.raises(ValidationError, match="Invalid column name"):
            ModelProjectorIndex(
                name="idx_valid",
                columns=[column],
            )

    def test_to_sql_definition_validates_table_name(self) -> None:
        """Test that to_sql_definition validates table_name parameter."""
        index = ModelProjectorIndex(
            name="idx_valid",
            columns=["col1"],
        )
        with pytest.raises(ValueError, match="Invalid table name"):
            index.to_sql_definition("table; DROP TABLE users;--")

    def test_to_sql_definition_quotes_all_identifiers(self) -> None:
        """Test that all identifiers are quoted in SQL output."""
        index = ModelProjectorIndex(
            name="idx_test",
            columns=["col1", "col2"],
            index_type="btree",
        )
        sql = index.to_sql_definition("test_table")
        assert '"idx_test"' in sql
        assert '"test_table"' in sql
        assert '"col1"' in sql
        assert '"col2"' in sql

    def test_index_with_where_clause(self) -> None:
        """Test index with where_clause (trust boundary - accepts raw SQL)."""
        index = ModelProjectorIndex(
            name="idx_partial",
            columns=["status"],
            where_clause="deleted_at IS NULL",
        )
        sql = index.to_sql_definition("items")
        assert "WHERE deleted_at IS NULL" in sql


@pytest.mark.unit
class TestSchemaNameValidation:
    """Test schema_name validation (only 'public' schema supported)."""

    def test_public_schema_accepted(self) -> None:
        """Test explicit 'public' schema is accepted."""
        schema = ModelProjectorSchema(
            table_name="test_table",
            schema_name="public",
            columns=[
                ModelProjectorColumn(
                    name="id",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                ),
            ],
        )
        assert schema.schema_name == "public"

    def test_default_schema_is_public(self) -> None:
        """Test default schema_name is 'public'."""
        schema = ModelProjectorSchema(
            table_name="test_table",
            columns=[
                ModelProjectorColumn(
                    name="id",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                ),
            ],
        )
        assert schema.schema_name == "public"

    @pytest.mark.parametrize(
        "schema_name",
        [
            "private",
            "custom_schema",
            "PUBLIC",  # Case sensitive
            "Public",
            "myapp",
            "",  # Empty string
        ],
    )
    def test_non_public_schema_rejected(self, schema_name: str) -> None:
        """Test non-public schemas are rejected."""
        with pytest.raises(ValidationError):
            ModelProjectorSchema(
                table_name="test_table",
                schema_name=schema_name,  # type: ignore[arg-type]
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        primary_key=True,
                        nullable=False,
                    ),
                ],
            )


@pytest.mark.unit
class TestSchemaSqlInjection:
    """Test SQL injection prevention in ModelProjectorSchema."""

    def test_valid_schema_accepted(self) -> None:
        """Test valid schema definition is accepted."""
        schema = ModelProjectorSchema(
            table_name="valid_table",
            columns=[
                ModelProjectorColumn(
                    name="id",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                ),
            ],
            schema_version="1.0.0",
        )
        assert schema.table_name == "valid_table"

    @pytest.mark.parametrize(
        "table_name",
        [
            "table; DROP TABLE users;--",
            "table' OR '1'='1",
            "table--comment",
            "table.schema",
        ],
    )
    def test_sql_injection_in_table_name_rejected(self, table_name: str) -> None:
        """Test SQL injection in table name is rejected."""
        with pytest.raises(ValidationError, match="Invalid table name"):
            ModelProjectorSchema(
                table_name=table_name,
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        primary_key=True,
                        nullable=False,
                    ),
                ],
            )

    def test_newline_in_schema_version_rejected(self) -> None:
        """Test newline in schema_version is rejected (comment injection)."""
        with pytest.raises(ValidationError, match="must not contain line breaks"):
            ModelProjectorSchema(
                table_name="valid_table",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        primary_key=True,
                        nullable=False,
                    ),
                ],
                schema_version="1.0.0\n-- DROP TABLE users;",
            )

    def test_invalid_schema_version_format_rejected(self) -> None:
        """Test non-semver schema_version is rejected."""
        with pytest.raises(ValidationError, match="must match semver"):
            ModelProjectorSchema(
                table_name="valid_table",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        primary_key=True,
                        nullable=False,
                    ),
                ],
                schema_version="not-semver",
            )

    def test_newline_in_description_rejected(self) -> None:
        """Test newline in description is rejected."""
        with pytest.raises(ValidationError, match="must not contain line breaks"):
            ModelProjectorSchema(
                table_name="valid_table",
                columns=[
                    ModelProjectorColumn(
                        name="id",
                        column_type="uuid",
                        primary_key=True,
                        nullable=False,
                    ),
                ],
                description="Valid text\n-- Injection",
            )

    def test_table_name_quoted_in_create_table(self) -> None:
        """Test table name is quoted in CREATE TABLE output."""
        schema = ModelProjectorSchema(
            table_name="users",
            columns=[
                ModelProjectorColumn(
                    name="id",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                ),
            ],
        )
        sql = schema.to_create_table_sql()
        assert '"users"' in sql
        assert '"id"' in sql

    def test_description_escaped_in_comment(self) -> None:
        """Test description with quotes is properly escaped."""
        schema = ModelProjectorSchema(
            table_name="users",
            columns=[
                ModelProjectorColumn(
                    name="id",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                ),
            ],
            description="User's data table",
        )
        comments = schema.to_comment_statements_sql()
        assert len(comments) == 1
        # Single quote should be doubled in SQL
        assert "User''s data table" in comments[0]


@pytest.mark.unit
class TestGeneratedSqlSafety:
    """End-to-end tests for SQL generation safety."""

    def test_full_migration_sql_all_identifiers_quoted(self) -> None:
        """Test that full migration SQL quotes all identifiers."""
        schema = ModelProjectorSchema(
            table_name="user_projections",
            columns=[
                ModelProjectorColumn(
                    name="user_id",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                ),
                ModelProjectorColumn(
                    name="display_name",
                    column_type="varchar",
                    length=128,
                    nullable=True,
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="idx_user_name",
                    columns=["display_name"],
                    index_type="btree",
                ),
            ],
            schema_version="1.0.0",
        )
        sql = schema.to_full_migration_sql()

        # Table name should be quoted
        assert '"user_projections"' in sql

        # Column names should be quoted
        assert '"user_id"' in sql
        assert '"display_name"' in sql

        # Index name should be quoted
        assert '"idx_user_name"' in sql

        # Version in comment should be safe (no injection possible)
        assert "version 1.0.0" in sql

    def test_sql_generation_with_all_features(self) -> None:
        """Test SQL generation with descriptions and defaults."""
        schema = ModelProjectorSchema(
            table_name="items",
            description="Item's inventory table",
            columns=[
                ModelProjectorColumn(
                    name="item_id",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                    default="gen_random_uuid()",
                    description="Unique item identifier",
                ),
                ModelProjectorColumn(
                    name="created_at",
                    column_type="timestamptz",
                    nullable=False,
                    default="now()",
                ),
            ],
            schema_version="2.1.0",
        )

        # Create table SQL
        create_sql = schema.to_create_table_sql()
        assert '"items"' in create_sql
        assert '"item_id"' in create_sql
        assert '"created_at"' in create_sql
        assert "gen_random_uuid()" in create_sql
        assert "now()" in create_sql

        # Comment SQL - should escape single quotes
        comments = schema.to_comment_statements_sql()
        assert any("Item''s inventory table" in c for c in comments)

    def test_no_unquoted_identifiers_in_output(self) -> None:
        """Verify no unquoted user-provided identifiers in SQL output."""
        schema = ModelProjectorSchema(
            table_name="test_table",
            columns=[
                ModelProjectorColumn(
                    name="test_column",
                    column_type="uuid",
                    primary_key=True,
                    nullable=False,
                ),
            ],
            indexes=[
                ModelProjectorIndex(
                    name="test_index",
                    columns=["test_column"],
                ),
            ],
        )

        sql = schema.to_full_migration_sql()

        # All identifiers should appear quoted
        # Check that identifiers don't appear unquoted (except in comments)
        lines = sql.split("\n")
        for line in lines:
            if not line.startswith("--"):  # Skip comment lines
                # In non-comment lines, identifiers should be quoted
                if "test_table" in line:
                    assert '"test_table"' in line
                if "test_column" in line:
                    assert '"test_column"' in line
                if "test_index" in line:
                    assert '"test_index"' in line
