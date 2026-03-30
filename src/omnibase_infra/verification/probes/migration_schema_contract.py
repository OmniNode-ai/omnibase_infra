# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Migration-aware schema contract validator.

Parses migration SQL files to extract table schemas and validates
that probe SQL queries only reference columns that actually exist
in the migration-derived schema.

This is a CI/test-time tool -- not used at runtime. Probes use
probe_schema_validator.py for runtime introspection.

Parser scope: lightweight DDL grounding for common migration patterns
(CREATE TABLE, ALTER TABLE ADD COLUMN). Not a full SQL parser. Edge
cases (multiline constraints, quoted identifiers, generated columns)
may produce partial results -- the tool returns what it can extract
rather than crashing.
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_columns_from_migration(ddl: str) -> set[str]:
    """Extract column names from CREATE TABLE and ALTER TABLE DDL.

    Handles:
    - CREATE TABLE ... (col type, col type, ...)
    - ALTER TABLE ... ADD COLUMN [IF NOT EXISTS] col type

    Args:
        ddl: Raw SQL DDL text.

    Returns:
        Set of column names found in the DDL.
    """
    columns: set[str] = set()

    # Match CREATE TABLE body -- use greedy match to capture nested parens
    # (e.g., CHECK constraints contain parens). Find the matching closing paren
    # by counting paren depth.
    create_pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+\s*\(",
        re.IGNORECASE,
    )
    for match in create_pattern.finditer(ddl):
        start = match.end()
        depth = 1
        pos = start
        while pos < len(ddl) and depth > 0:
            if ddl[pos] == "(":
                depth += 1
            elif ddl[pos] == ")":
                depth -= 1
            pos += 1
        body = ddl[start : pos - 1]
        for line in body.split(","):
            line = line.strip()
            if not line:
                continue
            # Skip constraints (PRIMARY KEY, UNIQUE, CONSTRAINT, CHECK)
            first_word = line.split()[0].upper() if line.split() else ""
            if first_word in (
                "PRIMARY",
                "UNIQUE",
                "CONSTRAINT",
                "CHECK",
                "FOREIGN",
            ):
                continue
            # First token is column name
            col_name = line.split()[0].strip('"')
            if col_name and not col_name.startswith("--"):
                columns.add(col_name)

    # Match ALTER TABLE ADD COLUMN
    alter_pattern = re.compile(
        r"ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s+",
        re.IGNORECASE,
    )
    for match in alter_pattern.finditer(ddl):
        columns.add(match.group(1))

    return columns


def load_table_schema_from_migrations(
    migration_dir: Path,
    table_name: str,
) -> set[str]:
    """Load complete column set for a table from all migration files.

    Reads all .sql files in the directory, extracts columns, and returns
    the union (representing the current schema after all migrations).

    Args:
        migration_dir: Path to forward migration directory.
        table_name: Table name to filter for.

    Returns:
        Set of all column names for the table.
    """
    columns: set[str] = set()
    for sql_file in sorted(migration_dir.glob("*.sql")):
        content = sql_file.read_text()
        if table_name in content:
            columns |= extract_columns_from_migration(content)
    return columns


def validate_sql_against_schema(
    sql: str,
    schema_columns: set[str],
) -> set[str]:
    """Validate that SQL identifiers reference actual schema columns.

    Extracts identifiers from SELECT, WHERE, and ORDER BY clauses
    and checks them against the known schema.

    Args:
        sql: SQL query string.
        schema_columns: Known valid column names.

    Returns:
        Set of column-like identifiers in the SQL that are not in the schema.
    """
    # SQL keywords and functions to ignore
    sql_keywords = {
        "select",
        "from",
        "where",
        "and",
        "or",
        "not",
        "in",
        "is",
        "null",
        "limit",
        "order",
        "by",
        "asc",
        "desc",
        "as",
        "on",
        "join",
        "left",
        "right",
        "inner",
        "outer",
        "group",
        "having",
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "distinct",
        "case",
        "when",
        "then",
        "else",
        "end",
        "like",
        "between",
        "exists",
        "insert",
        "into",
        "values",
        "update",
        "set",
        "delete",
        "true",
        "false",
        "table",
        "if",
    }

    # Strip string literals (single-quoted values) before extracting tokens
    # to avoid treating values like 'orchestrator' as column references
    stripped_sql = re.sub(r"'[^']*'", "", sql)

    # Extract all word tokens from SQL
    tokens = re.findall(r"\b([a-z_][a-z0-9_]*)\b", stripped_sql.lower())

    # Filter: keep tokens that look like column references
    # (not keywords, not table names, not string literals)
    table_names = {
        "registration_projections",
        "information_schema",
        "columns",
        "public",
    }
    candidate_columns = {
        t
        for t in tokens
        if t not in sql_keywords and t not in table_names and len(t) > 1
    }

    # Return candidates that are not in the known schema
    return candidate_columns - schema_columns


__all__: list[str] = [
    "extract_columns_from_migration",
    "load_table_schema_from_migrations",
    "validate_sql_against_schema",
]
