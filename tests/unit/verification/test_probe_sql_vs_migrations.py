# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""CI contract test: probe SQL must only reference columns in migration DDL.

This test reads all probe source files, extracts SQL query strings via AST,
then validates each SQL against the column set derived from migration files.

Must catch: any reference to `node_name` in registration_projections queries
(the column does not exist -- the bug that started this effort).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from omnibase_infra.verification.probes.migration_schema_contract import (
    load_table_schema_from_migrations,
    validate_sql_against_schema,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "src" / "omnibase_infra" / "verification"
MIGRATION_DIR = REPO_ROOT / "docker" / "migrations" / "forward"


def _extract_sql_strings_from_file(path: Path) -> list[tuple[str, int]]:
    """Extract string literals containing SQL keywords from a Python file.

    Returns:
        List of (sql_string, line_number) tuples.
    """
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sql_strings: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            # Heuristic: string contains SQL DML keywords and references a table
            if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", val, re.IGNORECASE):
                lineno = getattr(node, "lineno", 0)
                sql_strings.append((val, lineno))
    return sql_strings


@pytest.mark.unit
def test_migration_dir_exists_and_has_files() -> None:
    """Sanity check: migration directory must exist and contain SQL files."""
    assert MIGRATION_DIR.exists(), f"Migration dir not found: {MIGRATION_DIR}"
    sql_files = list(MIGRATION_DIR.glob("*.sql"))
    assert len(sql_files) > 0, "No SQL migration files found"


@pytest.mark.unit
def test_schema_extraction_produces_columns() -> None:
    """Sanity check: schema extraction from migrations must find columns."""
    schema = load_table_schema_from_migrations(
        MIGRATION_DIR, "registration_projections"
    )
    assert len(schema) > 0, "Failed to extract any columns from migrations"
    assert "entity_id" in schema, "Expected entity_id in extracted schema"
    assert "current_state" in schema, "Expected current_state in extracted schema"
    assert "node_type" in schema, "Expected node_type in extracted schema"


@pytest.mark.unit
def test_all_probe_sql_references_valid_columns() -> None:
    """Every SQL string in probe code must only reference columns in migration DDL."""
    schema = load_table_schema_from_migrations(
        MIGRATION_DIR, "registration_projections"
    )
    assert len(schema) > 0, "Failed to extract any columns from migrations"

    violations: list[str] = []
    for py_file in PROBE_DIR.rglob("*.py"):
        # Skip test files and __pycache__
        if "__pycache__" in str(py_file):
            continue
        for sql, lineno in _extract_sql_strings_from_file(py_file):
            # Only validate SQL that queries FROM registration_projections,
            # not SQL that merely mentions it as a string value (e.g.,
            # information_schema queries with table_name = 'registration_projections')
            if not re.search(
                r"\bFROM\s+registration_projections\b", sql, re.IGNORECASE
            ):
                continue
            bad_cols = validate_sql_against_schema(sql, schema)
            if bad_cols:
                rel_path = py_file.relative_to(REPO_ROOT)
                violations.append(
                    f"{rel_path}:{lineno}: SQL references nonexistent columns "
                    f"{sorted(bad_cols)}: {sql[:120]}"
                )

    assert not violations, (
        "Probe SQL references columns not in migration DDL:\n" + "\n".join(violations)
    )


@pytest.mark.unit
def test_no_probe_references_node_name_in_registration_sql() -> None:
    """Specific regression guard: no probe may reference `node_name` column.

    The `registration_projections` table has never had a `node_name` column.
    This test catches the exact bug that started the schema grounding effort.
    """
    violations: list[str] = []
    for py_file in PROBE_DIR.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        for sql, lineno in _extract_sql_strings_from_file(py_file):
            if "registration_projections" in sql and "node_name" in sql:
                rel_path = py_file.relative_to(REPO_ROOT)
                violations.append(
                    f"{rel_path}:{lineno}: SQL references nonexistent 'node_name' "
                    f"column in registration_projections query: {sql[:120]}"
                )

    assert not violations, (
        f"Found {len(violations)} reference(s) to nonexistent 'node_name' column:\n"
        + "\n".join(violations)
    )
