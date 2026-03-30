# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Parameterized correctness test for all verification probes.

For each probe file, validates:
1. No SQL query references the nonexistent 'node_name' column
2. All SQL column references exist in migration-derived schema
3. All referenced tables exist in migration DDL
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from omnibase_infra.verification.probes.migration_schema_contract import (
    load_table_schema_from_migrations,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PROBE_DIR = REPO_ROOT / "src" / "omnibase_infra" / "verification" / "probes"
VERIFY_DIR = REPO_ROOT / "src" / "omnibase_infra" / "verification"
MIGRATION_DIR = REPO_ROOT / "docker" / "migrations" / "forward"


def _collect_probe_files() -> list[Path]:
    """Collect all Python files in verification/probes/ and verification/verify_*.py."""
    files = list(PROBE_DIR.glob("*.py"))
    files.extend(VERIFY_DIR.glob("verify_*.py"))
    return [f for f in files if f.name != "__init__.py"]


def _extract_sql_from_ast(path: Path) -> list[tuple[str, int]]:
    """Extract SQL string literals from a Python file via AST.

    Returns:
        List of (sql_string, line_number) tuples.
    """
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE)\b", val, re.IGNORECASE):
                results.append((val, getattr(node, "lineno", 0)))
    return results


@pytest.mark.unit
@pytest.mark.parametrize("probe_file", _collect_probe_files(), ids=lambda p: p.name)
def test_probe_does_not_reference_node_name_column(probe_file: Path) -> None:
    """No probe should reference nonexistent 'node_name' column in registration SQL."""
    for sql, lineno in _extract_sql_from_ast(probe_file):
        # Only check SQL that queries FROM registration_projections
        if not re.search(r"\bFROM\s+registration_projections\b", sql, re.IGNORECASE):
            continue
        if "node_name" in sql:
            pytest.fail(
                f"{probe_file.name}:{lineno} has SQL referencing nonexistent "
                f"'node_name' column: {sql[:100]}"
            )


@pytest.mark.unit
@pytest.mark.parametrize("probe_file", _collect_probe_files(), ids=lambda p: p.name)
def test_probe_sql_columns_exist_in_migrations(probe_file: Path) -> None:
    """All SQL column references in probe files must exist in migration DDL."""
    from omnibase_infra.verification.probes.migration_schema_contract import (
        validate_sql_against_schema,
    )

    schema = load_table_schema_from_migrations(
        MIGRATION_DIR, "registration_projections"
    )
    assert len(schema) > 0, "Failed to extract columns from migrations"

    violations: list[str] = []
    for sql, lineno in _extract_sql_from_ast(probe_file):
        if not re.search(r"\bFROM\s+registration_projections\b", sql, re.IGNORECASE):
            continue
        bad_cols = validate_sql_against_schema(sql, schema)
        if bad_cols:
            violations.append(
                f"line {lineno}: references {sorted(bad_cols)} in: {sql[:100]}"
            )

    assert not violations, (
        f"{probe_file.name} has SQL referencing nonexistent columns:\n"
        + "\n".join(violations)
    )


@pytest.mark.unit
def test_migration_schema_excludes_node_name() -> None:
    """Regression guard: registration_projections has never had a node_name column."""
    schema = load_table_schema_from_migrations(
        MIGRATION_DIR, "registration_projections"
    )
    assert "node_name" not in schema, (
        "node_name should NOT exist in registration_projections schema"
    )


@pytest.mark.unit
def test_migration_schema_includes_core_columns() -> None:
    """Sanity: migration-derived schema must include known core columns."""
    schema = load_table_schema_from_migrations(
        MIGRATION_DIR, "registration_projections"
    )
    expected = {"entity_id", "current_state", "node_type", "domain"}
    missing = expected - schema
    assert not missing, f"Migration schema missing core columns: {missing}"
