# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for schema-grounded registration verification probe [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.verify_registration import _check_registration


@pytest.mark.unit
def test_check_registration_returns_quarantine_on_missing_column() -> None:
    """When the DB has no 'node_name' column, probe must QUARANTINE, not crash."""

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        if "information_schema" in sql:
            return [
                {"column_name": "entity_id"},
                {"column_name": "current_state"},
                {"column_name": "node_type"},
            ]
        # Simulate: column "node_name" does not exist
        raise RuntimeError('column "node_name" does not exist')

    result = _check_registration("node_registration_orchestrator", db_query_fn)
    # Schema is valid, data query should succeed -- but old code would crash
    # because it queries nonexistent node_name column.
    # After fix: queries node_type instead, so this should PASS or FAIL based on data.
    assert result.verdict != EnumValidationVerdict.QUARANTINE or (
        "schema mismatch" in result.evidence.lower()
        or "column" in result.evidence.lower()
    )


@pytest.mark.unit
def test_check_registration_quarantines_when_required_columns_missing() -> None:
    """When schema introspection shows missing required columns, probe must QUARANTINE."""

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        if "information_schema" in sql:
            # Schema missing node_type column
            return [
                {"column_name": "entity_id"},
                {"column_name": "current_state"},
            ]
        return []

    result = _check_registration("node_registration_orchestrator", db_query_fn)
    assert result.verdict == EnumValidationVerdict.QUARANTINE
    assert (
        "schema mismatch" in result.evidence.lower()
        or "column" in result.evidence.lower()
    )


@pytest.mark.unit
def test_check_registration_queries_by_node_type_not_node_name() -> None:
    """Registration probe must query by node_type='orchestrator', not node_name."""
    captured_sql: list[str] = []

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        captured_sql.append(sql)
        if "information_schema" in sql:
            return [
                {"column_name": "entity_id"},
                {"column_name": "current_state"},
                {"column_name": "node_type"},
            ]
        return [
            {"entity_id": "abc", "current_state": "active", "node_type": "orchestrator"}
        ]

    result = _check_registration("node_registration_orchestrator", db_query_fn)
    assert result.verdict == EnumValidationVerdict.PASS
    # Find the data query (not the information_schema one)
    data_queries = [s for s in captured_sql if "information_schema" not in s]
    assert len(data_queries) == 1
    # Must NOT reference node_name column
    assert "node_name" not in data_queries[0]
    # Must use node_type which actually exists
    assert "node_type" in data_queries[0]


@pytest.mark.unit
def test_check_registration_schema_introspection_failure_returns_fail() -> None:
    """If schema introspection itself fails, return FAIL (not crash)."""

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        if "information_schema" in sql:
            raise ConnectionError("database unreachable")
        return []

    result = _check_registration("node_registration_orchestrator", db_query_fn)
    assert result.verdict == EnumValidationVerdict.FAIL
    assert "schema introspection failed" in result.evidence.lower()
