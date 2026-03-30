# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
# ruff: noqa: S608
# S608 disabled: inputs are internal constants (table_name, schema_name), not user data.
"""Probe schema validation utility.

Provides reusable schema validation for all verification probes that query
database tables. Probes use this to validate their column assumptions against
the actual schema before executing queries.
"""

from __future__ import annotations


def validate_probe_columns(
    actual_columns: set[str],
    required_columns: set[str],
) -> set[str]:
    """Validate that required columns exist in the actual schema.

    Args:
        actual_columns: Column names from information_schema query.
        required_columns: Columns the probe needs to query.

    Returns:
        Set of missing column names. Empty set if all required columns exist.
    """
    return required_columns - actual_columns


def build_schema_query(table_name: str, schema_name: str = "public") -> str:
    """Build SQL to introspect table columns via information_schema.

    Args:
        table_name: Table to introspect.
        schema_name: Database schema. Defaults to 'public'.

    Returns:
        SQL string that returns column_name rows.
    """
    return (
        "SELECT column_name FROM information_schema.columns "
        f"WHERE table_schema = '{schema_name}' AND table_name = '{table_name}' "
        "ORDER BY ordinal_position"
    )


__all__: list[str] = ["build_schema_query", "validate_probe_columns"]
