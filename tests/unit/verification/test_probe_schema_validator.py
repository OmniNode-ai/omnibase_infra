# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for probe schema validation utility [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.verification.probes.probe_schema_validator import (
    build_schema_query,
    validate_probe_columns,
)


@pytest.mark.unit
def test_validate_probe_columns_passes_when_all_exist() -> None:
    """Returns empty set when all required columns exist."""
    actual = {"entity_id", "current_state", "node_type", "capabilities"}
    required = {"entity_id", "current_state"}
    missing = validate_probe_columns(actual, required)
    assert missing == set()


@pytest.mark.unit
def test_validate_probe_columns_returns_missing() -> None:
    """Returns set of columns that don't exist in actual schema."""
    actual = {"entity_id", "current_state"}
    required = {"entity_id", "current_state", "node_name"}
    missing = validate_probe_columns(actual, required)
    assert missing == {"node_name"}


@pytest.mark.unit
def test_validate_probe_columns_empty_actual() -> None:
    """All columns missing when actual is empty."""
    missing = validate_probe_columns(set(), {"entity_id", "current_state"})
    assert missing == {"entity_id", "current_state"}


@pytest.mark.unit
def test_validate_probe_columns_empty_required() -> None:
    """No columns missing when nothing required."""
    missing = validate_probe_columns({"entity_id"}, set())
    assert missing == set()


@pytest.mark.unit
def test_build_schema_query_default_schema() -> None:
    """Build query for public schema."""
    sql = build_schema_query("registration_projections")
    assert "information_schema.columns" in sql
    assert "registration_projections" in sql
    assert "'public'" in sql


@pytest.mark.unit
def test_build_schema_query_custom_schema() -> None:
    """Build query for custom schema."""
    sql = build_schema_query("my_table", schema_name="custom")
    assert "'custom'" in sql
    assert "'my_table'" in sql
