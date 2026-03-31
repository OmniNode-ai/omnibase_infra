# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for migration-aware schema validation.

Validates that probe SQL queries only reference columns that exist
in the actual migration DDL.
"""

from __future__ import annotations

import pytest

from omnibase_infra.verification.probes.migration_schema_contract import (
    extract_columns_from_migration,
    validate_sql_against_schema,
)


@pytest.mark.unit
def test_extract_columns_from_create_table() -> None:
    """Parse CREATE TABLE to extract column names."""
    ddl = """
    CREATE TABLE IF NOT EXISTS registration_projections (
        entity_id UUID NOT NULL,
        domain VARCHAR(128) NOT NULL DEFAULT 'registration',
        current_state registration_state NOT NULL,
        node_type VARCHAR(64) NOT NULL
    );
    """
    columns = extract_columns_from_migration(ddl)
    assert "entity_id" in columns
    assert "domain" in columns
    assert "current_state" in columns
    assert "node_type" in columns
    assert "node_name" not in columns


@pytest.mark.unit
def test_extract_columns_from_alter_table_add() -> None:
    """Parse ALTER TABLE ADD COLUMN to extract new columns."""
    ddl = """
    ALTER TABLE registration_projections
        ADD COLUMN IF NOT EXISTS contract_type TEXT,
        ADD COLUMN IF NOT EXISTS intent_types TEXT[] DEFAULT ARRAY[]::TEXT[];
    """
    columns = extract_columns_from_migration(ddl)
    assert "contract_type" in columns
    assert "intent_types" in columns


@pytest.mark.unit
def test_extract_columns_skips_constraints() -> None:
    """Constraint lines must not be extracted as column names."""
    ddl = """
    CREATE TABLE IF NOT EXISTS example (
        id UUID NOT NULL,
        name TEXT,
        PRIMARY KEY (id),
        UNIQUE (name),
        CONSTRAINT valid_name CHECK (name IS NOT NULL)
    );
    """
    columns = extract_columns_from_migration(ddl)
    assert "id" in columns
    assert "name" in columns
    assert "PRIMARY" not in columns
    assert "UNIQUE" not in columns
    assert "CONSTRAINT" not in columns


@pytest.mark.unit
def test_extract_columns_from_real_migration_001() -> None:
    """Verify extraction from a realistic registration_projections DDL."""
    ddl = """
    CREATE TABLE IF NOT EXISTS registration_projections (
        entity_id UUID NOT NULL,
        domain VARCHAR(128) NOT NULL DEFAULT 'registration',
        current_state registration_state NOT NULL,
        node_type VARCHAR(64) NOT NULL,
        node_version VARCHAR(32) NOT NULL DEFAULT '1.0.0',
        capabilities JSONB NOT NULL DEFAULT '{}',
        ack_deadline TIMESTAMPTZ,
        liveness_deadline TIMESTAMPTZ,
        last_heartbeat_at TIMESTAMPTZ,
        ack_timeout_emitted_at TIMESTAMPTZ,
        liveness_timeout_emitted_at TIMESTAMPTZ,
        last_applied_event_id UUID NOT NULL,
        last_applied_offset BIGINT NOT NULL DEFAULT 0,
        last_applied_sequence BIGINT,
        last_applied_partition VARCHAR(128),
        registered_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        correlation_id UUID,
        PRIMARY KEY (entity_id, domain),
        UNIQUE (entity_id),
        CONSTRAINT valid_offset CHECK (last_applied_offset >= 0),
        CONSTRAINT valid_sequence CHECK (last_applied_sequence IS NULL OR last_applied_sequence >= 0),
        CONSTRAINT valid_node_type CHECK (node_type IN ('effect', 'compute', 'reducer', 'orchestrator'))
    );
    """
    columns = extract_columns_from_migration(ddl)
    expected = {
        "entity_id",
        "domain",
        "current_state",
        "node_type",
        "node_version",
        "capabilities",
        "ack_deadline",
        "liveness_deadline",
        "last_heartbeat_at",
        "ack_timeout_emitted_at",
        "liveness_timeout_emitted_at",
        "last_applied_event_id",
        "last_applied_offset",
        "last_applied_sequence",
        "last_applied_partition",
        "registered_at",
        "updated_at",
        "correlation_id",
    }
    assert expected.issubset(columns), f"Missing columns: {expected - columns}"
    assert "node_name" not in columns


@pytest.mark.unit
def test_validate_sql_catches_nonexistent_column() -> None:
    """SQL referencing a column not in schema must be flagged."""
    schema_columns = {"entity_id", "current_state", "node_type"}
    sql = "SELECT node_name, current_state FROM registration_projections WHERE node_name = 'foo'"
    violations = validate_sql_against_schema(sql, schema_columns)
    assert "node_name" in violations


@pytest.mark.unit
def test_validate_sql_passes_for_valid_columns() -> None:
    """SQL referencing only valid columns must pass."""
    schema_columns = {"entity_id", "current_state", "node_type"}
    sql = "SELECT entity_id, current_state FROM registration_projections WHERE node_type = 'orchestrator'"
    violations = validate_sql_against_schema(sql, schema_columns)
    assert len(violations) == 0


@pytest.mark.unit
def test_validate_sql_ignores_sql_keywords() -> None:
    """SQL keywords and functions must not be flagged as invalid columns."""
    schema_columns = {"entity_id", "current_state"}
    sql = "SELECT entity_id, current_state FROM registration_projections WHERE current_state IS NOT NULL LIMIT 1"
    violations = validate_sql_against_schema(sql, schema_columns)
    assert len(violations) == 0


@pytest.mark.unit
def test_validate_sql_ignores_string_values() -> None:
    """String literal values inside SQL must not be flagged as columns."""
    schema_columns = {"entity_id", "current_state", "node_type"}
    sql = "SELECT entity_id FROM registration_projections WHERE node_type = 'orchestrator'"
    violations = validate_sql_against_schema(sql, schema_columns)
    assert "orchestrator" not in violations
    assert len(violations) == 0
