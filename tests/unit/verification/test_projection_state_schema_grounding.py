# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Tests for schema-grounded projection state probe [OMN-7040]."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.verify_registration import (
    _check_projection_state_via_db,
)


@pytest.mark.unit
def test_projection_state_validates_schema_before_querying() -> None:
    """Projection state probe must validate schema assumptions first."""
    call_log: list[str] = []

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        call_log.append(sql)
        if "information_schema" in sql:
            return [
                {"column_name": "entity_id"},
                {"column_name": "current_state"},
                {"column_name": "node_type"},
            ]
        return [{"entity_id": "abc", "current_state": "active"}]

    _check_projection_state_via_db("test_contract", db_query_fn)
    # After fix: first call should be schema introspection
    assert any("information_schema" in sql for sql in call_log), (
        "Probe must introspect schema before querying data"
    )


@pytest.mark.unit
def test_projection_state_quarantines_on_missing_column() -> None:
    """When required columns missing, projection probe must QUARANTINE."""

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        if "information_schema" in sql:
            # Schema missing current_state column
            return [
                {"column_name": "entity_id"},
                {"column_name": "node_type"},
            ]
        return []

    result = _check_projection_state_via_db("test_contract", db_query_fn)
    assert result.verdict == EnumValidationVerdict.QUARANTINE
    assert "schema mismatch" in result.evidence.lower()


@pytest.mark.unit
def test_projection_state_quarantines_on_introspection_failure() -> None:
    """Schema introspection failure = QUARANTINE, not silent fallback."""

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        if "information_schema" in sql:
            raise ConnectionError("database unreachable")
        return [{"entity_id": "abc", "current_state": "active"}]

    result = _check_projection_state_via_db("test_contract", db_query_fn)
    assert result.verdict == EnumValidationVerdict.QUARANTINE
    assert "schema introspection failed" in result.evidence.lower()


@pytest.mark.unit
def test_projection_state_passes_when_schema_valid_and_data_present() -> None:
    """When schema valid and active rows exist, probe should PASS."""

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        if "information_schema" in sql:
            return [
                {"column_name": "entity_id"},
                {"column_name": "current_state"},
                {"column_name": "node_type"},
            ]
        return [{"entity_id": "abc", "current_state": "active"}]

    result = _check_projection_state_via_db("test_contract", db_query_fn)
    assert result.verdict == EnumValidationVerdict.PASS
