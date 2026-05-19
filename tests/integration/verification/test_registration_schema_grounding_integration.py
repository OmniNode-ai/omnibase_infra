# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for schema-grounded registration verification."""

from __future__ import annotations

import pytest

from omnibase_infra.enums.enum_validation_verdict import EnumValidationVerdict
from omnibase_infra.verification.verify_registration import _check_registration


@pytest.mark.integration
def test_registration_probe_uses_live_projection_schema_shape() -> None:
    """Registration probe must validate schema before querying node state."""
    captured_sql: list[str] = []

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        captured_sql.append(sql)
        if "information_schema.columns" in sql:
            return [
                {"column_name": "entity_id"},
                {"column_name": "domain"},
                {"column_name": "current_state"},
                {"column_name": "node_type"},
                {"column_name": "node_version"},
                {"column_name": "capabilities"},
            ]
        return [
            {
                "entity_id": "registration-orchestrator",
                "current_state": "active",
                "node_type": "orchestrator",
            }
        ]

    result = _check_registration("node_registration_orchestrator", db_query_fn)

    assert result.verdict == EnumValidationVerdict.PASS
    data_queries = [sql for sql in captured_sql if "information_schema" not in sql]
    assert data_queries == [
        "SELECT entity_id, current_state, node_type "
        "FROM registration_projections "
        "WHERE node_type = 'orchestrator' LIMIT 1"
    ]


@pytest.mark.integration
def test_registration_probe_quarantines_when_live_schema_is_not_authoritative() -> None:
    """A missing projection column is quarantine evidence, not an empty PASS."""

    def db_query_fn(sql: str) -> list[dict[str, str]]:
        if "information_schema.columns" in sql:
            return [
                {"column_name": "entity_id"},
                {"column_name": "current_state"},
            ]
        raise AssertionError("data query should not run when schema is incomplete")

    result = _check_registration("node_registration_orchestrator", db_query_fn)

    assert result.verdict == EnumValidationVerdict.QUARANTINE
    assert "schema mismatch" in result.evidence.lower()
