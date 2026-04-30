# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for dispatch_eval_results migration contract."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omnibase_infra.models.projection import (
    EnumProjectionOrderingDirection,
    get_projection_ordering_contract,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "intelligence"
    / "023_create_dispatch_eval_results.sql"
)


def _normalized_sql() -> str:
    sql = MIGRATION.read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", sql).lower()


def test_dispatch_eval_results_migration_matches_ordering_contract() -> None:
    sql = _normalized_sql()
    contract = get_projection_ordering_contract("dispatch_eval_results")

    assert contract is not None
    assert contract.projection_table == "dispatch_eval_results"
    assert contract.primary_order_field == "evaluated_at"
    assert contract.tie_breaker_field == "dispatch_id"
    assert contract.direction is EnumProjectionOrderingDirection.DESCENDING
    assert contract.non_authoritative_fields == ("created_at",)

    assert "evaluated_at timestamptz not null" in sql
    assert "dispatch_id text not null" in sql
    assert (
        "create index if not exists idx_dispatch_eval_results_evaluated_at_desc "
        "on dispatch_eval_results (evaluated_at desc)"
    ) in sql
    assert "insertion timestamp only; non-authoritative" in sql
