# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static coverage for dispatch_eval_results intelligence migration."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from omnibase_infra.models.projection import (
    EnumProjectionOrderingDirection,
    get_projection_ordering_contract,
)

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


@pytest.mark.unit
def test_dispatch_eval_results_migration_declares_table_shape() -> None:
    sql = _normalized_sql()

    expected_fragments = [
        "create table if not exists dispatch_eval_results",
        "task_id text not null",
        "dispatch_id text not null",
        "ticket_id text",
        "verdict text not null",
        "quality_score numeric(4, 3)",
        "token_cost integer not null default 0",
        "dollars_cost numeric(10, 4) not null default 0",
        "model_calls jsonb not null default '[]'::jsonb",
        "evaluated_at timestamptz not null",
        "eval_latency_ms integer not null",
        "created_at timestamptz not null default now()",
        "usage_source text not null",
        "estimation_method text",
        "source_payload_hash text",
    ]

    for fragment in expected_fragments:
        assert fragment in sql


@pytest.mark.unit
def test_dispatch_eval_results_migration_declares_keys_checks_and_indexes() -> None:
    sql = _normalized_sql()

    expected_fragments = [
        "constraint pk_dispatch_eval_results primary key (task_id, dispatch_id)",
        "constraint uq_dispatch_eval_results_task_dispatch unique (task_id, dispatch_id)",
        "check (verdict in ('pass', 'fail', 'error', 'skipped'))",
        "check (usage_source in ('measured', 'estimated', 'unknown'))",
        "(usage_source = 'estimated' and estimation_method is not null)",
        "(usage_source <> 'estimated' and estimation_method is null)",
        "(usage_source = 'measured' and source_payload_hash is not null)",
        "(usage_source <> 'measured' and source_payload_hash is null)",
        "create index if not exists idx_dispatch_eval_results_ticket_id on dispatch_eval_results (ticket_id)",
        "create index if not exists idx_dispatch_eval_results_evaluated_at_desc on dispatch_eval_results (evaluated_at desc)",
    ]

    for fragment in expected_fragments:
        assert fragment in sql


@pytest.mark.unit
def test_dispatch_eval_results_projection_ordering_contract_registered() -> None:
    contract = get_projection_ordering_contract("dispatch_eval_results")

    assert contract is not None
    assert contract.projection_table == "dispatch_eval_results"
    assert contract.primary_order_field == "evaluated_at"
    assert contract.tie_breaker_field == "dispatch_id"
    assert contract.direction is EnumProjectionOrderingDirection.DESCENDING
    assert contract.non_authoritative_fields == ("created_at",)
