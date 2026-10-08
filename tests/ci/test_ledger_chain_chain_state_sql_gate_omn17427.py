# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: migration 110's SQL-gate exemption rests on the live catalog.

Migration 110 adds ``chain_state`` to ``public.ledger_chain``. The application
SQL gate refuses a ``public.``-qualified application relation unless the exact
path is exempt, and the exemption is justified only if the relation really
lives in ``public`` of the ``omnibase_infra`` service database. The recorded
reply is a read-only catalog query against the .201 dev lane, taken before
migration 110 was applied anywhere: ``ledger_chain`` exists in ``public`` and
in no other schema, and carries no ``chain_state`` column yet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.check_application_database_sql import (
    _is_legacy_default_schema_sql_path,
)

pytestmark = pytest.mark.unit

_MIGRATION = Path("docker/migrations/forward/110_add_ledger_chain_chain_state.sql")
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_contact("tests/ci/fixtures/omn17427_ledger_chain_catalog.json")
def test_migration_110_exemption_matches_the_recorded_catalog(
    recorded_response: dict[str, Any],
) -> None:
    catalog = recorded_response["response"]
    assert catalog["current_database"] == "omnibase_infra"
    assert catalog["ledger_chain_schemas"] == ["public"]
    assert "chain_state" not in catalog["ledger_chain_columns"]

    sql = (_REPO_ROOT / _MIGRATION).read_text(encoding="utf-8")
    statements = [
        line for line in sql.splitlines() if line and not line.startswith("--")
    ]
    targets = set(
        re.findall(r"ALTER TABLE\s+([A-Za-z_.]+)", "\n".join(statements))
    ) | set(re.findall(r"COMMENT ON COLUMN\s+([A-Za-z_]+\.[A-Za-z_]+)\.", sql))
    assert targets == {"public.ledger_chain"}
    assert re.search(
        r"ADD COLUMN IF NOT EXISTS chain_state TEXT NOT NULL DEFAULT ''", sql
    )
    assert _is_legacy_default_schema_sql_path(_MIGRATION)
