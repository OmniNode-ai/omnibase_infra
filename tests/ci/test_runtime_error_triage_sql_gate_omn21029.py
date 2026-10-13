# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21029: migration 111's SQL-gate exemptions rest on the live catalog.

Migration 111 and its rollback swap the active-fingerprint unique index on
``public.runtime_error_triage``. The application SQL gate refuses a
``public.``-qualified application relation unless the exact path is exempt,
and the exemption is justified only if the relation really lives in ``public``
of the ``omnibase_infra`` service database. The recorded reply is a read-only
catalog query against the dev lane, taken before migration 111 was applied
anywhere: ``runtime_error_triage`` exists only in ``public`` and carries the
original active-fingerprint unique index, with no v2 index yet.
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

_MIGRATION = Path(
    "docker/migrations/forward/"
    "111_widen_runtime_error_triage_active_unique_to_escalated.sql"
)
_ROLLBACK = Path(
    "docker/migrations/rollback/"
    "rollback_111_widen_runtime_error_triage_active_unique_to_escalated.sql"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live_contact(
    "tests/ci/fixtures/omn21029_runtime_error_triage_catalog.json"
)
def test_migration_111_exemption_matches_the_recorded_catalog(
    recorded_response: dict[str, Any],
) -> None:
    catalog = recorded_response["response"]
    assert catalog["current_database"] == "omnibase_infra"
    assert catalog["runtime_error_triage_schemas"] == ["public"]
    assert (
        "uq_runtime_error_triage_active_fingerprint"
        in catalog["runtime_error_triage_indexes"]
    )
    assert (
        "uq_runtime_error_triage_active_fingerprint_v2"
        not in catalog["runtime_error_triage_indexes"]
    )

    for path in (_MIGRATION, _ROLLBACK):
        sql = (_REPO_ROOT / path).read_text(encoding="utf-8")
        targets = set(re.findall(r"\bON\s+([A-Za-z_]+\.[A-Za-z_]+)\s*\(", sql))
        assert targets == {"public.runtime_error_triage"}
        for target in re.findall(
            r"DROP INDEX CONCURRENTLY IF EXISTS\s+([A-Za-z_.]+)", sql
        ):
            assert target.startswith("public.")
        assert _is_legacy_default_schema_sql_path(path) is True
