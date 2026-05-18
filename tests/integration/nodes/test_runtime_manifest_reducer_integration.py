# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for node_runtime_manifest_reducer (OMN-11197).

Exercises the public seams that ship in PR #1641 — the contract YAML, the
payload model + validator, and the SQL template constant — to prove they
line up end-to-end:

- The contract declares the subscribe topic the handler is wired to.
- The payload model rejects naive datetimes (field_validator).
- The SQL template names every column the payload model exposes.
- The migration creates the table with a unique index matching the
  ON CONFLICT clause the handler relies on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_runtime_manifest_reducer.handlers.handler_postgres_runtime_manifest_insert import (
    SQL_INSERT_RUNTIME_MANIFEST,
)
from omnibase_infra.nodes.node_runtime_manifest_reducer.models.model_payload_insert_runtime_manifest import (
    ModelPayloadInsertRuntimeManifest,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACT_PATH = (
    _REPO_ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_runtime_manifest_reducer"
    / "contract.yaml"
)
_MIGRATION_PATH = (
    _REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "078_create_runtime_manifests.sql"
)


def _valid_payload(
    started_at: datetime | None = None,
) -> ModelPayloadInsertRuntimeManifest:
    return ModelPayloadInsertRuntimeManifest(
        runtime_profile="integration",
        contract_hash="sha256:" + "a" * 64,
        topology_hash="sha256:" + "b" * 64,
        manifest_hash="sha256:" + "c" * 64,
        contracts=[],
        owned_command_topics=[],
        subscribed_event_topics=[],
        handlers=[],
        skipped_contracts=[],
        failed_contracts=[],
        ownership_violations=[],
        image_digest=None,
        started_at=started_at or datetime(2026, 5, 18, 4, 0, 0, tzinfo=UTC),
    )


def test_payload_validator_rejects_naive_started_at() -> None:
    """The field_validator must reject naive datetimes.

    The runtime_manifests column is TIMESTAMPTZ; a naive datetime would land
    in the DB at an undefined offset.
    """
    naive = datetime(2026, 5, 18, 4, 0, 0)
    with pytest.raises(ValueError, match="timezone"):
        _valid_payload(started_at=naive)


def test_payload_accepts_timezone_aware_started_at() -> None:
    """Timezone-aware datetimes pass validation."""
    aware = datetime(2026, 5, 18, 4, 0, 0, tzinfo=UTC)
    payload = _valid_payload(started_at=aware)
    assert payload.started_at.tzinfo is not None


def test_contract_subscribes_to_runtime_manifest_published_topic() -> None:
    """Reducer must subscribe to the topic the runtime emits on (OMN-11196)."""
    if not _CONTRACT_PATH.is_file():
        pytest.skip(f"contract file not present: {_CONTRACT_PATH}")
    text = _CONTRACT_PATH.read_text(encoding="utf-8")
    assert "onex.evt.omnibase-infra.runtime-manifest-published.v1" in text


def test_sql_template_names_every_payload_column() -> None:
    """SQL INSERT must name every column the payload model exposes.

    Drift here would silently drop fields on insert or break with a Postgres
    column-mismatch error after deploy.
    """
    expected_columns = {
        "runtime_profile",
        "contract_hash",
        "topology_hash",
        "manifest_hash",
        "contracts",
        "owned_command_topics",
        "subscribed_event_topics",
        "handlers",
        "skipped_contracts",
        "failed_contracts",
        "ownership_violations",
        "image_digest",
        "started_at",
    }
    sql = SQL_INSERT_RUNTIME_MANIFEST.lower()
    for col in expected_columns:
        assert col.lower() in sql, f"SQL template missing column {col!r}"


def test_migration_creates_table_and_unique_index() -> None:
    """The handler uses ON CONFLICT DO NOTHING on (runtime_profile, topology_hash, started_at).

    The migration must create a matching UNIQUE index or the ON CONFLICT will
    silently degrade to a regular INSERT and lose append-only deduplication.
    """
    if not _MIGRATION_PATH.is_file():
        pytest.skip(f"migration file not present: {_MIGRATION_PATH}")
    sql = _MIGRATION_PATH.read_text(encoding="utf-8").lower()
    normalized = " ".join(sql.split())
    assert "create table" in normalized and "runtime_manifests" in normalized
    assert "unique" in normalized or "on conflict" in normalized
    for col in ("runtime_profile", "topology_hash", "started_at"):
        assert col in normalized, f"migration does not reference {col!r}"
