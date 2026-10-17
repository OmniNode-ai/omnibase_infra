# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: the live-events writer's watermark upsert must be granted to role_omnidash.

``BaseProjectionRunner._update_watermark`` upserts
``omninode_internal.projection_watermarks`` for every projection writer. The dev
lane's live-events writer connects as ``role_omnidash`` (it writes
``public.live_events``, which ``omninode_runtime`` cannot be granted), and
``role_omnidash`` held nothing in ``omninode_internal``: every upsert was refused
with ``permission denied for schema omninode_internal``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORWARD = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"
_NODE_DIR = _FORWARD / "nodes" / "node_projection_registration"
_GRANT_FILE = _NODE_DIR / "0008_grant_role_omnidash_projection_watermarks.sql"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_DEV_LANE = _FORWARD.parents[1] / "docker-compose.dev-lane.yml"


def _statements() -> list[str]:
    """The file's SQL statements with ``--`` comments removed."""
    text = "\n".join(
        line
        for line in _GRANT_FILE.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    return [part.strip() for part in text.split(";") if part.strip()]


def _grants() -> list[str]:
    return [s for s in _statements() if s.upper().startswith("GRANT")]


def test_the_migration_file_exists_beside_the_table_it_grants() -> None:
    assert _GRANT_FILE.is_file()
    assert (
        "omninode_internal.projection_watermarks"
        in (_NODE_DIR / "0005_create_projection_watermarks.sql").read_text()
    )


def test_it_grants_schema_usage_and_the_upsert_privileges_to_role_omnidash() -> None:
    normalised = [re.sub(r"\s+", " ", g) for g in _grants()]
    assert normalised == [
        "GRANT USAGE ON SCHEMA omninode_internal TO role_omnidash",
        "GRANT SELECT, INSERT, UPDATE ON omninode_internal.projection_watermarks "
        "TO role_omnidash",
    ]


def test_it_grants_nothing_beyond_the_upsert() -> None:
    sql = " ".join(_statements()).upper()
    for forbidden in ("CREATE", "DELETE", "TRUNCATE", "ALL PRIVILEGES", "WITH GRANT"):
        assert forbidden not in sql, forbidden
    assert all("role_omnidash" in g for g in _grants())


def test_every_grant_is_asserted_after_it_is_issued() -> None:
    assertions = [s for s in _statements() if s.upper().startswith("SELECT 1 /")]
    privileges = {
        match.group(1)
        for s in assertions
        if (match := re.search(r"privilege_type = '(\w+)'", s))
    }
    assert privileges == {"SELECT", "INSERT", "UPDATE"}


def test_the_manifest_row_binds_this_files_exact_bytes() -> None:
    name = _GRANT_FILE.name
    rows = [
        line.split("\t")
        for line in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"nodes/node_projection_registration/{name}\t")
    ]
    assert len(rows) == 1
    _path, owner, source, domain, migration_id, checksum = rows[0]
    assert (owner, source, domain) == (
        "node:node_projection_registration",
        "node:node_projection_registration",
        "omninode_internal",
    )
    assert migration_id == f"node:node_projection_registration:{name}"
    assert checksum == hashlib.sha256(_GRANT_FILE.read_bytes()).hexdigest()


def test_the_live_events_writer_is_the_dev_lane_service_that_needs_it() -> None:
    """The grant exists for a writer that really connects as role_omnidash."""
    compose = _DEV_LANE.read_text(encoding="utf-8")
    block = compose.split("  projection-live-events-writer:\n", 1)[1]
    block = re.split(r"\n  [a-z][a-z-]*:\n", block, maxsplit=1)[0]
    assert "dev_lane_analytics_env" in block
    assert "dev_lane_internal_projection_env" not in block
