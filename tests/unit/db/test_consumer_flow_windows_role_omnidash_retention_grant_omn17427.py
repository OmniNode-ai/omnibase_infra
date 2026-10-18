# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the consumer_flow_windows retention prune must be granted to role_omnidash.

omnimarket ``node_retention_cleanup`` prunes ``omninode_internal.consumer_flow_windows``
through ``OMNIDASH_ANALYTICS_DB_URL``, which connects as ``role_omnidash``. That
role held nothing on the relation, so every prune was refused and the table grew
past 32 GB on the dev lane, where the projection-api's first-page read of the
consumer-flow exposure timed out and answered 503 to the C28 board probe.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import PRUNE_PRIVILEGES

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_FORWARD = _REPO / "docker" / "migrations" / "forward"
_NODE_DIR = _FORWARD / "nodes" / "node_projection_consumer_flow"
_GRANT_FILE = _NODE_DIR / "0006_grant_role_omnidash_consumer_flow_windows_retention.sql"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _REPO / "config" / "migration_classes.yaml"


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
        "omninode_internal.consumer_flow_windows"
        in (_NODE_DIR / "0000_create_consumer_flow_windows.sql").read_text()
    )


def test_it_grants_schema_usage_and_the_prune_privileges_to_role_omnidash() -> None:
    normalised = [re.sub(r"\s+", " ", g) for g in _grants()]
    assert normalised == [
        "GRANT USAGE ON SCHEMA omninode_internal TO role_omnidash",
        "GRANT SELECT, DELETE ON omninode_internal.consumer_flow_windows "
        "TO role_omnidash",
    ]


def test_it_grants_nothing_beyond_the_prune() -> None:
    sql = " ".join(_statements()).upper()
    for forbidden in (
        "INSERT",
        "UPDATE",
        "TRUNCATE",
        "CREATE",
        "ALL PRIVILEGES",
        "WITH GRANT",
        "REVOKE",
    ):
        assert forbidden not in sql, forbidden
    assert all("role_omnidash" in g for g in _grants())
    # Positive control: the scan above reads the real statements.
    assert "DELETE" in sql


def test_every_grant_is_asserted_after_it_is_issued() -> None:
    assertions = [s for s in _statements() if s.upper().startswith("SELECT 1 /")]
    privileges = {
        match.group(1)
        for s in assertions
        if (match := re.search(r"privilege_type = '(\w+)'", s))
    }
    assert privileges == {"SELECT", "DELETE"}


def test_the_manifest_row_binds_this_files_exact_bytes() -> None:
    name = _GRANT_FILE.name
    rows = [
        line.split("\t")
        for line in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"nodes/node_projection_consumer_flow/{name}\t")
    ]
    assert len(rows) == 1
    _path, owner, source, domain, migration_id, checksum = rows[0]
    assert (owner, source, domain) == (
        "node:node_projection_consumer_flow",
        "node:node_projection_consumer_flow",
        "omninode_internal",
    )
    assert migration_id == f"node:node_projection_consumer_flow:{name}"
    assert checksum == hashlib.sha256(_GRANT_FILE.read_bytes()).hexdigest()


def test_the_migration_is_classed_expand_only() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    key = f"forward/nodes/node_projection_consumer_flow/{_GRANT_FILE.name}"
    assert classes[key] == "expand-only"


def test_the_prune_shape_is_select_and_delete_only() -> None:
    assert (
        frozenset({EnumDatabasePrivilege.SELECT, EnumDatabasePrivilege.DELETE})
        == PRUNE_PRIVILEGES
    )


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_every_profile_declares_the_grant_in_the_prune_shape(profile: str) -> None:
    principal = (
        load_topology_profile(profile)
        .databases["application"]
        .principals["role_omnidash"]
    )
    grants = [
        grant
        for grant in principal.grants
        if "consumer_flow_windows" in grant.objects
        and grant.schema == "omninode_internal"
    ]
    assert len(grants) == 1
    assert frozenset(grants[0].privileges) == PRUNE_PRIVILEGES
