# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: the lab job check store's three omninode_internal relations are granted to role_omnidash.

omnimarket ``node_lab_job_check_effect`` reads ``omninode_internal.lab_job_state``,
``work_ledger_rows`` and ``pr_state`` (and ``public.hook_events``) through the
shared pool the dev lane's ``omninode-runtime-effects`` binds, which connects as
``role_omnidash``. That role held SELECT on none of the three, so the dev lane's
Postgres log read 200 ``permission denied for table lab_job_state`` lines in 24
hours. Each relation is granted in its owning node's lineage, SELECT only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import READ_PRIVILEGES

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_FORWARD = _REPO / "docker" / "migrations" / "forward"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _REPO / "config" / "migration_classes.yaml"


@dataclass(frozen=True)
class _Grant:
    node: str
    file_name: str
    table: str
    create_file: str


_GRANTS = (
    _Grant(
        "node_projection_lab_job",
        "0002_grant_role_omnidash_lab_job_state_select.sql",
        "lab_job_state",
        "0000_create_lab_job.sql",
    ),
    _Grant(
        "node_projection_work_ledger",
        "0005_grant_role_omnidash_work_ledger_rows_select.sql",
        "work_ledger_rows",
        "0000_create_work_ledger.sql",
    ),
    _Grant(
        "node_projection_pr_state",
        "0002_grant_role_omnidash_pr_state_select.sql",
        "pr_state",
        "0000_create_pr_state.sql",
    ),
)
_IDS = [grant.table for grant in _GRANTS]


def _path(grant: _Grant) -> Path:
    return _FORWARD / "nodes" / grant.node / grant.file_name


def _statements(grant: _Grant) -> list[str]:
    """The file's SQL statements with ``--`` comments removed."""
    text = "\n".join(
        line
        for line in _path(grant).read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    )
    return [part.strip() for part in text.split(";") if part.strip()]


def _grants(grant: _Grant) -> list[str]:
    return [s for s in _statements(grant) if s.upper().startswith("GRANT")]


@pytest.mark.parametrize("grant", _GRANTS, ids=_IDS)
def test_the_migration_sits_beside_the_file_that_creates_the_table(
    grant: _Grant,
) -> None:
    assert _path(grant).is_file()
    created = (_path(grant).parent / grant.create_file).read_text(encoding="utf-8")
    assert f"omninode_internal.{grant.table}" in created


@pytest.mark.parametrize("grant", _GRANTS, ids=_IDS)
def test_it_grants_schema_usage_and_select_on_the_one_table_to_role_omnidash(
    grant: _Grant,
) -> None:
    normalised = [re.sub(r"\s+", " ", g) for g in _grants(grant)]
    assert normalised == [
        "GRANT USAGE ON SCHEMA omninode_internal TO role_omnidash",
        f"GRANT SELECT ON omninode_internal.{grant.table} TO role_omnidash",
    ]


@pytest.mark.parametrize("grant", _GRANTS, ids=_IDS)
def test_it_grants_nothing_beyond_select(grant: _Grant) -> None:
    sql = " ".join(_statements(grant)).upper()
    for forbidden in (
        "INSERT",
        "UPDATE",
        "DELETE",
        "TRUNCATE",
        "CREATE",
        "ALL PRIVILEGES",
        "WITH GRANT",
        "REVOKE",
    ):
        assert forbidden not in sql, forbidden
    assert all("role_omnidash" in g for g in _grants(grant))
    # Positive control: the scan above reads the real statements.
    assert "SELECT" in sql


@pytest.mark.parametrize("grant", _GRANTS, ids=_IDS)
def test_the_select_grant_is_asserted_after_it_is_issued(grant: _Grant) -> None:
    assertions = [s for s in _statements(grant) if s.upper().startswith("SELECT 1 /")]
    assert len(assertions) == 1
    assert f"table_name = '{grant.table}'" in assertions[0]
    assert "grantee = 'role_omnidash'" in assertions[0]
    assert "privilege_type = 'SELECT'" in assertions[0]


@pytest.mark.parametrize("grant", _GRANTS, ids=_IDS)
def test_the_manifest_row_binds_this_files_exact_bytes(grant: _Grant) -> None:
    rows = [
        line.split("\t")
        for line in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.startswith(f"nodes/{grant.node}/{grant.file_name}\t")
    ]
    assert len(rows) == 1
    _path_cell, owner, source, domain, migration_id, checksum = rows[0]
    assert (owner, source, domain) == (
        f"node:{grant.node}",
        f"node:{grant.node}",
        "omninode_internal",
    )
    assert migration_id == f"node:{grant.node}:{grant.file_name}"
    assert checksum == hashlib.sha256(_path(grant).read_bytes()).hexdigest()


@pytest.mark.parametrize("grant", _GRANTS, ids=_IDS)
def test_the_migration_is_classed_expand_only(grant: _Grant) -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{grant.node}/{grant.file_name}"] == "expand-only"


def test_the_grant_shape_is_select_only() -> None:
    assert frozenset({EnumDatabasePrivilege.SELECT}) == READ_PRIVILEGES


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_every_profile_declares_the_three_tables_with_select_only(
    profile: str,
) -> None:
    principal = (
        load_topology_profile(profile)
        .databases["application"]
        .principals["role_omnidash"]
    )
    for grant in _GRANTS:
        declared = [
            g
            for g in principal.grants
            if grant.table in g.objects and g.schema == "omninode_internal"
        ]
        assert len(declared) == 1, grant.table
        assert frozenset(declared[0].privileges) == READ_PRIVILEGES
