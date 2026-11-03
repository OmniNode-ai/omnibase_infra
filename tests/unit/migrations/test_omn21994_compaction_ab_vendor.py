# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the interim vendored compaction A/B migrations and their topology grants.

omnimarket's node_projection_compaction_ab (OMN-21994) declares two
omninode_internal relations with access read_write, and its Event Chain Gate
loads the contract against this repo's shipped topology. Until the node's
contract is in the pinned omnimarket release, the two SQL files are held here:
the digests pin the byte-for-byte copy of the reviewed bytes, and the
``omninode_runtime`` table grants are derived from the interim declarations, then
proved through the same resolver the auto-wiring gate calls.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_LEDGER = _FORWARD / "_ledger/application-migrations.tsv"
_NODE = "node_projection_compaction_ab"
_CREATE = "0000_create_compaction_ab.sql"
_GRANT = "0001_grant_omninode_runtime_compaction_ab.sql"
_VENDORED = [
    (_CREATE, "934d8b2aff324f0be7ab2836fab8547769dd1f3bea451220a27a5746a7b3e97e"),
    (_GRANT, "3bc2e87ba848872098ef2f4b358943a7dd788201482ddc431041ba0cbaeade15"),
]
_CLASSES = {_CREATE: "forward-only", _GRANT: "expand-only"}
_TABLES = ("compaction_ab_rows", "compaction_ab_build_series")


def test_only_the_create_and_grant_files_are_vendored() -> None:
    names = sorted(path.name for path in (_FORWARD / "nodes" / _NODE).glob("*"))
    assert names == [_CREATE, _GRANT]


@pytest.mark.parametrize(("name", "sha256"), _VENDORED)
def test_vendor_bytes_and_manifest_binding_are_exact(name: str, sha256: str) -> None:
    artifact = f"nodes/{_NODE}/{name}"
    assert hashlib.sha256((_FORWARD / artifact).read_bytes()).hexdigest() == sha256
    rows = [line.split("\t") for line in _LEDGER.read_text().splitlines()]
    assert [row for row in rows if row[0] == artifact] == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{name}",
            sha256,
        ]
    ]


@pytest.mark.parametrize(("name", "sha256"), _VENDORED)
def test_the_vendor_is_declared_with_its_class(name: str, sha256: str) -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{_NODE}/{name}"] == _CLASSES[name]


def test_neither_migration_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    for name, _ in _VENDORED:
        assert f"node:{_NODE}:{name}" not in ids


@pytest.mark.parametrize("table_name", _TABLES)
def test_the_interim_declaration_names_the_vendored_create(table_name: str) -> None:
    (declaration,) = [
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == table_name
    ]
    table = declaration.table
    assert table.database_ref == "application"
    assert table.schema == "omninode_internal"
    assert table.access == "read_write"
    assert table.role == table_name
    assert table.migration == f"docker/migrations/forward/nodes/{_NODE}/{_CREATE}"


def _contract_declaration(table_name: str) -> ModelDbTableDeclaration:
    """The declaration omnimarket's node contract carries, spelled independently."""
    return ModelDbTableDeclaration(
        name=table_name,
        database_ref="application",
        schema="omninode_internal",
        migration=_CREATE,
        access="read_write",
        role=table_name,
    )


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
@pytest.mark.parametrize("table_name", _TABLES)
def test_runtime_principal_resolves_the_contract_declared_table(
    profile: str, table_name: str
) -> None:
    """The gate's own check, on the shipped topology: read and write succeed."""
    target = _resolve_projection_database_target(
        [_contract_declaration(table_name)], load_topology_profile(profile)
    )
    (resolved,) = target.table_targets
    assert resolved.table.name == table_name
    for binding in (resolved.read_binding, resolved.write_binding):
        assert binding is not None
        assert binding.principal == "omninode_runtime"
