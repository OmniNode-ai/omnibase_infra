# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A relation is declared as data in the shape of its node contract (OMN-20134).

The TABLE grant derivation used to carry the relations a node's vendored
migrations create ahead of the pinned omnimarket contracts as a Python tuple in
``table_grant_derivation.py``, a baselined file the canonical-file-shape ratchet
(OMN-20304) lets only shrink. A new projection relation could therefore get no
runtime grant: the tuple could not grow, and the grant is what omnimarket's own
wiring test needs before the relation's node can merge.

The declaration is data now: ``docker/migrations/forward/nodes/<node>/db_io.yaml``,
the ``db_io`` block of the node's contract (``ModelDbTableDeclaration``, the field
the contract already uses for an owned relation), read by
``load_vendored_declarations`` exactly as ``load_contract_declarations`` reads a
contract. The principal is the schema domain's projection binding and the
privileges follow from ``access``, so no principal and no privilege is named in a
declaration file or in the derivation module.

What keeps a data file honest is checked here: it names a migration this
repository holds, it declares each relation once, it uses an access mode the
derivation maps, it agrees with the pinned omnimarket contract wherever that
contract declares the same relation, and a relation it declares derives its
grant on every shipped instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.models.core import ModelDeploymentTopology
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    VENDORED_DECLARATION_FILE,
    VENDORED_NODES_ROOT,
    VENDORED_TABLE_DECLARATIONS,
    derive_topology_table_grants,
    load_contract_declarations,
    load_vendored_declarations,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_INSTANCES = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_PIN_NODES = (
    _REPO_ROOT
    / ".proof-dependencies"
    / "omnimarket-pin"
    / "src"
    / "omnimarket"
    / "nodes"
)
_SINGLE_NODES = (
    _REPO_ROOT / ".proof-dependencies" / "omnimarket" / "src" / "omnimarket" / "nodes"
)
_PINNED = _PIN_NODES if _PIN_NODES.is_dir() else _SINGLE_NODES
_DERIVABLE_ACCESS = {"read", "write", "read_write"}
_DECLARATION_FILES = sorted(VENDORED_NODES_ROOT.glob(f"*/{VENDORED_DECLARATION_FILE}"))


def test_the_vendored_declarations_are_found_and_non_empty() -> None:
    """Positive control: a loader that saw no file would make every check below vacuous."""
    assert len(_DECLARATION_FILES) >= 10
    assert load_vendored_declarations() == VENDORED_TABLE_DECLARATIONS
    assert len(VENDORED_TABLE_DECLARATIONS) >= len(_DECLARATION_FILES)


def test_a_missing_vendored_root_is_an_error_and_not_an_empty_set(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_vendored_declarations(tmp_path / "absent")


def test_a_declaration_file_is_read_like_a_contract_db_io_block(
    tmp_path: Path,
) -> None:
    node = tmp_path / "node_projection_planted"
    node.mkdir()
    (node / VENDORED_DECLARATION_FILE).write_text(
        yaml.safe_dump(
            {
                "db_io": {
                    "db_tables": [
                        {
                            "name": "planted_rows",
                            "database_ref": "application",
                            "schema": "omninode_internal",
                            "migration": "docker/migrations/forward/nodes/x/0000.sql",
                            "access": "read_write",
                            "role": "planted",
                        }
                    ]
                }
            }
        )
    )
    (planted,) = load_vendored_declarations(tmp_path)
    assert planted.node == "vendored:node_projection_planted"
    assert planted.table.name == "planted_rows"
    assert str(planted.contract_path) == planted.table.migration


@pytest.mark.parametrize("path", _DECLARATION_FILES, ids=lambda path: path.parent.name)
def test_each_declaration_names_a_migration_this_repository_holds(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(document) == {"db_io"}
    for entry in document["db_io"]["db_tables"]:
        migration = _REPO_ROOT / entry["migration"]
        assert migration.is_file(), f"{path.parent.name}: {entry['migration']}"
        assert migration.is_relative_to(
            _REPO_ROOT / "docker" / "migrations" / "forward"
        )
        assert entry["access"] in _DERIVABLE_ACCESS


def test_a_relation_is_declared_once_across_every_vendored_file() -> None:
    keys = [
        (d.table.database_ref, d.table.schema, d.table.name)
        for d in VENDORED_TABLE_DECLARATIONS
    ]
    assert len(keys) == len(set(keys))


@pytest.mark.skipif(
    not _PINNED.is_dir(),
    reason="needs the pinned omnimarket checkout the OMN-15361 enforcement job provides",
)
def test_a_vendored_relation_agrees_with_the_pinned_contract_that_declares_it() -> None:
    """A data file is a copy of the contract's declaration, never a second opinion."""
    pinned = {
        (d.table.database_ref, d.table.schema, d.table.name): d.table
        for d in load_contract_declarations(_PINNED)
    }
    disagreements = []
    for declaration in VENDORED_TABLE_DECLARATIONS:
        mine = declaration.table
        theirs = pinned.get((mine.database_ref, mine.schema, mine.name))
        if theirs is not None and (mine.access, mine.role) != (
            theirs.access,
            theirs.role,
        ):
            disagreements.append(
                f"{mine.name}: vendored ({mine.access}, {mine.role}) != "
                f"pinned ({theirs.access}, {theirs.role})"
            )
    assert not disagreements, disagreements


@pytest.mark.parametrize(
    "instance", sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values()))
)
def test_every_vendored_relation_derives_a_grant_on_every_shipped_instance(
    instance: str,
) -> None:
    topology = ModelDeploymentTopology.from_yaml(_INSTANCES / f"{instance}.yaml")
    derived = derive_topology_table_grants(topology, VENDORED_TABLE_DECLARATIONS)
    granted = {
        name
        for per_database in derived.per_database.values()
        for grants in per_database.grants.values()
        for grant in grants
        for name in grant.objects
    }
    missing = [
        d.table.name
        for d in VENDORED_TABLE_DECLARATIONS
        if d.table.database_ref == "application" and d.table.name not in granted
    ]
    assert not missing, f"{instance}: declared but derives no grant: {missing}"
    assert not derived.unmappable
