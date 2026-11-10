# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The goal trace supplemental bridge is declared and live (OMN-23341).

`goal_trace_events` and `goal_traces` reach the shipped topology instances
through the same infra-first supplemental-bridge mechanism documented in
`test_dod_verify_runs_bridge_retired_omn18900.py`: this repository vendors
the migrations that create and grant the two relations before the omnimarket
node package `node_projection_goal_trace` (whose migrations sit under
`docker/migrations/forward/nodes/node_projection_goal_trace/`) declares them
in its contract, so a hand-authored ``LEGACY_MIGRATION_TABLE_DECLARATIONS``
entry carries them in the interim (``table_grant_derivation.py``, tracked for
expiry in ``_INTERIM_ENTRIES`` in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py``).

Unlike the retired bridge that module asserts, these two entries must still
be present: the pinned omnimarket contracts have not yet declared
`goal_trace_events` and `goal_traces`, so the derivation cannot reproduce the
shipped grant without them. The vendored grant migration
`0001_grant_omninode_runtime_goal_trace.sql` gives the runtime the writer's
scope (SELECT, INSERT, UPDATE) ahead of the contract, and the expiry module
names the two relations against OMN-20034 so the bridge is told to expire the
moment the pinned contract declares them.

This module asserts that live state, on committed repository state alone
(the pin-vs-declared-set comparison itself needs the omnimarket checkout the
CI enforcement job provides, and lives in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py`` rather than
here):

1. each relation has exactly one supplemental bridge in
   ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` with ``read_write`` access whose
   migration path sits under the vendored node directory and is a real file;
2. the vendored grant migration names both relations and the
   ``omninode_runtime`` principal it grants;
3. the shipped topology instances still declare both relations, so the bridge
   is not a phantom;
4. the OMN-18863 expiry ledger tracks both relations against OMN-20034, so the
   bridge expires when the omnimarket contract reaches the pin;
5. a failing control: the helper used by the first assertion reports a
   bridge as missing the moment it is dropped, so the first assertion cannot
   go green for the wrong reason.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    ContractTableDeclaration,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATIONS = ("goal_trace_events", "goal_traces")
_NEIGHBOUR_RELATION = "dod_verify_runs"
_INSTANCES = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_NODE_DIR = "node_projection_goal_trace"
_MIGRATION_PREFIX = f"docker/migrations/forward/nodes/{_NODE_DIR}/"
_GRANT_MIGRATION = f"{_MIGRATION_PREFIX}0001_grant_omninode_runtime_goal_trace.sql"
_EXPIRY_MODULE = (
    _REPO_ROOT / "tests" / "ci" / "test_supplemental_declaration_expiry_omn18863.py"
)


def _instance_documents() -> dict[str, str]:
    """Every shipped instance that declares the neighbouring projection."""
    return {
        path.name: yaml.safe_dump(yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(_INSTANCES.glob("*.yaml"))
        if _NEIGHBOUR_RELATION in path.read_text(encoding="utf-8")
    }


def _bridge_entries(
    declarations: tuple[ContractTableDeclaration, ...],
    relation: str,
) -> list[ContractTableDeclaration]:
    """The supplemental entries whose declared table is ``relation``."""
    return [
        declaration
        for declaration in declarations
        if declaration.table.name == relation
    ]


class TestTheGoalTraceBridgeIsDeclaredAndExpires:
    def test_each_relation_has_one_supplemental_bridge(self) -> None:
        for relation in _RELATIONS:
            entries = _bridge_entries(LEGACY_MIGRATION_TABLE_DECLARATIONS, relation)
            assert len(entries) == 1, (
                f"{relation} expected exactly one LEGACY_MIGRATION_TABLE_"
                "DECLARATIONS entry, found "
                f"{len(entries)}. The goal trace bridge is live (OMN-23341) "
                "and must not be deleted until the pinned omnimarket "
                "contract declares the relation."
            )
            entry = entries[0]
            assert entry.table.access == "read_write", (
                f"{relation} bridge declares access {entry.table.access!r}, "
                "not the writer scope 'read_write' the vendored grant "
                "migration delivers."
            )
            assert entry.table.migration.startswith(_MIGRATION_PREFIX), (
                f"{relation} bridge points at {entry.table.migration!r}, "
                f"outside the vendored node directory {_MIGRATION_PREFIX}"
            )
            assert (_REPO_ROOT / entry.table.migration).is_file(), (
                f"{relation} bridge migration {entry.table.migration} is not "
                "a file in this repo."
            )

    def test_the_vendored_grant_migration_names_the_runtime_principal(
        self,
    ) -> None:
        """The grant the bridge relies on must exist and name both relations
        plus the principal it grants."""
        text = (_REPO_ROOT / _GRANT_MIGRATION).read_text(encoding="utf-8")
        for relation in _RELATIONS:
            assert relation in text, f"{_GRANT_MIGRATION} does not grant {relation}."
        assert "omninode_runtime" in text, (
            f"{_GRANT_MIGRATION} names no runtime principal; the bridge "
            "would grant a relation nobody can use."
        )

    def test_the_shipped_instances_declare_both_relations(self) -> None:
        documents = _instance_documents()
        assert documents, "no shipped instance declares the neighbouring projection"
        for name, declared in documents.items():
            for relation in _RELATIONS:
                assert relation in declared, (
                    f"{name} declares {_NEIGHBOUR_RELATION} but no longer "
                    f"declares {relation}. The live supplemental bridge must "
                    "keep the shipped grant the vendored migration grants."
                )

    def test_the_expiry_ledger_tracks_both_relations(self) -> None:
        lines = _EXPIRY_MODULE.read_text(encoding="utf-8").splitlines()
        for relation in _RELATIONS:
            quoted = f'"{relation}"'
            tracked = any(quoted in line and "OMN-20034" in line for line in lines)
            assert tracked, (
                f"{relation} is not tracked by the OMN-18863 expiry module: "
                "no line quotes the relation alongside OMN-20034, so the bridge would never be told "
                "to expire."
            )

    def test_a_missing_bridge_is_detected(self) -> None:
        """Failing control: the helper reports a missing bridge, so the first
        assertion cannot go green for the wrong reason."""
        for relation in _RELATIONS:
            assert (
                len(_bridge_entries(LEGACY_MIGRATION_TABLE_DECLARATIONS, relation)) == 1
            )
        filtered = tuple(
            declaration
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
            if declaration.table.name not in _RELATIONS
        )
        for relation in _RELATIONS:
            assert _bridge_entries(filtered, relation) == [], (
                f"the helper still found a bridge for {relation} after the "
                "declaration was dropped; it cannot detect a missing bridge."
            )
