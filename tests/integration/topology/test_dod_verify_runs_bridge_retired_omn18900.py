# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The dod_verify_runs supplemental bridge was retired (OMN-18900 / OMN-18863).

`dod_verify_runs` reached the shipped topology instances through the same
infra-first supplemental-bridge mechanism documented in
`test_runtime_error_fingerprints_grant_omn18863.py`: this repository vendors
the migration that creates and grants the relation before the omnimarket
node package that declares it in its contract merges, so a hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entry carried it in the interim
(``table_grant_derivation.py``, tracked for expiry in ``_INTERIM_ENTRIES`` in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py``).

The omnimarket contract pin advancing to bf88396 (OMN-17292) makes that
bridge redundant -- the pinned contracts now declare the relation themselves
-- and ``test_no_interim_entry_is_redundant`` in that expiry module named the
remedy: delete the entry from both files, since the regenerated grant catalogs
come back byte-identical.

This module asserts the deletion actually happened and stuck, on committed
repository state alone (the pin-vs-declared-set comparison itself needs the
omnimarket checkout the CI enforcement job provides, and lives in
``tests/ci/test_supplemental_declaration_expiry_omn18863.py`` rather than
here):

1. no supplemental bridge for ``dod_verify_runs`` remains in
   ``LEGACY_MIGRATION_TABLE_DECLARATIONS``; and
2. the shipped topology instances still declare the relation, so the removal
   above did not silently drop a projection binding the vendored migration
   still grants -- the OMN-18768 reverse-ratchet failure mode a regeneration
   done wrong would trip.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATION = "dod_verify_runs"
_PROFILES = ("local", "onex-dev", "onex-prod")


def _instance_text(profile: str) -> str:
    return (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    ).read_text(encoding="utf-8")


class TestTheDodVerifyRunsBridgeWasRetired:
    def test_no_supplemental_bridge_remains(self) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert _RELATION not in carried, (
            f"{_RELATION} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry in table_grant_derivation.py, but the pinned "
            "omnimarket contracts (bf88396, OMN-17292) declare it. A "
            "redundant bridge contributes byte-identical output and nothing "
            "else will tell you it is there."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self,
    ) -> None:
        """Deleting the bridge must never delete what it bridged."""
        migration = Path(
            "docker/migrations/forward/nodes/node_projection_dod_verdict/"
            "0000_create_dod_verify_runs.sql"
        )
        assert (_REPO_ROOT / migration).is_file()

    def test_the_shipped_instance_still_declares_it(self) -> None:
        for profile in _PROFILES:
            document = yaml.safe_load(_instance_text(profile))
            declared = yaml.safe_dump(document)
            assert _RELATION in declared, (
                f"{profile} no longer declares {_RELATION}. Retiring the "
                "supplemental bridge must leave the shipped grant "
                "byte-identical, since the pinned contract now derives what "
                "the entry used to -- it must not remove the declaration"
            )
