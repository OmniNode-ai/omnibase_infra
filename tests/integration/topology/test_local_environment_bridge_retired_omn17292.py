# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The local_environment supplemental bridge was retired (OMN-17292).

Infra vendored the ``local_environment`` migration before omnimarket#4161
landed the ``node_projection_local_environment`` contract, so a hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entry (OMN-19988) carried it in the
interim. The omnimarket contract pin advancing to release tag v0.4.319 carries
omnimarket#4161, so the bridge is redundant and was deleted from
``table_grant_derivation.py`` and ``_INTERIM_ENTRIES``.

This module asserts, on committed repository state alone, that the deletion
stuck and did not drop what it bridged:

1. no supplemental bridge for ``local_environment`` remains; and
2. the vendored migration lineage and the shipped topology instances still
   declare the relation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATION = "local_environment"
_INSTANCES = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"


class TestTheLocalEnvironmentBridgeWasRetired:
    def test_no_supplemental_bridge_remains(self) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert _RELATION not in carried, (
            f"{_RELATION} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry, but the pinned omnimarket contracts "
            "(v0.4.319, omnimarket#4161) declare it."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self,
    ) -> None:
        for filename in (
            "0000_create_local_environment.sql",
            "0001_grant_tenant_projection_writer_local_environment.sql",
            "0002_force_rls_local_environment.sql",
        ):
            migration = Path(
                "docker/migrations/forward/nodes/node_projection_local_environment/"
                f"{filename}"
            )
            assert (_REPO_ROOT / migration).is_file()

    def test_the_shipped_instance_still_declares_it(self) -> None:
        instances = sorted(_INSTANCES.glob("*.yaml"))
        assert instances, f"no shipped topology instance under {_INSTANCES}"
        for instance in instances:
            text = instance.read_text(encoding="utf-8")
            declared = yaml.safe_dump(yaml.safe_load(text))
            assert _RELATION in declared, (
                f"{instance.stem} no longer declares {_RELATION}; retiring the "
                "bridge must leave the shipped grant byte-identical"
            )
