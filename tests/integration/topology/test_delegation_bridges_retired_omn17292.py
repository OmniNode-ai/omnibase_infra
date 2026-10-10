# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegation_routing_feedback and delegation_budget_applied_events bridges were retired (OMN-17292).

Infra vendored both migrations (OMN-20578, OMN-20613) before the omnimarket
contracts that declare them were in the pinned set, so hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries carried them in the interim.
The omnimarket contract pin advancing to b7c20160d72c carries
``node_projection_routing_feedback`` (omnimarket#3416) and
``node_projection_delegation`` (OMN-20613), so both bridges are redundant and
were deleted from ``table_grant_derivation.py`` and ``_INTERIM_ENTRIES``.

This module asserts, on committed repository state alone, that the deletion
stuck and did not drop what it bridged, for each relation:

1. no supplemental bridge remains; and
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
_INSTANCES = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_RETIRED_BRIDGES = {
    "delegation_routing_feedback": Path(
        "docker/migrations/forward/nodes/node_projection_routing_feedback/"
        "0000_create_delegation_routing_feedback.sql"
    ),
    "delegation_budget_applied_events": Path(
        "docker/migrations/forward/nodes/node_projection_delegation/"
        "0056_delegation_budget_applied_events.sql"
    ),
}


@pytest.mark.parametrize("relation", sorted(_RETIRED_BRIDGES))
class TestTheDelegationBridgesWereRetired:
    def test_no_supplemental_bridge_remains(self, relation: str) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry, but the pinned omnimarket contracts "
            "(b7c20160d72c) declare it."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self, relation: str
    ) -> None:
        assert (_REPO_ROOT / _RETIRED_BRIDGES[relation]).is_file()

    def test_the_shipped_instance_still_declares_it(self, relation: str) -> None:
        instances = sorted(_INSTANCES.glob("*.yaml"))
        assert instances, f"no shipped topology instances under {_INSTANCES}"
        for instance in instances:
            text = instance.read_text(encoding="utf-8")
            declared = yaml.safe_dump(yaml.safe_load(text))
            assert relation in declared, (
                f"{instance.name} no longer declares {relation}; retiring the "
                "bridge must leave the shipped grant byte-identical"
            )
