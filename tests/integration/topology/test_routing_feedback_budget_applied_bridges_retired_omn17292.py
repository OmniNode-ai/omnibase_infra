# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The routing feedback and budget applied bridges were retired (OMN-17292).

Infra vendored the ``delegation_routing_feedback`` and
``delegation_budget_applied_events`` migrations before the declaring node
contracts landed, so hand-authored ``LEGACY_MIGRATION_TABLE_DECLARATIONS``
entries carried them in the interim. The omnimarket contract pin advancing
to d2b0d9986ed7 carries omnimarket#3416 (OMN-20578), whose
``node_projection_routing_feedback`` contract declares the former, and
omnimarket#3436 (OMN-20613), whose ``node_projection_delegation`` contract
declares the latter, so the bridges are redundant and were deleted.

This module asserts, on committed repository state alone, that the deletion
stuck and did not drop what it bridged:

1. no supplemental bridge for either relation remains; and
2. the vendored migration lineage and the shipped topology instances still
   declare the relations.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATIONS = (
    (
        "delegation_routing_feedback",
        "docker/migrations/forward/nodes/node_projection_routing_feedback/"
        "0000_create_delegation_routing_feedback.sql",
    ),
    (
        "delegation_budget_applied_events",
        "docker/migrations/forward/nodes/node_projection_delegation/"
        "0056_delegation_budget_applied_events.sql",
    ),
)
_PROFILES = ("local", "onex-dev", "onex-prod")


@pytest.mark.parametrize(("relation", "migration"), _RELATIONS)
class TestTheRoutingFeedbackBudgetAppliedBridgesWereRetired:
    def test_no_supplemental_bridge_remains(
        self, relation: str, migration: str
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry, but the pinned omnimarket contracts "
            "(d2b0d9986ed7, omnimarket#3416, omnimarket#3436) declare it."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self, relation: str, migration: str
    ) -> None:
        assert (_REPO_ROOT / migration).is_file()

    def test_the_shipped_instance_still_declares_it(
        self, relation: str, migration: str
    ) -> None:
        for profile in _PROFILES:
            text = (
                _REPO_ROOT
                / "src"
                / "omnibase_infra"
                / "topology"
                / "instances"
                / f"{profile}.yaml"
            ).read_text(encoding="utf-8")
            declared = yaml.safe_dump(yaml.safe_load(text))
            assert relation in declared, (
                f"{profile} no longer declares {relation}; retiring the "
                "bridge must leave the shipped grant byte-identical"
            )
