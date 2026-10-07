# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The alert_channel_liveness_verdicts supplemental bridge was retired (OMN-17292).

Infra vendored the ``alert_channel_liveness_verdicts`` migration (OMN-20578)
before omnimarket#3417 landed the ``node_projection_alert_channel_liveness``
contract, so a hand-authored ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entry
carried it in the interim. The omnimarket contract pin advancing to
3d39b97615a6 carries omnimarket#3417, so the bridge is redundant and was
deleted from ``table_grant_derivation.py`` and ``_INTERIM_ENTRIES``.

This module asserts, on committed repository state alone, that the deletion
stuck and did not drop what it bridged:

1. no supplemental bridge for ``alert_channel_liveness_verdicts`` remains; and
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
_RELATION = "alert_channel_liveness_verdicts"
_PROFILES = ("local", "onex-dev", "onex-prod")


class TestTheAlertChannelLivenessBridgeWasRetired:
    def test_no_supplemental_bridge_remains(self) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert _RELATION not in carried, (
            f"{_RELATION} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry, but the pinned omnimarket contracts "
            "(3d39b97615a6, omnimarket#3417) declare it."
        )

    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self,
    ) -> None:
        migration = Path(
            "docker/migrations/forward/nodes/node_projection_alert_channel_liveness/"
            "0000_create_alert_channel_liveness_verdicts.sql"
        )
        assert (_REPO_ROOT / migration).is_file()

    def test_the_shipped_instance_still_declares_it(self) -> None:
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
            assert _RELATION in declared, (
                f"{profile} no longer declares {_RELATION}; retiring the "
                "bridge must leave the shipped grant byte-identical"
            )
