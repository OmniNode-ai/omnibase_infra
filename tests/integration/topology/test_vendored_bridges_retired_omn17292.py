# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Five supplemental table bridges were retired by the pin advance (OMN-17292).

Infra vendored these migrations before the omnimarket node contracts that
declare their relations landed, so hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries carried them in the interim:

- ``work_ledger_rows`` and ``work_ledger_state`` (omnimarket#3050)
- ``pr_state`` (omnimarket#3054)
- ``usage_by_model_day`` and ``usage_by_model_day_calls`` (omnimarket#3073)

The omnimarket contract pin advancing to 92bc73177b90 carries all three pull
requests, so the bridges are redundant and were deleted from
``table_grant_derivation.py`` and ``_INTERIM_ENTRIES``.

This module asserts, on committed repository state alone, that the deletion
stuck and did not drop what it bridged:

1. no supplemental bridge for any of the five relations remains; and
2. each shipped topology instance still declares each relation.
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
    "work_ledger_rows",
    "work_ledger_state",
    "pr_state",
    "usage_by_model_day",
    "usage_by_model_day_calls",
)
_PROFILES = ("local", "onex-dev", "onex-prod")


class TestTheVendoredBridgesWereRetired:
    @pytest.mark.parametrize("relation", _RELATIONS)
    def test_no_supplemental_bridge_remains(self, relation: str) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry, but the pinned omnimarket contracts "
            "(92bc73177b90) declare it."
        )

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_the_shipped_instance_still_declares_every_relation(
        self, profile: str
    ) -> None:
        text = (
            _REPO_ROOT
            / "src"
            / "omnibase_infra"
            / "topology"
            / "instances"
            / f"{profile}.yaml"
        ).read_text(encoding="utf-8")
        declared = yaml.safe_dump(yaml.safe_load(text))
        missing = [relation for relation in _RELATIONS if relation not in declared]
        assert not missing, (
            f"{profile} no longer declares {missing}; retiring the bridges "
            "must leave the shipped grants byte-identical"
        )
