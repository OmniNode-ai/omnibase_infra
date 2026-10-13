# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Six supplemental table bridges were retired by the pin advance (OMN-17292).

Infra vendored these migrations before the omnimarket node contracts that
declare their relations landed, so hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries carried them in the interim:

- ``lab_job_state`` and ``lab_job_transitions`` (OMN-20604, omnimarket#3434)
- ``automation_liveness_state``, ``automation_run_history`` and
  ``automation_alarm_episodes`` (OMN-20802)
- ``run_trace`` (OMN-19987)

The omnimarket contract pin advancing to 69fcd9778c46 carries the pull
requests that declare all six, so the bridges are redundant and were deleted
from ``table_grant_derivation.py`` and ``_INTERIM_ENTRIES``.

This module asserts, on committed repository state alone, that the deletion
stuck and did not drop what it bridged:

1. no supplemental bridge for any of the six relations remains; and
2. each shipped topology instance still declares each relation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

_INSTANCES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "topology"
    / "instances"
)
_RELATIONS = (
    "lab_job_state",
    "lab_job_transitions",
    "automation_liveness_state",
    "automation_run_history",
    "automation_alarm_episodes",
    "run_trace",
)
_INSTANCES = tuple(sorted(_INSTANCES_DIR.glob("*.yaml")))


class TestTheLabJobLivenessRunTraceBridgesWereRetired:
    @pytest.mark.parametrize("relation", _RELATIONS)
    def test_no_supplemental_bridge_remains(self, relation: str) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried, (
            f"{relation} still has a supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entry, but the pinned omnimarket contracts "
            "(69fcd9778c46) declare it."
        )

    def test_the_instances_are_found(self) -> None:
        assert _INSTANCES, f"no topology instance under {_INSTANCES_DIR}"

    @pytest.mark.parametrize("instance", _INSTANCES, ids=lambda path: path.stem)
    def test_the_shipped_instance_still_declares_every_relation(
        self, instance: Path
    ) -> None:
        text = instance.read_text(encoding="utf-8")
        declared = yaml.safe_dump(yaml.safe_load(text))
        missing = [
            relation
            for relation in _RELATIONS
            if not re.search(rf"\b{relation}\b", declared)
        ]
        assert not missing, (
            f"{instance.name} no longer declares {missing}; retiring the bridges "
            "must leave the shipped grants byte-identical"
        )
