# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Supplemental bridges retired by the 24030b21d16d pin advance (OMN-17292).

The omnimarket contract pin advancing to 24030b21d16d192a46020a78214d350947328124
carries node contracts for the relations below, so their hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries are deleted. This proof uses
committed repository state alone: no supplemental declaration remains, and the
declarations that stay are untouched.
"""

from __future__ import annotations

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_RETIRED_RELATIONS = (
    "delegation_lab_jobs",
    "delegation_lab_job_outbox",
    "delegation_budget_readings",
    "compaction_receipts",
    "delegation_capability_cells",
    "local_environment",
)

_STILL_BRIDGED_RELATIONS = (
    "local_alerts",
    "local_identity",
    "compaction_current_context",
)


@pytest.mark.parametrize("relation", _RETIRED_RELATIONS)
def test_no_supplemental_bridge_remains(relation: str) -> None:
    names = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert relation not in names


@pytest.mark.parametrize("relation", _STILL_BRIDGED_RELATIONS)
def test_unrelated_bridges_survive(relation: str) -> None:
    names = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert relation in names
