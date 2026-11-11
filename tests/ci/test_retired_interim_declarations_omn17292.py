# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Interim table declarations retired by the omnimarket pin advance (OMN-17292).

The omnimarket pin advance makes the supplemental declarations for these
relations redundant; the entries must be gone from the tuple.
"""

from __future__ import annotations

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

_RETIRED = (
    "context_audit_events",
    "local_alerts",
    "overlay_store",
    "view_seam_readings",
)


@pytest.mark.unit
@pytest.mark.parametrize("relation", _RETIRED)
def test_retired_interim_declaration_absent(relation: str) -> None:
    declared = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert relation not in declared
