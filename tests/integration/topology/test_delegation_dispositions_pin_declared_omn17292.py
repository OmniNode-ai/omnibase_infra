# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Assert the supplemental delegation dispositions bridge stays gone (OMN-17292).

The supplemental ``legacy_migration:delegation_dispositions`` declaration was
deleted because the advanced omnimarket pin declares the relation itself. This
module asserts the bridge stays gone. Whether the pinned contracts declare the
relation is enforced by the Application Database Domain Enforcement (OMN-15361)
job, which reads the pinned checkout, so no test here is collected-but-skipped
(the OMN-18776 skip ratchet refused the earlier skip-guarded variant).
"""

from __future__ import annotations

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)


def test_no_supplemental_bridge_remains_for_delegation_dispositions() -> None:
    bridged = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert "delegation_dispositions" not in bridged
