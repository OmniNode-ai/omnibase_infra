# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Assert the supplemental local_alerts interim declaration stays gone (OMN-15361).

The supplemental ``local_alerts`` declaration was deleted because the advanced
omnimarket pin declares the relation itself (``node_projection_local_alerts``,
OMN-19991). This module asserts the bridge stays gone and that the deletion took
only that relation: ``local_identity`` still waits on its own node contract.
Whether the pinned contracts declare ``local_alerts`` is enforced by the
Application Database Domain Enforcement (OMN-15361) job, which reads the pinned
checkout, so no test here is collected-but-skipped (the OMN-18776 skip ratchet
refuses a skip-guarded variant).
"""

from __future__ import annotations

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)


def test_no_supplemental_bridge_remains_for_local_alerts() -> None:
    bridged = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert "local_alerts" not in bridged


def test_the_local_identity_bridge_is_still_carried() -> None:
    """Positive control: the deletion removed one relation, not the tuple."""
    bridged = {d.table.name for d in LEGACY_MIGRATION_TABLE_DECLARATIONS}
    assert "local_identity" in bridged
