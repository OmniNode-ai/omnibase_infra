# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The retired supplemental bridges no longer shadow the pinned declarations (OMN-17292)."""

import pytest

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_RETIRED = (
    "agent_actions",
    "agent_detection_failures",
    "agent_execution_logs",
    "agent_status_events",
    "agent_transformation_events",
    "context_audit_events",
    "local_alerts",
    "overlay_store",
    "router_performance_metrics",
    "view_seam_readings",
)


@pytest.mark.parametrize("table", _RETIRED)
def test_retired_bridge_is_not_declared_by_the_legacy_table(table: str) -> None:
    assert not [d for d in LEGACY_MIGRATION_TABLE_DECLARATIONS if d.table.name == table]
