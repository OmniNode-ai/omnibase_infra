# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17426: savings overview uses the same public-backed tenant bridge.

``projection_cost_savings_overview`` is declared as a tenant-domain read view,
but its source migration physically created it in ``public`` alongside
``projection_delegation_savings`` and ``projection_delegation_savings_series``.
Migration 089 is the first deployable migration after the bridge gate to name
the overview view, so the topology grant resolver and runtime SQL target must
agree on the physical schema before the vendored migration can merge.
"""

from __future__ import annotations

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology.physical_schema_mapping import (
    physical_grant_schema_for_table,
)
from tests.helpers.application_db_topology import application_topology

pytestmark = pytest.mark.integration


def test_savings_overview_bridge_agrees_with_the_runtime_read_target() -> None:
    declaration = ModelDbTableDeclaration(
        name="projection_cost_savings_overview",
        database_ref="application",
        schema="tenant",
        migration=(
            "docker/migrations/forward/nodes/node_projection_savings/"
            "089_savings_aggregate_views_per_tenant.sql"
        ),
        access="read",
        role="savings_overview",
    )

    target = _resolve_projection_database_target((declaration,), application_topology())
    read_target_schema = target.table_targets[0].physical_schema
    grant_check_schema = physical_grant_schema_for_table(
        "tenant", "projection_cost_savings_overview"
    )

    assert grant_check_schema == "public"
    assert read_target_schema == grant_check_schema
