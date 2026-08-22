# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16316: physical-schema bridge proof for tenant_inference_credentials.

Pure topology resolution, no live database required -- mirrors the paired
seam proof in
tests/integration/runtime/test_live_events_projection_write_path_omn15359.py::
test_grant_derivation_schema_agrees_with_the_insert_target_schema.

The migration's own docstring records the promotion path: this table is
logically tenant-domain but physically created bare in `public` (no `tenant`
Postgres schema exists on any lane yet), same bridge as its siblings
delegation_events and delegation_routing_tenant_overlay
(TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
src/omnibase_infra/topology/physical_schema_mapping.py). The node's own
contract currently declares `schema: public` directly (matching the physical
location with no bridge indirection needed today), but the bridge entry
itself must independently resolve correctly for the day this table's logical
schema is promoted to `tenant` -- this test proves that resolution now,
before any runtime code depends on it, closing the exact class of gap
CodeRabbit flagged on infra#2823 (grant derivation silently disagreeing with
the real physical location).
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


def test_tenant_inference_credentials_bridge_agrees_with_the_insert_target_schema() -> (
    None
):
    """The grant-privilege check and the real SQL INSERT target must resolve
    to the identical physical schema for a hypothetical `schema: tenant`
    declaration -- the exact seam a bridge-set omission would silently
    reintroduce, proven independently of whichever schema the node's own
    contract declares today."""
    declaration = ModelDbTableDeclaration(
        name="tenant_inference_credentials",
        database_ref="application",
        schema="tenant",
        migration=(
            "docker/migrations/forward/nodes/node_projection_tenant_credentials/"
            "0000_create_tenant_inference_credentials.sql"
        ),
        access="write",
        role="credentials",
    )
    target = _resolve_projection_database_target((declaration,), application_topology())
    insert_target_schema = target.table_targets[0].physical_schema

    grant_check_schema = physical_grant_schema_for_table(
        "tenant", "tenant_inference_credentials"
    )

    assert grant_check_schema == "public"
    assert grant_check_schema == insert_target_schema
