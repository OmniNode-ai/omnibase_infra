# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17426 / OMN-17887: savings overview resolves to ``public`` as TENANT.

``projection_cost_savings_overview`` is a tenant-domain read view that its
source migration physically created in ``public`` alongside
``projection_delegation_savings`` and ``projection_delegation_savings_series``.
Migration 089 is the first deployable migration to name the overview view, so
the topology grant resolver and runtime SQL target must agree on the physical
schema.

OMN-17887 (operator ruling 2026-09-24) retired the ``tenant`` schema: the
TENANT domain's schema is ``public`` for good. The tenant->public bridge this
test originally proved is deleted, and the seam now holds by construction --
the relation is declared ``public``, reads from ``public``, is granted in
``public``, and is still classified TENANT and bound to ``tenant_projection``.
A ``tenant`` declaration is refused as an unknown schema rather than silently
bridged.
"""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
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

_RELATION = "projection_cost_savings_overview"


def _declaration(schema: str) -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=_RELATION,
        database_ref="application",
        schema=schema,
        migration=(
            "docker/migrations/forward/nodes/node_projection_savings/"
            "089_savings_aggregate_views_per_tenant.sql"
        ),
        access="read",
        role="savings_overview",
    )


def test_savings_overview_bridge_agrees_with_the_runtime_read_target() -> None:
    topology = application_topology()
    target = _resolve_projection_database_target((_declaration("public"),), topology)
    read_target_schema = target.table_targets[0].physical_schema
    grant_check_schema = physical_grant_schema_for_table("public", _RELATION)

    assert grant_check_schema == "public"
    assert read_target_schema == grant_check_schema
    assert target.table_targets[0].table.schema == "public"
    assert target.domains == (EnumDatabaseSchemaDomain.TENANT,)
    assert [binding.binding_ref for binding in target.bindings] == ["tenant_projection"]

    database = topology.databases["application"]
    principal = database.bindings["tenant_projection"].principal
    assert any(
        grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == grant_check_schema
        and _RELATION in grant.objects
        for grant in database.principals[principal].grants
    ), "the shipped topology must grant the view where the runtime reads it"

    with pytest.raises(
        ValueError, match="Unknown schema 'tenant' for database_ref 'application'"
    ):
        _resolve_projection_database_target((_declaration("tenant"),), topology)
