# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-16316 / OMN-17887: physical-schema proof for tenant_inference_credentials.

Pure topology resolution, no live database required -- mirrors the paired
seam proof in
tests/integration/runtime/test_live_events_projection_write_path_omn15359.py::
test_grant_derivation_schema_agrees_with_the_insert_target_schema.

This table is tenant-domain and physically created bare in ``public``. It was
originally proved through the tenant->public bridge for the day its logical
schema was promoted to ``tenant``. OMN-17887 (operator ruling 2026-09-24)
retired that promotion: the TENANT domain's schema is ``public`` for good, the
bridge entry is deleted, and no ``tenant`` Postgres schema will be built.

The seam this test guards -- grant derivation silently disagreeing with the
real INSERT target, the class of gap CodeRabbit flagged on infra#2823 -- is
now proved against the declaration the node's own contract actually makes,
``schema: public``: the grant check, the INSERT target and the shipped TABLE
grant all name ``public``, the relation is still TENANT-domain and bound to
``tenant_projection``, and a ``tenant`` declaration is refused as unknown
rather than silently bridged.
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

_RELATION = "tenant_inference_credentials"


def _declaration(schema: str) -> ModelDbTableDeclaration:
    return ModelDbTableDeclaration(
        name=_RELATION,
        database_ref="application",
        schema=schema,
        migration=(
            "docker/migrations/forward/nodes/node_projection_tenant_credentials/"
            "0000_create_tenant_inference_credentials.sql"
        ),
        access="write",
        role="credentials",
    )


def test_tenant_inference_credentials_bridge_agrees_with_the_insert_target_schema() -> (
    None
):
    """The grant-privilege check and the real SQL INSERT target must resolve
    to the identical physical schema, ``public``, for the TENANT-domain
    declaration -- and the retired ``tenant`` schema must be refused."""
    topology = application_topology()
    target = _resolve_projection_database_target((_declaration("public"),), topology)
    insert_target_schema = target.table_targets[0].physical_schema

    grant_check_schema = physical_grant_schema_for_table("public", _RELATION)

    assert grant_check_schema == "public"
    assert grant_check_schema == insert_target_schema
    assert target.table_targets[0].table.schema == "public"
    assert target.domains == (EnumDatabaseSchemaDomain.TENANT,)
    assert [binding.binding_ref for binding in target.bindings] == ["tenant_projection"]

    database = topology.databases["application"]
    principal = database.bindings["tenant_projection"].principal
    assert any(
        grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == insert_target_schema
        and _RELATION in grant.objects
        for grant in database.principals[principal].grants
    ), "the shipped topology must grant the table where the runtime inserts"

    with pytest.raises(
        ValueError, match="Unknown schema 'tenant' for database_ref 'application'"
    ):
        _resolve_projection_database_target((_declaration("tenant"),), topology)
