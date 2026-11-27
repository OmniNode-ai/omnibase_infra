# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The shipped topology delivers the writer's grant on ``savings_priced_models`` (OMN-26444).

omnimarket's metering summary writer fills ``public.savings_priced_models``
before it reads runs, and the fill deletes the rows the pricing manifest no
longer names. The runtime resolver proves a projection binding against the
grants each shipped topology profile declares, not against the database. This
module runs that resolver against every real profile and derives the grants
from the real topology, with no fixture topology and no copy of the migration:

1. every supported profile resolves a read_write ``savings_priced_models``
   declaration (on a topology without the grant the resolver raises "lacks
   declared read privileges: SELECT on table public.savings_priced_models");
2. every shipped instance gives ``tenant_projection_writer`` SELECT, INSERT,
   UPDATE and DELETE on it; and
3. the derivation of the pruned relation yields the DELETE + SELECT grant the
   renderer writes into those instances.

The declaration is built here rather than read from the vendored migration, so
the test fails for the missing grant and not for a missing declaration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_database_grant import (
    ModelDeploymentTopologyDatabaseGrant,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import SUPPORTED_TOPOLOGY_PROFILES, load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    ContractTableDeclaration,
    derive_topology_table_grants,
)

pytestmark = pytest.mark.integration

_TABLE = "savings_priced_models"
_WRITER = "tenant_projection_writer"
_REQUIRED = {"SELECT", "INSERT", "UPDATE", "DELETE"}
_PROFILES = tuple(sorted(SUPPORTED_TOPOLOGY_PROFILES))

_DECLARATION = ModelDbTableDeclaration(
    name=_TABLE,
    database_ref="application",
    schema="public",
    migration="docker/migrations/forward/nodes/node_projection_savings/102_delegation_savings_unpriced_matches_refresh.sql",
    access="read_write",
    role=_TABLE,
)


@pytest.mark.parametrize("profile", _PROFILES)
def test_every_profile_resolves_the_writers_binding(profile: str) -> None:
    _resolve_projection_database_target((_DECLARATION,), load_topology_profile(profile))


@pytest.mark.parametrize("profile", _PROFILES)
def test_the_writer_holds_all_four_privileges_on_the_table(profile: str) -> None:
    topology = load_topology_profile(profile)
    granted: set[str] = set()
    for principal_name, grants in _declared_grants(topology):
        if principal_name != _WRITER:
            continue
        for grant in grants:
            if grant.schema == "public" and _TABLE in grant.objects:
                granted.update(privilege.value for privilege in grant.privileges)
    assert granted >= _REQUIRED, (
        f"{profile}: {_WRITER} holds {sorted(granted)} on public.{_TABLE}, "
        f"missing {sorted(_REQUIRED - granted)}"
    )


def test_the_derivation_delivers_delete_and_select_to_the_writer() -> None:
    derived = derive_topology_table_grants(
        load_topology_profile("local"),
        [
            ContractTableDeclaration(
                node=f"legacy_migration:{_TABLE}",
                contract_path=Path(_DECLARATION.migration),
                table=_DECLARATION,
            )
        ],
    )
    delete_grants = [
        grant
        for principal_grants in derived.per_database["application"].grants.values()
        for grant in principal_grants
        if _TABLE in grant.objects and EnumDatabasePrivilege.DELETE in grant.privileges
    ]
    assert len(delete_grants) == 1
    assert {privilege.value for privilege in delete_grants[0].privileges} == {
        "DELETE",
        "SELECT",
    }


def _declared_grants(
    topology: ModelDeploymentTopology,
) -> list[tuple[str, tuple[ModelDeploymentTopologyDatabaseGrant, ...]]]:
    pairs: list[tuple[str, tuple[ModelDeploymentTopologyDatabaseGrant, ...]]] = []
    for database in topology.databases.values():
        for principal_name, principal in database.principals.items():
            pairs.append((principal_name, tuple(principal.grants)))
    return pairs
