# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""role_omnibase_infra holds write on infra_routing_decisions (OMN-21935).

omnimarket's node_projection_infra_routing_decisions declares a ``db_io``
write on ``public.infra_routing_decisions`` in the ``omnibase_infra`` database,
the relation migration 080 creates. The runtime refuses to wire a projection
whose binding principal lacks SELECT, INSERT and UPDATE on the table plus USAGE
on the schema, so the shipped topology must grant it in every shipped instance.
"""

from pathlib import Path

import pytest

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import TOPOLOGY_PROFILE_INSTANCE_MAP
from omnibase_infra.topology.table_grant_derivation import (
    STATE_IO_TABLE_DECLARATIONS,
    derive_table_grants,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TABLE = "infra_routing_decisions"
_PRINCIPAL = "role_omnibase_infra"
_MIGRATION = "docker/migrations/forward/080_create_infra_routing_decisions.sql"
_PROFILES = tuple(sorted(set(TOPOLOGY_PROFILE_INSTANCE_MAP.values())))


def _declaration():
    (declaration,) = (d for d in STATE_IO_TABLE_DECLARATIONS if d.table.name == _TABLE)
    return declaration


def test_the_migration_that_creates_the_relation_is_in_the_tree() -> None:
    assert (_REPO_ROOT / _MIGRATION).is_file()


def test_the_declaration_names_the_omnibase_infra_database() -> None:
    table = _declaration().table
    assert table.database_ref == "omnibase_infra"
    assert table.schema == "public"
    assert table.migration == _MIGRATION
    assert table.access == "read_write"


@pytest.mark.parametrize("profile", _PROFILES)
def test_the_shipped_instance_grants_select_insert_update(profile: str) -> None:
    topology = load_topology_profile(profile)
    principal = topology.databases["omnibase_infra"].principals[_PRINCIPAL]
    privileges = {
        p.value
        for grant in principal.grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == "public"
        and _TABLE in grant.objects
        for p in grant.privileges
    }
    assert privileges == {"INSERT", "SELECT", "UPDATE"}


@pytest.mark.parametrize("profile", _PROFILES)
def test_the_shipped_instance_grants_schema_usage(profile: str) -> None:
    topology = load_topology_profile(profile)
    principal = topology.databases["omnibase_infra"].principals[_PRINCIPAL]
    assert any(
        grant.object_type is EnumDatabaseGrantObjectType.SCHEMA
        and grant.schema == "public"
        and {p.value for p in grant.privileges} == {"USAGE"}
        for grant in principal.grants
    )


@pytest.mark.parametrize("profile", _PROFILES)
def test_the_runtime_binding_check_accepts_the_declaration(profile: str) -> None:
    topology = load_topology_profile(profile)
    target = _resolve_projection_database_target((_declaration().table,), topology)
    (table_target,) = target.table_targets
    assert table_target.write_binding is not None
    assert table_target.write_binding.principal == _PRINCIPAL
    assert table_target.read_binding is not None
    assert table_target.read_binding.principal == _PRINCIPAL


@pytest.mark.parametrize("profile", _PROFILES)
def test_no_other_omnibase_infra_principal_is_granted_it(profile: str) -> None:
    topology = load_topology_profile(profile)
    for name, principal in topology.databases["omnibase_infra"].principals.items():
        if name == _PRINCIPAL:
            continue
        assert not any(_TABLE in grant.objects for grant in principal.grants)


def _derived_tables(profile: str, declarations) -> set[str]:
    derived = derive_table_grants(
        load_topology_profile(profile), declarations, database_ref="omnibase_infra"
    )
    return {
        table
        for grants in derived.grants.values()
        for grant in grants
        for table in grant.objects
    }


@pytest.mark.parametrize("profile", _PROFILES)
def test_removing_only_the_declaration_removes_only_its_grant(profile: str) -> None:
    """Negative control: every other omnibase_infra declaration keeps its grant."""
    others = tuple(
        d
        for d in STATE_IO_TABLE_DECLARATIONS
        if d.table.database_ref == "omnibase_infra" and d.table.name != _TABLE
    )
    assert others
    with_it = _derived_tables(profile, (*others, _declaration()))
    without_it = _derived_tables(profile, others)
    assert _TABLE in with_it
    assert _TABLE not in without_it
    assert with_it - {_TABLE} == without_it
    assert without_it == {d.table.name for d in others}
