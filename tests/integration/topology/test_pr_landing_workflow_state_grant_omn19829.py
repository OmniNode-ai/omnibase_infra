# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ``pr_landing_workflow_state`` state_io table derives its shipped GRANT (OMN-19829).

Migration 108 creates ``pr_landing_workflow_state``, the durable ``state_io``
row omnimarket's ``node_pr_landing_orchestrator`` reads and CAS-writes on every
leg, keyed on ``landing_key``. ``state_io`` predates
:class:`~omnibase_core.models.contracts.subcontracts.model_db_table_declaration.ModelDbTableDeclaration`
and has no core model of its own, so the ONLY way this relation participates
in the OMN-15656 grant derivation at all is the checked-in
``STATE_IO_TABLE_DECLARATIONS`` entry -- the same seam ``delegation_workflow_state``
(090) and ``session_phase_state`` (102) already use.

``tests/unit/db/test_migration_108_pr_landing_workflow_state_omn19829.py`` pins
the migration shape, the adapter's identifier guard and that the table NAME
appears in the shipped instance YAML text. None of that runs
``derive_table_grants`` itself, so a declaration whose ``database_ref`` or
``schema`` the derivation cannot route -- the exact "declared but never
granted" shape OMN-15656 exists to prevent -- would pass every unit assertion
here while shipping zero TABLE grants. This module closes that gap:

1. the migration that creates the relation is in the tree;
2. the shipped instance actually GRANTS it (not merely mentions its name);
3. the derivation maps the declaration onto exactly the same principal as its
   ``session_phase_state`` sibling, with exactly the write privileges the
   OMN-15418 validator demands; and
4. removing the declaration removes the grant -- the negative control, since
   deleting it and regenerating the instances to match would pass (1)-(3) on
   a tree with zero coverage of the direction that matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    STATE_IO_TABLE_DECLARATIONS,
    WRITE_PRIVILEGES,
    ContractTableDeclaration,
    derive_table_grants,
    physical_grant_schema_for_table,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TABLE = "pr_landing_workflow_state"
_SIBLING_TABLE = "session_phase_state"
_DATABASE_REF = "omnibase_infra"
_MIGRATION = Path("docker/migrations/forward/108_create_pr_landing_workflow_state.sql")

# The profiles the migration-108 unit test itself pins as shipped.
_PROFILES = ("local", "onex-dev", "onex-prod")

_DECLARATION = next(d for d in STATE_IO_TABLE_DECLARATIONS if d.table.name == _TABLE)
_SIBLING_DECLARATION = next(
    d for d in STATE_IO_TABLE_DECLARATIONS if d.table.name == _SIBLING_TABLE
)


def _instance_text(profile: str) -> str:
    return (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    ).read_text(encoding="utf-8")


def _granted_principals(
    profile: str, table: str, declarations: tuple[ContractTableDeclaration, ...]
) -> set[str]:
    topology = load_topology_profile(profile)
    derived = derive_table_grants(
        topology,
        declarations,
        database_ref=_DATABASE_REF,
    )
    physical_schema = physical_grant_schema_for_table("public", table)
    return {
        principal
        for principal, grants in derived.grants.items()
        for grant in grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == physical_schema
        and table in grant.objects
    }


def test_the_migration_is_in_the_tree() -> None:
    """A grant for a relation nothing creates is a dangling grant."""
    assert (_REPO_ROOT / _MIGRATION).is_file()


@pytest.mark.parametrize("profile", _PROFILES)
def test_the_shipped_instance_declares_it(profile: str) -> None:
    """The regression, stated as the thing a reader can check by eye.

    The sibling ``session_phase_state`` is granted in the same file as a
    positive control: an instance that lost both would pass a bare "the
    string session_phase_state is absent too" reading as unrelated drift
    rather than a state_io regression.
    """
    document = yaml.safe_load(_instance_text(profile))
    declared = yaml.safe_dump(document)
    assert _SIBLING_TABLE in declared
    assert _TABLE in declared, (
        f"{profile} does not declare {_TABLE}, but migration 108 creates it "
        "and node_pr_landing_orchestrator's state_io seam reads/writes it on "
        "every leg; an ungranted relation fails the runtime's OMN-15418 "
        "privilege check at wiring time"
    )


@pytest.mark.parametrize("profile", _PROFILES)
def test_the_derivation_grants_the_same_principal_as_its_sibling(
    profile: str,
) -> None:
    principals = _granted_principals(profile, _TABLE, (_DECLARATION,))
    sibling_principals = _granted_principals(
        profile, _SIBLING_TABLE, (_SIBLING_DECLARATION,)
    )
    assert sibling_principals, (
        "positive control: the sibling state_io declaration derived no "
        "principal at all, which would make an equality assertion vacuous"
    )
    assert principals == sibling_principals, (
        f"profile {profile} derives {sorted(principals)} for {_TABLE} but "
        f"{sorted(sibling_principals)} for its {_SIBLING_TABLE} sibling; both "
        "are the same state_io seam and must resolve to the same runtime "
        "principal"
    )


@pytest.mark.parametrize("profile", _PROFILES)
def test_privileges_are_exactly_what_the_validator_demands(profile: str) -> None:
    """No wildcard, no widening. Over-granting is its own defect."""
    topology = load_topology_profile(profile)
    derived = derive_table_grants(topology, (_DECLARATION,), database_ref=_DATABASE_REF)
    physical_schema = physical_grant_schema_for_table("public", _TABLE)
    seen = 0
    for grants in derived.grants.values():
        for grant in grants:
            if grant.object_type is not EnumDatabaseGrantObjectType.TABLE:
                continue
            if grant.schema != physical_schema or _TABLE not in grant.objects:
                continue
            assert set(grant.privileges) == WRITE_PRIVILEGES
            seen += 1
    assert seen, "no TABLE grant on the relation at all; the check is vacuous"


@pytest.mark.parametrize("profile", _PROFILES)
def test_an_empty_declaration_set_derives_no_grant(profile: str) -> None:
    """The negative control, and the reason the one-line fix is wrong.

    Without this, the suite above would pass just as happily on a tree where
    the declaration was deleted and the instances regenerated to match --
    exactly the "declared but never granted" shape OMN-15656 exists to
    prevent. This asserts the causal direction: the grant exists BECAUSE
    ``STATE_IO_TABLE_DECLARATIONS`` carries the entry.
    """
    principals = _granted_principals(profile, _TABLE, ())
    assert principals == set()
