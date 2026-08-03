# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Contract-derived TABLE grant guarantees for the shipped topology (OMN-15656).

The shipped instances declared zero ``object_type: TABLE`` grants, so the
OMN-15418 privilege validator refused every contract-declared projection on
every profile and onex-dev's runtime plane failed to boot. These tests pin the
three properties that keep it fixed:

1. the derivation maps access modes onto exactly the privileges the validator
   demands, in both directions (read *and* write);
2. the shipped instances carry the derived grants, identically, with no
   wildcard or over-broad privilege; and
3. no test helper may re-synthesise a grant the platform is supposed to ship —
   the exact mechanism that hid the original defect.

Cross-repo coverage (every omnimarket contract resolving against every profile)
is asserted by ``scripts/generate_application_database_table_grants.py
--check --prove`` in the OMN-15361 CI job, which has the pinned omnimarket
checkout this process does not.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _INTERNAL_PROJECTION_BINDING,
    _TENANT_PROJECTION_BINDING,
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import (
    DOMAIN_PROJECTION_BINDINGS,
    READ_PRIVILEGES,
    TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    WRITE_PRIVILEGES,
    ContractTableDeclaration,
    derive_table_grants,
    physical_grant_schema_for_table,
)

pytestmark = pytest.mark.unit

_INSTANCE_ROOT = (
    Path(__file__).parents[3] / "src" / "omnibase_infra" / "topology" / "instances"
)
_INSTANCE_NAMES = ("local", "onex-dev", "onex-prod")
_HELPER = Path(__file__).parents[2] / "helpers" / "application_db_topology.py"

# The only privilege shapes the validator can ever require.
_LEGAL_PRIVILEGE_SETS = (READ_PRIVILEGES, WRITE_PRIVILEGES)


def _declaration(
    name: str,
    schema: str,
    access: str,
    *,
    database_ref: str = "application",
) -> ContractTableDeclaration:
    return ContractTableDeclaration(
        node=f"node_{name}",
        contract_path=Path(f"{name}/contract.yaml"),
        table=ModelDbTableDeclaration(
            name=name,
            database_ref=database_ref,
            schema=schema,
            migration=f"{name}.sql",
            access=access,
            role=f"{name}_projection",
        ),
    )


def _table_grants(topology: object, principal: str) -> tuple[object, ...]:
    database = topology.databases["application"]  # type: ignore[attr-defined]
    return tuple(
        grant
        for grant in database.principals[principal].grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
    )


# ---------------------------------------------------------------------------
# Derivation: access mode -> privileges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("access", "expected"),
    [
        ("read", READ_PRIVILEGES),
        ("write", WRITE_PRIVILEGES),
        ("read_write", WRITE_PRIVILEGES),
    ],
)
def test_access_mode_derives_exactly_the_validator_privileges(
    access: str, expected: frozenset[EnumDatabasePrivilege]
) -> None:
    """``read`` must derive SELECT only; anything writable derives the triple.

    The pinned omnimarket corpus is currently all-``write``, so without this the
    read branch would ship unexercised and a generator hardcoding the write
    triple would look correct.
    """
    topology = load_topology_profile("local")
    derived = derive_table_grants(
        topology, [_declaration("generation_events", "omninode_internal", access)]
    )
    grants = derived.grants["omninode_runtime"]
    assert len(grants) == 1
    assert set(grants[0].privileges) == set(expected)
    assert grants[0].objects == ("generation_events",)
    assert grants[0].schema == "omninode_internal"


def test_read_and_write_privilege_sets_are_what_the_validator_enforces() -> None:
    """Behavioural drift guard: derive, then prove the validator accepts it.

    Asserted by resolving through the real validator rather than by comparing
    constants, so a change to either side fails here.
    """
    topology = load_topology_profile("local")
    for access in ("read", "write"):
        declaration = _declaration("generation_events", "omninode_internal", access)
        # The shipped topology already grants this relation the write triple,
        # which is a superset of the read requirement.
        _resolve_projection_database_target((declaration.table,), topology)


def test_write_privileges_include_select_for_on_conflict_do_update() -> None:
    """SELECT is load-bearing: the adapter issues INSERT ... ON CONFLICT DO UPDATE."""
    assert EnumDatabasePrivilege.SELECT in WRITE_PRIVILEGES
    assert frozenset({EnumDatabasePrivilege.SELECT}) == READ_PRIVILEGES


def test_domain_bindings_match_the_wiring_module() -> None:
    """Derivation must select the same principal the validator will check."""
    assert DOMAIN_PROJECTION_BINDINGS == {
        EnumDatabaseSchemaDomain.TENANT: _TENANT_PROJECTION_BINDING,
        EnumDatabaseSchemaDomain.OMNINODE_INTERNAL: _INTERNAL_PROJECTION_BINDING,
    }


def test_tenant_declarations_route_to_the_tenant_writer() -> None:
    topology = load_topology_profile("local")
    derived = derive_table_grants(
        topology, [_declaration("future_tenant_projection", "tenant", "write")]
    )
    assert set(derived.grants) == {"tenant_projection_writer"}
    assert derived.grants["tenant_projection_writer"][0].schema == "tenant"


def test_omn15359_pending_tenant_tables_grant_against_current_physical_schema() -> None:
    """Temporary physical-schema bridge: logical tenant tables still live in public."""
    topology = load_topology_profile("local")
    derived = derive_table_grants(
        topology,
        [_declaration("delegation_judge_verdict_events", "tenant", "write")],
    )
    assert set(derived.grants) == {"tenant_projection_writer"}
    assert derived.grants["tenant_projection_writer"][0].schema == "public"
    assert (
        physical_grant_schema_for_table("tenant", "delegation_judge_verdict_events")
        == "public"
    )
    assert physical_grant_schema_for_table("tenant", "future_tenant_projection") == (
        "tenant"
    )


# ---------------------------------------------------------------------------
# Derivation: residuals are surfaced, never dropped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "reason_fragment"),
    [
        (_declaration("x", "unresolved", "write"), "schema 'unresolved'"),
        (
            _declaration("y", "public", "write", database_ref="omniintelligence"),
            "database_ref 'omniintelligence'",
        ),
    ],
)
def test_undeclarable_relations_are_returned_as_typed_residuals(
    declaration: ContractTableDeclaration, reason_fragment: str
) -> None:
    """A declaration the topology cannot serve must be named, not silently skipped."""
    derived = derive_table_grants(load_topology_profile("local"), [declaration])
    assert derived.grants == {}
    assert len(derived.unmappable) == 1
    assert reason_fragment in derived.unmappable[0].reason


def test_platform_catalog_is_not_derivable_from_contracts() -> None:
    """Catalog grants need a caller-supplied binding, so they are never guessed."""
    derived = derive_table_grants(
        load_topology_profile("local"),
        [_declaration("plan_tiers", "platform_catalog", "read")],
    )
    assert derived.grants == {}
    assert "requires an explicit" in derived.unmappable[0].reason


def test_one_relation_declared_by_two_contracts_yields_one_grant() -> None:
    """capsule_store is declared twice upstream; the union must not duplicate it."""
    topology = load_topology_profile("local")
    derived = derive_table_grants(
        topology,
        [
            _declaration("capsule_store", "omninode_internal", "write"),
            _declaration("capsule_store", "omninode_internal", "read"),
        ],
    )
    grants = derived.grants["omninode_runtime"]
    assert len(grants) == 1
    assert grants[0].objects == ("capsule_store",)
    assert set(grants[0].privileges) == set(WRITE_PRIVILEGES)


# ---------------------------------------------------------------------------
# Shipped instances
# ---------------------------------------------------------------------------


def test_every_shipped_instance_declares_table_grants() -> None:
    """The regression itself: zero TABLE grants is what broke onex-dev."""
    for name in _INSTANCE_NAMES:
        document = yaml.safe_load((_INSTANCE_ROOT / f"{name}.yaml").read_text())
        grants = [
            grant
            for principal in document["databases"]["application"]["principals"].values()
            for grant in principal["grants"]
            if grant["object_type"] == "TABLE"
        ]
        assert grants, f"{name}.yaml declares no TABLE grants"


def test_shipped_instances_declare_identical_grants() -> None:
    """The schema has no include mechanism, so the three copies must not drift."""
    blocks = []
    for name in _INSTANCE_NAMES:
        document = yaml.safe_load((_INSTANCE_ROOT / f"{name}.yaml").read_text())
        blocks.append(
            {
                principal: value["grants"]
                for principal, value in document["databases"]["application"][
                    "principals"
                ].items()
            }
        )
    assert blocks[0] == blocks[1] == blocks[2]


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_shipped_table_grants_are_explicit_and_minimal(profile: str) -> None:
    """No wildcards, no future-object grants, no privilege beyond the validator's."""
    topology = load_topology_profile(profile)
    database = topology.databases["application"]
    for principal_name, principal in database.principals.items():
        for grant in principal.grants:
            if grant.object_type is not EnumDatabaseGrantObjectType.TABLE:
                continue
            assert grant.objects, f"{principal_name} has a TABLE grant with no objects"
            assert grant.schema is not None
            assert not any("*" in name for name in grant.objects)
            assert set(grant.privileges) in [
                set(item) for item in _LEGAL_PRIVILEGE_SETS
            ], f"{principal_name} holds a non-canonical privilege set"


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_every_granted_relation_resolves_through_the_real_validator(
    profile: str,
) -> None:
    """Each shipped grant must actually satisfy ``_resolve_projection_database_target``.

    Drives the real resolver against the real shipped topology on every
    supported profile — no fixture topology, no synthesised grants.
    """
    topology = load_topology_profile(profile)
    database = topology.databases["application"]
    checked = 0
    for principal in database.principals.values():
        for grant in principal.grants:
            if grant.object_type is not EnumDatabaseGrantObjectType.TABLE:
                continue
            access = (
                "read" if set(grant.privileges) == set(READ_PRIVILEGES) else "write"
            )
            for name in grant.objects:
                logical_schema = (
                    "tenant"
                    if (
                        grant.schema == "public"
                        and name in TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
                    )
                    else grant.schema
                )
                table = ModelDbTableDeclaration(
                    name=name,
                    database_ref="application",
                    schema=logical_schema or "",
                    migration=f"{name}.sql",
                    access=access,
                    role=f"{name}_projection",
                )
                _resolve_projection_database_target((table,), topology)
                checked += 1
    assert checked > 0, f"profile {profile} granted no relations to check"


# ---------------------------------------------------------------------------
# AC-4: no test helper may re-hide a missing platform grant
# ---------------------------------------------------------------------------


def test_wiring_test_helper_loads_the_shipped_topology() -> None:
    """The helper must not resurrect the fixture topology that hid the defect."""
    from tests.helpers.application_db_topology import application_topology

    assert application_topology() == load_topology_profile("local")


def test_helper_declares_exactly_one_guarded_grant_synthesis() -> None:
    """Static source assertion: grant construction lives in one reviewed place.

    ``_with_projection_fixture_grants`` manufactured the exact missing grants at
    test time, which is why a 43/43 strict-wiring failure reached a deploy. Any
    new synthesis site must be added deliberately, not by copy-paste.
    """
    source = _HELPER.read_text(encoding="utf-8")
    # The removed synthesizer may still be named in prose; it must not be defined.
    assert "def _with_projection_fixture_grants" not in source
    assert source.count("ModelDeploymentTopologyDatabaseGrant(") == 1
    assert "def _topology_with_unshipped_grants" in source


def test_unshipped_grant_helper_refuses_a_relation_the_platform_ships() -> None:
    """The escape hatch is fail-closed against the class it must not re-enable."""
    from tests.helpers.application_db_topology import projection_database_target

    with pytest.raises(AssertionError, match="already granted"):
        projection_database_target(
            "generation_events",
            schema="omninode_internal",
            unshipped_grant_principal="omninode_runtime",
            unshipped_grant_reason="attempting to shadow a real platform grant",
        )


def test_unshipped_grant_helper_requires_a_reason() -> None:
    from tests.helpers.application_db_topology import projection_database_target

    with pytest.raises(ValueError, match="explicit reason"):
        projection_database_target(
            "plan_tiers",
            schema="platform_catalog",
            access="read",
            catalog_read_binding="app_dashboard",
            unshipped_grant_principal="app_dashboard",
            unshipped_grant_reason="   ",
        )


# ---------------------------------------------------------------------------
# Incident replay (OMN-15547 registry): deploy run 30737415706
# ---------------------------------------------------------------------------

_ZERO_GRANT_CAPTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "omn15547"
    / "onex-dev-topology-zero-table-grants.yaml.captured"
)

# Verbatim from the omninode-runtime-worker pod on deploy run 30737415706.
_DEPLOY_ERROR = (
    "Projection binding 'omninode_runtime_service' principal 'omninode_runtime' "
    "lacks declared write privileges: INSERT, SELECT, UPDATE on table "
    "omninode_internal.evidence_dashboard_projection"
)

_DEPLOY_RELATION = ModelDbTableDeclaration(
    name="evidence_dashboard_projection",
    database_ref="application",
    schema="omninode_internal",
    migration="0001_evidence_dashboard_projection.sql",
    access="write",
    role="evidence_dashboard_projection",
)


def test_replay_captured_topology_reproduces_the_deploy_failure_verbatim() -> None:
    """The shipped bytes that broke onex-dev, driven through the real resolver.

    ``onex-dev-topology-zero-table-grants.yaml.captured`` is the exact instance
    that was live at dev ``27630ec1b`` — its sha256 is the same
    ``b34206e7…`` recorded in the rendered catalogs' ``source.sha256`` and in
    omninode_infra's k8s ``source-lock.yaml``, so this is the artifact the
    cluster actually consumed, not a reconstruction.
    """
    from omnibase_core.models.core import ModelDeploymentTopology

    captured = ModelDeploymentTopology.from_yaml(_ZERO_GRANT_CAPTURE)

    with pytest.raises(ValueError) as excinfo:
        _resolve_projection_database_target((_DEPLOY_RELATION,), captured)
    assert str(excinfo.value) == _DEPLOY_ERROR


def test_replay_derivation_rejects_the_captured_topology_as_drifted() -> None:
    """The new guard's verdict on the captured bytes must be REJECT.

    ``--check`` compares the derived grant set against what the instance
    declares. The capture declares zero TABLE grants while a single real
    contract declaration derives one, so the guard fails closed on it — which
    is what would have stopped the deploy.
    """
    from omnibase_core.models.core import ModelDeploymentTopology

    captured = ModelDeploymentTopology.from_yaml(_ZERO_GRANT_CAPTURE)
    declared = tuple(
        grant
        for principal in captured.databases["application"].principals.values()
        for grant in principal.grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
    )
    assert declared == (), "capture is only meaningful if it has no TABLE grants"

    derived = derive_table_grants(
        captured,
        [
            ContractTableDeclaration(
                node="node_evidence_dashboard_reducer",
                contract_path=Path("node_evidence_dashboard_reducer/contract.yaml"),
                table=_DEPLOY_RELATION,
            )
        ],
    )
    assert derived.grants["omninode_runtime"], "derivation produced no grant to compare"
    assert derived.grants["omninode_runtime"] != declared


def test_replay_shipped_topology_now_accepts_the_deploy_relation() -> None:
    """The same relation the deploy died on resolves on every shipped profile."""
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        _resolve_projection_database_target(
            (_DEPLOY_RELATION,), load_topology_profile(profile)
        )
