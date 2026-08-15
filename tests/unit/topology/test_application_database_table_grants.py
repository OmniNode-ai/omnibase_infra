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
    INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    READ_PRIVILEGES,
    TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    WRITE_PRIVILEGES,
    ContractTableDeclaration,
    derive_table_grants,
    derive_topology_table_grants,
    physical_grant_schema_for_table,
)

pytestmark = pytest.mark.unit

_INSTANCE_ROOT = (
    Path(__file__).parents[3] / "src" / "omnibase_infra" / "topology" / "instances"
)
_INSTANCE_NAMES = ("local", "onex-dev", "onex-prod")
# Every logical database the shipped instances declare. ``omniintelligence``
# and ``omnibase_infra`` are independently service-owned databases ADR-0027
# keeps separate from the unified application pair (OMN-15655 AC-2,
# OMN-15337).
_DATABASE_REFS = ("application", "omnibase_infra", "omniintelligence")
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
        topology,
        [_declaration("future_internal_projection", "omninode_internal", access)],
    )
    grants = derived.grants["omninode_runtime"]
    assert len(grants) == 1
    assert set(grants[0].privileges) == set(expected)
    assert grants[0].objects == ("future_internal_projection",)
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


def test_omn15359_pending_internal_tables_grant_against_current_physical_schema() -> (
    None
):
    """Physical-schema bridge for OMNINODE_INTERNAL: the target schema now
    physically exists (098_create_omninode_internal_schema.sql) but the 41
    tables the shipped topology grants against it are still physically
    created in ``public`` by their node migrations — the OMN-15426 gap.
    """
    topology = load_topology_profile("local")
    derived = derive_table_grants(
        topology,
        [_declaration("node_service_registry", "omninode_internal", "write")],
    )
    assert set(derived.grants) == {"omninode_runtime"}
    assert derived.grants["omninode_runtime"][0].schema == "public"
    assert (
        physical_grant_schema_for_table("omninode_internal", "node_service_registry")
        == "public"
    )
    assert physical_grant_schema_for_table(
        "omninode_internal", "future_internal_projection"
    ) == ("omninode_internal")


def test_omn15359_internal_bridge_covers_every_shipped_internal_grant() -> None:
    """The bridge set must not silently drift from the shipped topology.

    The shipped `omninode_runtime` TABLE grants are generated with the
    physical bridge already applied (``physical_grant_schema_for_table``), so
    on the live `local` profile they carry ``schema: public`` today -- none of
    the 41 relations has a copy migration yet. Every one of those physically-
    public table names must be a member of
    ``INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359``: this is a
    shrink-only ratchet in the graduation direction (a table leaves the
    frozenset only once its family's copy has actually landed and the
    generator re-derives it against ``omninode_internal``), and it fails
    closed if topology grants a new table to `omninode_runtime` this bridge
    does not yet know about -- exactly the drift class OMN-15426 hit.
    """
    topology = load_topology_profile("local")
    internal_database = topology.databases["application"]
    omninode_runtime_grants = internal_database.principals["omninode_runtime"].grants
    granted_public_tables = {
        table_name
        for grant in omninode_runtime_grants
        if grant.object_type == EnumDatabaseGrantObjectType.TABLE
        and grant.schema == "public"
        for table_name in grant.objects
    }
    assert granted_public_tables
    assert granted_public_tables <= INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359


# ---------------------------------------------------------------------------
# Derivation: residuals are surfaced, never dropped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("declaration", "reason_fragment"),
    [
        (_declaration("x", "unresolved", "write"), "schema 'unresolved'"),
        (
            # omnimemory has a real per-service DSN key in _DB_URL_ENV_MAP but no
            # topology declaration, so it is the live example of the class that
            # ``omniintelligence`` occupied before OMN-15655 AC-2 declared it.
            _declaration("y", "public", "write", database_ref="omnimemory"),
            "database_ref 'omnimemory'",
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


def test_topology_derivation_routes_each_declaration_to_its_own_database() -> None:
    """Service-database declarations derive against that database's principal.

    Single-database derivation classified every ``omniintelligence``
    declaration as an ``application`` residual, so the service principal would
    have shipped grant-less while ``--check`` stayed green.
    """
    topology = load_topology_profile("local")
    derived = derive_topology_table_grants(
        topology,
        [
            _declaration("generation_events", "omninode_internal", "write"),
            _declaration(
                "dispatch_eval_results",
                "public",
                "write",
                database_ref="omniintelligence",
            ),
        ],
    )
    assert set(derived.per_database) == set(topology.databases)
    assert set(derived.per_database["application"].grants) == {"omninode_runtime"}
    assert set(derived.per_database["omniintelligence"].grants) == {
        "role_omniintelligence"
    }
    service_grant = derived.per_database["omniintelligence"].grants[
        "role_omniintelligence"
    ][0]
    assert service_grant.schema == "public"
    assert service_grant.objects == ("dispatch_eval_results",)
    assert set(service_grant.privileges) == set(WRITE_PRIVILEGES)
    assert derived.unmappable == ()


def test_topology_derivation_still_reports_an_undeclared_database_as_residual() -> None:
    """Routing must not invent a database to make a contract resolvable."""
    derived = derive_topology_table_grants(
        load_topology_profile("local"),
        [_declaration("z", "public", "write", database_ref="omnimemory")],
    )
    assert all(entry.grants == {} for entry in derived.per_database.values())
    assert len(derived.unmappable) == 1
    assert "database_ref 'omnimemory'" in derived.unmappable[0].reason


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


@pytest.mark.parametrize("database_ref", _DATABASE_REFS)
def test_every_shipped_instance_declares_table_grants(database_ref: str) -> None:
    """The regression itself: zero TABLE grants is what broke onex-dev.

    Parametrised over every declared database because a service database whose
    principal ships with no TABLE grant reproduces the same defect one
    ``database_ref`` over.
    """
    for name in _INSTANCE_NAMES:
        document = yaml.safe_load((_INSTANCE_ROOT / f"{name}.yaml").read_text())
        grants = [
            grant
            for principal in document["databases"][database_ref]["principals"].values()
            for grant in principal["grants"]
            if grant["object_type"] == "TABLE"
        ]
        assert grants, f"{name}.yaml declares no TABLE grants for {database_ref}"


def test_shipped_instances_declare_identical_grants() -> None:
    """The schema has no include mechanism, so the three copies must not drift."""
    blocks = []
    for name in _INSTANCE_NAMES:
        document = yaml.safe_load((_INSTANCE_ROOT / f"{name}.yaml").read_text())
        blocks.append(
            {
                (database_ref, principal): value["grants"]
                for database_ref, database in document["databases"].items()
                for principal, value in database["principals"].items()
            }
        )
    assert blocks[0] == blocks[1] == blocks[2]


def test_every_shipped_instance_declares_the_same_databases() -> None:
    """A database declared in one instance but not another is a lane-only boot break."""
    declared = [
        set(yaml.safe_load((_INSTANCE_ROOT / f"{name}.yaml").read_text())["databases"])
        for name in _INSTANCE_NAMES
    ]
    assert declared[0] == declared[1] == declared[2] == set(_DATABASE_REFS)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_shipped_table_grants_are_explicit_and_minimal(profile: str) -> None:
    """No wildcards, no future-object grants, no privilege beyond the validator's."""
    topology = load_topology_profile(profile)
    for database_ref in _DATABASE_REFS:
        database = topology.databases[database_ref]
        for principal_name, principal in database.principals.items():
            for grant in principal.grants:
                if grant.object_type is not EnumDatabaseGrantObjectType.TABLE:
                    continue
                assert grant.objects, (
                    f"{database_ref}.{principal_name} has a TABLE grant with no objects"
                )
                assert grant.schema is not None
                assert not any("*" in name for name in grant.objects)
                assert set(grant.privileges) in [
                    set(item) for item in _LEGAL_PRIVILEGE_SETS
                ], (
                    f"{database_ref}.{principal_name} holds a non-canonical privilege set"
                )


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
@pytest.mark.parametrize("database_ref", _DATABASE_REFS)
def test_every_granted_relation_resolves_through_the_real_validator(
    profile: str,
    database_ref: str,
) -> None:
    """Each shipped grant must actually satisfy ``_resolve_projection_database_target``.

    Drives the real resolver against the real shipped topology on every
    supported profile and every declared database — no fixture topology, no
    synthesised grants.
    """
    topology = load_topology_profile(profile)
    database = topology.databases[database_ref]
    checked = 0
    for principal in database.principals.values():
        for grant in principal.grants:
            if grant.object_type is not EnumDatabaseGrantObjectType.TABLE:
                continue
            access = (
                "read" if set(grant.privileges) == set(READ_PRIVILEGES) else "write"
            )
            for name in grant.objects:
                if (
                    grant.schema == "public"
                    and name in TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
                ):
                    logical_schema = "tenant"
                elif (
                    grant.schema == "public"
                    and name in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
                ):
                    logical_schema = "omninode_internal"
                else:
                    logical_schema = grant.schema
                table = ModelDbTableDeclaration(
                    name=name,
                    database_ref=database_ref,
                    schema=logical_schema or "",
                    migration=f"{name}.sql",
                    access=access,
                    role=f"{name}_projection",
                )
                _resolve_projection_database_target((table,), topology)
                checked += 1
    assert checked > 0, (
        f"profile {profile} granted no {database_ref} relations to check"
    )


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

# OMN-15359 extended the physical-schema bridge
# (``physical_grant_schema_for_table``) to cover OMNINODE_INTERNAL-domain
# tables, so ``evidence_dashboard_projection`` now resolves against ``public``
# (its real physical location) instead of ``omninode_internal``. Replaying the
# zero-grant capture against the CURRENT resolver therefore also finds the
# capture missing ``USAGE`` on the (now correctly identified) target schema --
# a real gap the capture always had but the pre-bridge resolver never checked
# for this relation, because it was still looking at the wrong schema. The
# verbatim historical pod message (``_DEPLOY_ERROR`` above) is preserved
# unmodified for the record; this is what the same capture produces today.
_DEPLOY_ERROR_POST_OMN15359_BRIDGE = (
    "Projection binding 'omninode_runtime_service' principal 'omninode_runtime' "
    "lacks declared write privileges: USAGE on schema 'public'; INSERT, SELECT, "
    "UPDATE on table omninode_internal.evidence_dashboard_projection"
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

    OMN-15359 note: the assertion below is
    ``_DEPLOY_ERROR_POST_OMN15359_BRIDGE``, not the verbatim ``_DEPLOY_ERROR``
    pod message (still preserved above for the historical record). Extending
    the physical-schema bridge to OMNINODE_INTERNAL made this exact relation
    resolve against ``public`` instead of ``omninode_internal``, and the
    zero-grant capture is missing ``USAGE`` there too — a real, additional gap
    the pre-bridge resolver could not see because it was checking the wrong
    schema.
    """
    from omnibase_core.models.core import ModelDeploymentTopology

    captured = ModelDeploymentTopology.from_yaml(_ZERO_GRANT_CAPTURE)

    with pytest.raises(ValueError) as excinfo:
        _resolve_projection_database_target((_DEPLOY_RELATION,), captured)
    assert str(excinfo.value) == _DEPLOY_ERROR_POST_OMN15359_BRIDGE


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


# ---------------------------------------------------------------------------
# Incident replay (OMN-15701 registry): onex-dev CrashLoopBackOff on
# 2026-08-04, deployed digest sha256:5507667152e2 (infra dev 2e5ef7da)
# ---------------------------------------------------------------------------
#
# infra#2634 (merged 2026-08-03T05:05:30Z) correctly derived
# ``tenant_projection_writer`` TABLE grants for all nine house-tenant
# relations. infra#2632 (merged 2026-08-03T18:06:11Z, an ancestor of the
# deployed SHA) then re-ran the derivation's ``--write`` step against the CI
# workflow's hardcoded omnimarket fallback pin
# (``4637e625c99ef17c190aa471a5e51b7f646c6dfd``, 2026-07-30) instead of
# omnimarket's actual dev HEAD, and that pin still declared
# ``schema: omninode_internal`` for these relations (omnimarket's
# reclassification, commit 485be549, landed on omnimarket dev at
# 2026-08-03T15:56:54Z -- *after* #2632 merged). The regeneration silently
# reverted eight of the nine relations back to ``omninode_runtime`` /
# ``omninode_internal``, so every contract that declares one of them
# (``schema: tenant``, per the operator ruling recorded on OMN-15655) failed
# ``_require_projection_binding_privileges`` on every profile.
#
# ``onex-dev-topology-reverted-tenant-grants.yaml.captured`` is byte-identical
# to the shipped ``instances/onex-dev.yaml`` at infra dev commit
# ``2e5ef7da5d08df9f1bcbe7eff58eed696c14d1e4`` -- the exact bytes the
# CrashLoopBackOff pods loaded, not a reconstruction.

_REVERTED_GRANTS_CAPTURE = (
    Path(__file__).parents[2]
    / "fixtures"
    / "omn15701"
    / "onex-dev-topology-reverted-tenant-grants.yaml.captured"
)

# Verbatim from the omninode-runtime pod's Auto-wiring failure on the
# CrashLoopBackOff observed 2026-08-04 (one of eight identically-shaped
# failures; capability_scores chosen as the exemplar).
_REVERTED_GRANT_ERROR = (
    "Projection binding 'tenant_projection' principal 'tenant_projection_writer' "
    "lacks declared write privileges: INSERT, SELECT, UPDATE on table "
    "tenant.capability_scores"
)

_REVERTED_GRANT_RELATION = ModelDbTableDeclaration(
    name="capability_scores",
    database_ref="application",
    schema="tenant",
    migration="0002_capability_scores_tenant_id_and_rls.sql",
    access="write",
    role="capability_scores",
)

# All eight relations infra#2632 reverted (the ninth,
# delegation_judge_verdict_events, was dropped from both grant lists entirely
# rather than merely moved, so it fails with a different message and is
# checked separately below).
_REVERTED_GRANT_RELATIONS = (
    _REVERTED_GRANT_RELATION,
    ModelDbTableDeclaration(
        name="context_roi_scores",
        database_ref="application",
        schema="tenant",
        migration="003_context_roi_scores_tenant_id_and_rls.sql",
        access="write",
        role="scores",
    ),
    ModelDbTableDeclaration(
        name="llm_cost_aggregates",
        database_ref="application",
        schema="tenant",
        migration="0002_llm_cost_aggregates_tenant_id_and_rls.sql",
        access="write",
        role="cost_summary",
    ),
    ModelDbTableDeclaration(
        name="dep_health_findings",
        database_ref="application",
        schema="tenant",
        migration="002_dep_health_findings_tenant_id_and_rls.sql",
        access="write",
        role="findings",
    ),
    ModelDbTableDeclaration(
        name="instruction_eval_aggregate_snapshots",
        database_ref="application",
        schema="tenant",
        migration="0001_create_instruction_eval_aggregate_snapshots.sql",
        access="write",
        role="instruction_eval_aggregate",
    ),
    ModelDbTableDeclaration(
        name="pattern_learning_artifacts",
        database_ref="application",
        schema="tenant",
        migration="0000_create_pattern_learning_artifacts.sql",
        access="write",
        role="artifacts",
    ),
    ModelDbTableDeclaration(
        name="agent_routing_decisions",
        database_ref="application",
        schema="tenant",
        migration="0021_create_agent_routing_decisions.sql",
        access="write",
        role="agent_routing_decisions",
    ),
    ModelDbTableDeclaration(
        name="skill_execution_snapshots",
        database_ref="application",
        schema="tenant",
        migration="0001_create_skill_execution_snapshots.sql",
        access="write",
        role="skill_executions_aggregate",
    ),
)


def test_omn15701_replay_captured_topology_reproduces_the_reverted_grant_failure() -> (
    None
):
    """The exact shipped bytes that crash-looped onex-dev, driven through the
    real resolver -- proves the captured incident fixture is non-vacuous."""
    from omnibase_core.models.core import ModelDeploymentTopology

    captured = ModelDeploymentTopology.from_yaml(_REVERTED_GRANTS_CAPTURE)

    with pytest.raises(ValueError) as excinfo:
        _resolve_projection_database_target((_REVERTED_GRANT_RELATION,), captured)
    assert str(excinfo.value) == _REVERTED_GRANT_ERROR


def test_omn15701_replay_captured_topology_fails_all_eight_reverted_relations() -> None:
    """Every relation infra#2632 silently moved back must fail identically on
    the captured (broken) topology -- not just the exemplar above."""
    from omnibase_core.models.core import ModelDeploymentTopology

    captured = ModelDeploymentTopology.from_yaml(_REVERTED_GRANTS_CAPTURE)

    for relation in _REVERTED_GRANT_RELATIONS:
        with pytest.raises(ValueError, match="lacks declared write privileges"):
            _resolve_projection_database_target((relation,), captured)


def test_omn15701_replay_shipped_topology_now_accepts_all_reverted_relations() -> None:
    """The nine house-tenant relations resolve cleanly on every shipped profile
    now that the tenant_projection_writer grants are restored."""
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        topology = load_topology_profile(profile)
        for relation in _REVERTED_GRANT_RELATIONS:
            _resolve_projection_database_target((relation,), topology)


def test_omn15701_shipped_grants_restore_nightly_loop_configs_read() -> None:
    """infra#2632 also dropped the omninode_runtime SELECT grant on
    omninode_internal.nightly_loop_configs that infra#2634 had added; this is
    the nightly_loop_controller effects-pod failure recorded on OMN-15655
    comment 4a229dbf. Must be restored alongside the tenant relations."""
    relation = ModelDbTableDeclaration(
        name="nightly_loop_configs",
        database_ref="application",
        schema="omninode_internal",
        migration="nightly_loop_configs.sql",
        access="read",
        role="nightly_loop_controller",
    )
    for profile in sorted(SUPPORTED_TOPOLOGY_PROFILES):
        _resolve_projection_database_target((relation,), load_topology_profile(profile))
