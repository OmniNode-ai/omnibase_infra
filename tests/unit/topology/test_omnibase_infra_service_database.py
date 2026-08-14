# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shipped-topology proof for the omnibase_infra runtime database (OMN-15337).

``delegation_workflow_state`` (``docker/migrations/forward/090_create_delegation_workflow_state.sql``,
OMN-14208 durable delegation FSM state) self-declared tenant-scoped in its
migration header and carries a bare ``tenant_id TEXT NOT NULL`` column, yet
has zero RLS, zero policy, and zero grant anywhere in the migration corpus.
It lives in the flat ``docker/migrations/forward/*.sql`` set, which targets
the ``omnibase_infra`` physical database
(``docker/docker-compose.infra.yml`` ``POSTGRES_DB=omnibase_infra``) --
a database the deployment topology never declared, so the isolation
mechanism this table's header implies does not exist anywhere it could
attach to.

Operator ruling R-q (2026-08-05, recorded in the rolling ledger) classifies
``delegation_workflow_state`` ``OMNINODE_INTERNAL``: the table is
orchestration state (the runtime's own FSM working set), the bare ``tenant_id``
column is denormalized *provenance* extracted from the opaque FSM payload
(never an authorization key), and FORCE-RLS on the runtime's own state store
would break non-tenant-context FSM access (recovery, retries, staleness
sweeps). The column is retained; no RLS or tenant-scoped grant is added.

Everything here drives the **shipped** ``load_topology_profile(profile)``
against the real resolver, the same way
``test_omniintelligence_service_database.py`` (OMN-15655 AC-2) proves its own
service database. Unlike that database, ``delegation_workflow_state`` is
consumed through the legacy ``state_io`` contract subcontract
(``handler_wiring._read_state_io``), not ``db_io.db_tables`` -- it therefore
has no upstream node contract for ``load_contract_declarations`` to discover,
so it is declared instead in the checked-in
``table_grant_derivation.STATE_IO_TABLE_DECLARATIONS`` manifest and proven
here against the same resolver ``db_io``-declared relations use.
"""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _DB_URL_ENV_MAP,
    _INTERNAL_PROJECTION_BINDING,
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import (
    OMNIBASE_INFRA_DATABASE_REF,
    SUPPORTED_TOPOLOGY_PROFILES,
    validate_omnibase_infra_database_invariants,
)
from omnibase_infra.topology.table_grant_derivation import (
    STATE_IO_TABLE_DECLARATIONS,
    derive_table_grants,
    derive_topology_table_grants,
)

pytestmark = pytest.mark.unit

# The one declared state_io relation this manifest carries today.
(_STATE_IO_DECLARATION,) = STATE_IO_TABLE_DECLARATIONS
_DELEGATION_WORKFLOW_STATE_RELATION = _STATE_IO_DECLARATION.table


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_delegation_workflow_state_relation_resolves_on_every_shipped_profile(
    profile: str,
) -> None:
    """The declared state_io relation resolves through the real resolver.

    Proves the same machinery ``db_io``-declared tables rely on
    (``_resolve_projection_database_target``) already accepts this relation,
    even though the runtime's own state_io read/write path does not call it
    yet -- the topology declaration this ticket adds is necessary but not
    sufficient for the runtime seam itself to consume it.
    """
    target = _resolve_projection_database_target(
        (_DELEGATION_WORKFLOW_STATE_RELATION,), load_topology_profile(profile)
    )
    assert target.physical_database == "omnibase_infra"
    assert target.database_refs == (OMNIBASE_INFRA_DATABASE_REF,)
    assert target.schemas == ("public",)
    assert target.domains == (EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_delegation_workflow_state_resolves_to_the_service_binding_and_dsn_key(
    profile: str,
) -> None:
    """A read_write access must select the internal binding, role, and DSN key.

    Asserted rather than assumed: resolving to any other pool would connect
    to the wrong physical database while still reporting a green boot.
    """
    target = _resolve_projection_database_target(
        (_DELEGATION_WORKFLOW_STATE_RELATION,), load_topology_profile(profile)
    )
    (binding,) = target.bindings
    assert binding.binding_ref == _INTERNAL_PROJECTION_BINDING
    assert binding.principal == "role_omnibase_infra"
    assert binding.physical_database == "omnibase_infra"
    assert binding.dsn_env == "OMNIBASE_INFRA_DB_URL"
    assert target.dsn_envs == ("OMNIBASE_INFRA_DB_URL",)


def test_declared_dsn_key_matches_the_per_service_db_url_contract() -> None:
    """The topology DSN key must be the one wiring and docs already agree on.

    ``docs/patterns/db_url_contract.md`` documents ``OMNIBASE_INFRA_DB_URL`` /
    ``omnibase_infra`` / ``role_omnibase_infra`` as the per-service contract
    row; this is the first typed, validated home for that row.
    """
    assert _DB_URL_ENV_MAP["omnibase_infra"] == "OMNIBASE_INFRA_DB_URL"


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_omnibase_infra_is_physically_separate_from_the_application_pair(
    profile: str,
) -> None:
    """ADR-0027 unified the application pair and kept service databases separate."""
    topology = load_topology_profile(profile)
    assert (
        topology.databases[OMNIBASE_INFRA_DATABASE_REF].physical_name
        == "omnibase_infra"
    )
    assert topology.databases["application"].physical_name == "omnidash_analytics"


def test_delegation_workflow_state_is_internal_despite_its_tenant_id_column() -> None:
    """OMNINODE_INTERNAL is a ruled classification (R-q), not the absence-of-column
    default the omniintelligence sibling test uses.

    Unlike ``dispatch_eval_results``, this relation DOES carry a ``tenant_id``
    column -- that column is exactly why the table's migration header
    self-declared it tenant-scoped and why OMN-15337 filed a gap. Operator
    ruling R-q settled it internal anyway: the column is denormalized
    provenance extracted from the opaque FSM payload by the state_io wiring
    seam, never an authorization key, and this table is the runtime's own
    FSM working state (recovery/retries/staleness sweeps need
    non-tenant-context access a FORCE-RLS policy would break).
    """
    topology = load_topology_profile("local")
    assert (
        topology.table_domain(_DELEGATION_WORKFLOW_STATE_RELATION)
        is EnumDatabaseSchemaDomain.OMNINODE_INTERNAL
    )


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_shipped_profiles_pass_the_omnibase_infra_database_invariants(
    profile: str,
) -> None:
    """``load_topology_profile`` already enforces these; assert the guard exists."""
    validate_omnibase_infra_database_invariants(load_topology_profile(profile))


def test_invariants_reject_a_repointed_physical_database() -> None:
    """Silently repointing the runtime database must fail the loader, not the pod."""
    topology = load_topology_profile("local")
    database = topology.databases[OMNIBASE_INFRA_DATABASE_REF]
    drifted = topology.model_copy(
        update={
            "databases": {
                **topology.databases,
                OMNIBASE_INFRA_DATABASE_REF: database.model_copy(
                    update={"physical_name": "omnidash_analytics"}
                ),
            }
        }
    )
    with pytest.raises(ValueError, match="must resolve to 'omnibase_infra'"):
        validate_omnibase_infra_database_invariants(drifted)


# ---------------------------------------------------------------------------
# Grant derivation: the RED discriminator this ticket closes.
#
# ``load_contract_declarations`` can only ever discover ``db_io.db_tables``
# entries. Without ``STATE_IO_TABLE_DECLARATIONS`` folded into the derivation
# input, ``delegation_workflow_state`` derives zero TABLE grants for the
# ``omnibase_infra`` database on every profile -- the same "declared but
# never granted" shape OMN-15656 exists to prevent. These two tests pin both
# sides of that fix.
# ---------------------------------------------------------------------------


def test_without_the_state_io_manifest_the_relation_derives_no_grant() -> None:
    """RED: contract-only derivation cannot see a state_io-owned relation."""
    topology = load_topology_profile("local")
    derived = derive_topology_table_grants(topology, ())
    assert derived.per_database[OMNIBASE_INFRA_DATABASE_REF].grants == {}


def test_state_io_manifest_derives_the_shipped_grant() -> None:
    """GREEN: the checked-in manifest recovers the grant without a contract."""
    topology = load_topology_profile("local")
    derived = derive_table_grants(
        topology,
        STATE_IO_TABLE_DECLARATIONS,
        database_ref=OMNIBASE_INFRA_DATABASE_REF,
    )
    assert derived.unmappable == ()
    (grant,) = derived.grants["role_omnibase_infra"]
    assert grant.schema == "public"
    assert grant.objects == ("delegation_workflow_state",)
    assert set(grant.privileges) == {
        EnumDatabasePrivilege.SELECT,
        EnumDatabasePrivilege.INSERT,
        EnumDatabasePrivilege.UPDATE,
    }


def test_shipped_grant_matches_the_derivation_exactly() -> None:
    """The checked-in instance TABLE grant is not hand-drifted from derivation."""
    topology = load_topology_profile("local")
    derived = derive_table_grants(
        topology,
        STATE_IO_TABLE_DECLARATIONS,
        database_ref=OMNIBASE_INFRA_DATABASE_REF,
    )
    shipped = tuple(
        grant
        for grant in topology.databases[OMNIBASE_INFRA_DATABASE_REF]
        .principals["role_omnibase_infra"]
        .grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
    )
    assert shipped == derived.grants["role_omnibase_infra"]


# ---------------------------------------------------------------------------
# The original OMN-15337 finding, now proven intentional rather than missing.
# ---------------------------------------------------------------------------


def test_no_rls_or_policy_exists_for_delegation_workflow_state() -> None:
    """Reproduces the finding's own grep: OMNINODE_INTERNAL requires none.

    Per the domain rules table (docs/plans/2026-07-29-one-application-database-
    domain-separation-plan.md §2.4), only ``TENANT`` requires
    ``ENABLE + FORCE + fail-closed policy``. This test pins that the migration
    that created the table carries neither -- now a proven, ruled absence, not
    an accidental one.
    """
    from pathlib import Path

    migration_path = (
        Path(__file__).parents[3]
        / "docker"
        / "migrations"
        / "forward"
        / "090_create_delegation_workflow_state.sql"
    )
    sql = migration_path.read_text(encoding="utf-8")
    assert "ROW LEVEL SECURITY" not in sql
    assert "CREATE POLICY" not in sql
    assert "tenant_id" in sql  # provenance column retained per R-q, not RLS'd
