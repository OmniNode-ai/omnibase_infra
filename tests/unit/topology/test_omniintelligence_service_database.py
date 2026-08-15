# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shipped-topology proof for the omniintelligence service database (OMN-15655 AC-2).

``node_dispatch_outcome_bridge_effect`` (omnimarket) declares
``db_io.db_tables[0].database_ref: omniintelligence`` and runs on the
``effects`` runtime profile, where ``ONEX_WIRING_STRICT_MODE`` turns an
unresolved reference into a boot-fatal error rather than a skipped handler.
Before this declaration landed, ``_resolve_projection_database_target`` raised
``ValueError: Unknown database_ref 'omniintelligence'`` on all seven supported
profiles.

Everything here drives the **shipped** ``load_topology_profile(profile)`` and
the **real** resolver. No fixture topology, no synthesised grants, no
cross-repo checkout: the contract's declared values are pinned verbatim below
so this suite runs inside omnibase_infra's own CI, the same way the OMN-15547
incident replay pins the relation that broke onex-dev.
"""

from __future__ import annotations

import pytest

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _DB_URL_ENV_MAP,
    _INTERNAL_PROJECTION_BINDING,
    _make_projection_dispatch_callback,
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import (
    OMNIINTELLIGENCE_DATABASE_REF,
    SUPPORTED_TOPOLOGY_PROFILES,
    validate_omniintelligence_database_invariants,
)

pytestmark = pytest.mark.unit

# Verbatim from omnimarket
# src/omnimarket/nodes/node_dispatch_outcome_bridge_effect/contract.yaml
# (db_io.db_tables[0]). If omnimarket edits that block, this declaration must
# be updated with it — the cross-repo derivation gate
# (scripts/generate_application_database_table_grants.py --check --prove) is
# what catches the drift against the pinned checkout.
_BRIDGE_RELATION = ModelDbTableDeclaration(
    name="dispatch_eval_results",
    database_ref="omniintelligence",
    schema="public",
    migration=(
        "omniintelligence/deployment/database/migrations/"
        "023_create_debug_intelligence_tables.sql"
    ),
    access="write",
    role="eval_results",
)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_bridge_effect_relation_resolves_on_every_shipped_profile(
    profile: str,
) -> None:
    """The exact declaration that was boot-fatal now resolves, on every profile."""
    target = _resolve_projection_database_target(
        (_BRIDGE_RELATION,), load_topology_profile(profile)
    )
    assert target.physical_database == "omniintelligence"
    assert target.database_refs == (OMNIINTELLIGENCE_DATABASE_REF,)
    assert target.schemas == ("public",)
    assert target.domains == (EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,)


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_bridge_effect_resolves_to_the_service_binding_and_dsn_key(
    profile: str,
) -> None:
    """A write must select the internal binding, the service role, and its DSN key.

    Asserted rather than assumed: resolving to the *application* pool would
    connect the effects pod to the wrong physical database while still
    reporting a green boot.
    """
    target = _resolve_projection_database_target(
        (_BRIDGE_RELATION,), load_topology_profile(profile)
    )
    (binding,) = target.bindings
    assert binding.binding_ref == _INTERNAL_PROJECTION_BINDING
    assert binding.principal == "role_omniintelligence"
    assert binding.physical_database == "omniintelligence"
    assert binding.dsn_env == "OMNIINTELLIGENCE_DB_URL"
    assert target.dsn_envs == ("OMNIINTELLIGENCE_DB_URL",)


def test_declared_dsn_key_matches_the_per_service_db_url_contract() -> None:
    """The topology DSN key must be the one the wiring module already knows.

    ``docs/patterns/db_url_contract.md`` is the authority both surfaces
    project; a mismatch here would give the same database two DSN names.
    """
    assert _DB_URL_ENV_MAP["omniintelligence"] == "OMNIINTELLIGENCE_DB_URL"


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_service_database_is_physically_separate_from_the_application_pair(
    profile: str,
) -> None:
    """ADR-0027 unified the application pair and kept service databases separate."""
    topology = load_topology_profile(profile)
    assert (
        topology.databases[OMNIINTELLIGENCE_DATABASE_REF].physical_name
        == "omniintelligence"
    )
    assert topology.databases["application"].physical_name == "omnidash_analytics"


def test_dispatch_eval_results_is_internal_because_it_carries_no_tenant_key() -> None:
    """OMNINODE_INTERNAL is falsifiable here, not a default.

    The live relation has no ``tenant_id`` column and its only writer
    (omniintelligence ``handler_dispatch_outcome.py``) enumerates thirteen
    columns, none of them a tenant key. A TENANT domain would select
    ``TenantProjectionTableOperation``, which unconditionally stamps
    ``row["tenant_id"]`` and demands a verified tenant authority — every write
    would fail on a column that does not exist.
    """
    topology = load_topology_profile("local")
    assert (
        topology.table_domain(_BRIDGE_RELATION)
        is EnumDatabaseSchemaDomain.OMNINODE_INTERNAL
    )


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_shipped_profiles_pass_the_service_database_invariants(profile: str) -> None:
    """``load_topology_profile`` already enforces these; assert the guard exists."""
    validate_omniintelligence_database_invariants(load_topology_profile(profile))


def test_invariants_reject_a_repointed_physical_database() -> None:
    """Silently repointing the service database must fail the loader, not the pod."""
    topology = load_topology_profile("local")
    database = topology.databases[OMNIINTELLIGENCE_DATABASE_REF]
    drifted = topology.model_copy(
        update={
            "databases": {
                **topology.databases,
                OMNIINTELLIGENCE_DATABASE_REF: database.model_copy(
                    update={"physical_name": "omnidash_analytics"}
                ),
            }
        }
    )
    with pytest.raises(ValueError, match="must resolve to 'omniintelligence'"):
        validate_omniintelligence_database_invariants(drifted)


def test_wiring_still_requires_a_configured_dsn_after_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measured residual: resolution is necessary but not sufficient for boot.

    ``_make_projection_dispatch_callback`` reads ``binding.dsn_env`` from the
    environment at **wiring** time and fails closed on an empty value. The
    onex-dev runtime deployments pin ``OMNIINTELLIGENCE_DB_URL: ""``
    (omninode_infra ``k8s/onex-dev/runtime/deployment-omninode-runtime*.yaml``,
    the OMN-13769 workaround), so on that lane this declaration converts a
    ``Unknown database_ref`` boot failure into a ``requires topology bindings
    with configured DSNs`` boot failure until the secretKeyRef is restored.
    This test pins that fact so it is not mistaken for a green path.
    """
    # Reproduce the lane exactly: the k8s manifest sets the key to "", which
    # ``os.environ.get(dsn_env, "")`` cannot distinguish from unset.
    monkeypatch.setenv("OMNIINTELLIGENCE_DB_URL", "")
    target = _resolve_projection_database_target(
        (_BRIDGE_RELATION,), load_topology_profile("onex-dev")
    )
    with pytest.raises(ValueError, match="requires topology bindings with configured"):
        _make_projection_dispatch_callback(
            object(),
            target,
            ("onex.evt.omniclaude.dispatch_worker-completed.v1",),
        )
