# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ``role_omnidash`` watermark grant is declared in the topology [OMN-19998].

Migration ``node_projection_registration/0008`` grants ``role_omnidash`` schema
USAGE on ``omninode_internal`` and SELECT, INSERT, UPDATE on
``omninode_internal.projection_watermarks``, because the dev lane's live-events
writer connects as ``role_omnidash`` and upserts a watermark after every batch.
The OMN-17886 live-ACL gate builds ``omnidash_analytics`` from empty and diffs
its ``omninode_internal`` ACLs against the topology, in both directions, so a
grant the topology does not declare is a finding and the gate's shrink-only
allow-list may not absorb it. The declaration therefore has to live in the three
shipped instances, which the schema keeps identical (it has no include
mechanism).

``role_omnidash`` has no binding: ``application_database.py`` checks binding-set
equality and principal-set containment, so an unbound principal is structurally
permitted. Its TABLE grants are hand-declared, which the grant generator must
leave alone: it rewrites only the TABLE grants of principals a binding names,
because those are the ones a contract's ``db_io.db_tables`` can derive.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import (
    DerivedTableGrants,
    TopologyTableGrants,
)
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTANCE_ROOT = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
GENERATOR = REPO_ROOT / "scripts" / "generate_application_database_table_grants.py"

ROLE = "role_omnidash"
_SHIPPED_INSTANCES = tuple(sorted(path.stem for path in INSTANCE_ROOT.glob("*.yaml")))
_EXPECTED_GRANTS = [
    {"object_type": "SCHEMA", "privileges": ["USAGE"], "schema": "omninode_internal"},
    {
        "object_type": "TABLE",
        "objects": ["projection_watermarks"],
        "privileges": ["INSERT", "SELECT", "UPDATE"],
        "schema": "omninode_internal",
    },
    # OMN-17427: node_projection_consumer_flow/0006 grants the omnimarket
    # retention prune, which connects through OMNIDASH_ANALYTICS_DB_URL.
    {
        "object_type": "TABLE",
        "objects": ["consumer_flow_windows"],
        "privileges": ["DELETE", "SELECT"],
        "schema": "omninode_internal",
    },
    # OMN-17427: node_projection_lab_job/0002, node_projection_pr_state/0002 and
    # node_projection_work_ledger/0005 grant the omnimarket lab job check store,
    # which reads through the same OMNIDASH_ANALYTICS_DB_URL.
    {
        "object_type": "TABLE",
        "objects": ["lab_job_state", "pr_state", "work_ledger_rows"],
        "privileges": ["SELECT"],
        "schema": "omninode_internal",
    },
    # The grants the corpus already delivers to role_omnidash by name. Declaring
    # the principal makes the OMN-18768 reverse arm of
    # scripts/validation/check_topology_grant_delivery.py hold it to account for
    # every relation a migration grants it, so these are declared with it.
    {
        "object_type": "TABLE",
        "objects": ["generation_events"],
        "privileges": ["INSERT", "SELECT", "UPDATE"],
        "schema": "public",
    },
    {
        "object_type": "TABLE",
        "objects": ["tenant_registry_mirror"],
        "privileges": ["SELECT"],
        "schema": "public",
    },
]


def _application(name: str) -> dict[str, object]:
    document = yaml.safe_load((INSTANCE_ROOT / f"{name}.yaml").read_text("utf-8"))
    application = document["databases"]["application"]
    assert isinstance(application, dict)
    return application


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("generate_grants_omn19998", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
def test_role_omnidash_is_declared_with_exactly_the_grants_the_corpus_delivers(
    instance: str,
) -> None:
    spec = _application(instance)["principals"][ROLE]
    assert spec["bypass_rls"] is False
    assert spec["grants"] == _EXPECTED_GRANTS


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
def test_role_omnidash_has_no_binding(instance: str) -> None:
    bindings = _application(instance)["bindings"]
    assert ROLE not in {binding["principal"] for binding in bindings.values()}


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
def test_the_live_acl_gate_resolves_the_declaration(profile: str) -> None:
    declared = declared_acl(load_topology_profile(profile))
    expected = (
        {
            (ROLE, "projection_watermarks", privilege)
            for privilege in ("INSERT", "SELECT", "UPDATE")
        }
        | {
            (ROLE, "consumer_flow_windows", privilege)
            for privilege in ("DELETE", "SELECT")
        }
        | {
            (ROLE, table, "SELECT")
            for table in ("lab_job_state", "pr_state", "work_ledger_rows")
        }
    )
    assert expected <= declared.table_grants
    assert (ROLE, "USAGE") in declared.schema_grants
    assert not {g for g in declared.table_grants if g[0] == ROLE} - expected


@pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
def test_the_generator_keeps_an_unbound_principals_table_grants(instance: str) -> None:
    generator = _generator()
    path = INSTANCE_ROOT / f"{instance}.yaml"
    no_derivation = TopologyTableGrants(
        per_database={"application": DerivedTableGrants(grants={}, unmappable=())},
        unmappable=(),
    )
    rendered = yaml.safe_load(generator._render_instance(path, no_derivation))
    principals = rendered["databases"]["application"]["principals"]
    assert principals[ROLE]["grants"] == _EXPECTED_GRANTS
    # Positive control: a bound principal's TABLE grants are cleared when no
    # contract derives them, so the retention above is not "nothing was cleared".
    assert not [
        grant
        for grant in principals["omninode_runtime"]["grants"]
        if grant["object_type"] == "TABLE"
    ]


def test_the_instance_glob_finds_the_shipped_instances() -> None:
    """Positive control: the parametrised cases above are not over an empty set."""
    assert len(_SHIPPED_INSTANCES) == 3
