# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Enforcement for the CONNECT half of the application-database ACL inputs.

OMN-15355. The generated deployment matrix refuses to become appliable while it
carries no expected CONNECT set, and it has carried none since it was first
generated: ``allowed_connect_principals`` on the committed candidate is ``{}``.
The reason is an INPUT gap, not a code defect. ``build_application_database_acl_matrix``
resolves the allowed principals for one database from an ``acl_policy`` source,
cross-checks them against the typed deployment topology, and blocks when the two
disagree -- but the deployment CONNECT universe is a frozenset of eight physical
databases while the typed topology declares only three.

The five it cannot declare are not a bug to be fixed by writing them down. There
is no authority that says which principal connects to ``keycloak``,
``omniclaude``, ``omnimemory``, ``omninode_cloud`` or ``umami``; inventing one
would encode a guess as live privilege, which is the precise mistake the whole
matrix exists to prevent. So they are recorded here, by name, as the open
question -- and the gap becomes a named red line rather than five entries buried
in a 300-plus blocker wall that nobody reads to the end.

What this module enforces:

* a CONNECT allowlist cannot WIDEN silently. Every allowlist the topology does
  declare is pinned to its exact expected set, in every topology instance. A new
  principal gaining CONNECT to a governed database is a privilege change and
  fails here before it can reach a matrix.
* the undeclared set cannot GROW. Shrinking is always allowed -- that is the gap
  closing -- but a new database entering the deployment scope without a topology
  declaration fails.
* the committed candidate's empty CONNECT expectation is pinned, so the day it
  stops being empty someone has to come back and say so deliberately.

Every zero this module reports is covered by a positive control: the same
derivation, run against a seeded topology that declares the full eight, must
return an empty undeclared set. Without that control a derivation that silently
returned nothing would read exactly like a gap that had been closed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.validation.application_database_acl import (
    _DEPLOYMENT_CONNECT_DATABASES,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
)

_ROOT = Path(__file__).parents[2]
_TOPOLOGY_DIR = _ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_CANDIDATE = (
    _ROOT / "docker" / "application-acl-proof" / "generated" / "candidate-matrix.yaml"
)

# Every topology instance that describes a governed deployment. The allowlists
# below are asserted against ALL of them: an instance that quietly granted one
# extra principal on staging would otherwise be invisible here.
_TOPOLOGY_INSTANCES = ("local", "onex-dev", "onex-prod")

# The measured, topology-declared CONNECT expectation, physical database ->
# (topology database_ref, exact allowed principals). Derived from
# src/omnibase_infra/topology/instances/*.yaml, never hand-authored: each entry
# is the set of principals whose contract carries a DATABASE-object grant
# including CONNECT. Changing any tuple here is a deliberate privilege decision.
_DECLARED_CONNECT: dict[str, tuple[str, tuple[str, ...]]] = {
    "omnidash_analytics": (
        "application",
        (
            "app_dashboard",
            "omninode_runtime",
            "onex_api",
            "tenant_projection_writer",
            "validator_ro",
        ),
    ),
    "omnibase_infra": ("omnibase_infra", ("role_omnibase_infra",)),
    "omniintelligence": ("omniintelligence", ("role_omniintelligence",)),
}

# The open question, stated as data. These five are inside the approved
# eight-database deployment CONNECT scope and NO typed topology instance carries
# a ModelDeploymentTopology entry for them, so no expected CONNECT set can be
# derived for any of them. Each needs an operator answer naming the service
# identity that connects; until then the matrix is correctly BLOCKED on them.
_UNDECLARED_DEPLOYMENT_DATABASES = frozenset(
    {
        "keycloak",
        "omniclaude",
        "omnimemory",
        "omninode_cloud",
        "umami",
    }
)


def _connect_allowlists(
    topology: ModelDeploymentTopology,
) -> dict[str, tuple[str, ...]]:
    """Return physical database -> principals holding a DATABASE CONNECT grant.

    This mirrors the resolution ``build_application_database_acl_matrix`` performs
    when it cross-checks a CONNECT policy against the topology, so a drift between
    the two surfaces shows up here rather than as a matrix blocker.
    """
    allowlists: dict[str, tuple[str, ...]] = {}
    for database in topology.databases.values():
        allowed = sorted(
            principal
            for principal, contract in database.principals.items()
            if any(
                grant.object_type is EnumDatabaseGrantObjectType.DATABASE
                and EnumDatabasePrivilege.CONNECT in grant.privileges
                for grant in contract.grants
            )
        )
        if allowed:
            allowlists[database.physical_name] = tuple(allowed)
    return allowlists


def _rekey(value: Any, old_ref: str, new_ref: str) -> Any:
    """Re-point every ``database_ref`` in a cloned database entry at its new key.

    ``ModelDeploymentTopology`` validates that a binding names its containing
    database, so a clone keeps failing validation until its internal refs follow
    it. Only the ref is rewritten; nothing else about the entry is invented.
    """
    if isinstance(value, dict):
        return {
            key: (
                new_ref
                if key == "database_ref" and item == old_ref
                else _rekey(item, old_ref, new_ref)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rekey(item, old_ref, new_ref) for item in value]
    return value


def _load_topology(instance: str) -> ModelDeploymentTopology:
    return ModelDeploymentTopology.model_validate(
        yaml.safe_load((_TOPOLOGY_DIR / f"{instance}.yaml").read_text(encoding="utf-8"))
    )


@pytest.mark.parametrize("instance", _TOPOLOGY_INSTANCES)
def test_declared_connect_allowlists_never_widen_silently(instance: str) -> None:
    """A principal gaining CONNECT on a governed database fails before it applies."""
    allowlists = _connect_allowlists(_load_topology(instance))

    expected = {
        physical: principals for physical, (_, principals) in _DECLARED_CONNECT.items()
    }
    assert allowlists == expected, (
        f"{instance}: topology CONNECT allowlists drifted from the pinned "
        "expectation. Widening an allowlist is a privilege change and must be "
        "made deliberately here, with the added principal named."
    )


@pytest.mark.parametrize("instance", _TOPOLOGY_INSTANCES)
def test_declared_refs_match_the_topology_database_refs(instance: str) -> None:
    """The pinned database_ref for each allowlist is the ref the matrix resolves."""
    topology = _load_topology(instance)
    refs = {database.physical_name: ref for ref, database in topology.databases.items()}

    for physical, (expected_ref, _) in _DECLARED_CONNECT.items():
        assert refs.get(physical) == expected_ref, (
            f"{instance}: {physical} resolves to topology ref {refs.get(physical)!r}, "
            f"not the pinned {expected_ref!r}; the CONNECT policy cross-check keys "
            "on this ref."
        )


@pytest.mark.parametrize("instance", _TOPOLOGY_INSTANCES)
def test_deployment_scope_names_every_database_the_topology_cannot_declare(
    instance: str,
) -> None:
    """The undeclared set is named, and it may shrink but never grow.

    ``declared`` is derived from the topology rather than read back from
    ``_DECLARED_CONNECT``, so this fails on BOTH drift directions: a database
    entering the deployment scope with no declaration, and a database losing the
    declaration it had.
    """
    declared = set(_connect_allowlists(_load_topology(instance)))
    undeclared = set(_DEPLOYMENT_CONNECT_DATABASES) - declared

    assert declared <= set(_DEPLOYMENT_CONNECT_DATABASES), (
        "a database carries a topology CONNECT declaration but is outside the "
        f"approved deployment scope: {sorted(declared - set(_DEPLOYMENT_CONNECT_DATABASES))!r}"
    )
    assert undeclared <= _UNDECLARED_DEPLOYMENT_DATABASES, (
        "a database entered the deployment CONNECT scope with no typed topology "
        "declaration, so no expected CONNECT set can be derived for it: "
        f"{sorted(undeclared - _UNDECLARED_DEPLOYMENT_DATABASES)!r}. Declare its "
        "principals in the topology, or take it out of the deployment scope."
    )


def test_positive_control_a_fully_declared_topology_reports_no_gap() -> None:
    """The zero above is real: the same derivation returns an empty gap when it is empty.

    Without this, a derivation that silently found nothing would be
    indistinguishable from a gap that had been closed.
    """
    local = _load_topology("local")
    seeded: dict[str, Any] = yaml.safe_load(
        (_TOPOLOGY_DIR / "local.yaml").read_text(encoding="utf-8")
    )
    template_ref, template = next(iter(seeded["databases"].items()))

    for physical in sorted(_UNDECLARED_DEPLOYMENT_DATABASES):
        seeded_ref = f"seeded_{physical}"
        clone = _rekey(
            yaml.safe_load(yaml.safe_dump(template)), template_ref, seeded_ref
        )
        clone["physical_name"] = physical
        seeded["databases"][seeded_ref] = clone

    seeded_topology = ModelDeploymentTopology.model_validate(seeded)
    gap = set(_DEPLOYMENT_CONNECT_DATABASES) - set(_connect_allowlists(seeded_topology))

    assert not gap, f"positive control failed to close the gap: {sorted(gap)!r}"
    # ...and the same derivation on the real topology still reports the real gap,
    # which is what proves the control discriminates rather than always passing.
    assert set(_DEPLOYMENT_CONNECT_DATABASES) - set(_connect_allowlists(local)) == (
        _UNDECLARED_DEPLOYMENT_DATABASES
    )


def test_committed_candidate_still_carries_no_connect_expectation() -> None:
    """Pin the input gap on the artifact the apply CLI actually refuses."""
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(
        yaml.safe_load(_CANDIDATE.read_text(encoding="utf-8"))
    )

    assert matrix.status == "BLOCKED"
    assert matrix.allowed_connect_principals == {}
    assert matrix.observed_connect_principals == {}
    assert matrix.absent_connect_principals == {}

    # The source lock declares no acl_policy and no principal_inventory source, so
    # every one of the eight databases reports a missing CONNECT policy. Both
    # halves are required: the policy carries the allowlist, the inventory carries
    # the presence/absence and full-day activity evidence it is checked against.
    missing_policy = {
        blocker.split(":", 1)[0]
        for blocker in matrix.blockers
        if "requires exactly one CONNECT policy" in blocker
    }
    assert missing_policy == set(_DEPLOYMENT_CONNECT_DATABASES)
    assert (
        "matrix requires an independent typed acl_policy source for every database"
        in matrix.blockers
    )
    assert (
        "matrix requires a typed principal_inventory source for every database"
        in matrix.blockers
    )
