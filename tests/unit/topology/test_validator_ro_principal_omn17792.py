# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The ``validator_ro`` read-only principal on the onex-dev RDS (OMN-17792).

OMN-17792 AC6 asks for a read-only PostgreSQL role that makes the database half
of OMN-17298 / OMN-15359 / OMN-15425 / OMN-17440 answerable. The role that was
provisioned first, on the ``.201`` dev lane, is not the one those four tickets
validate against: they run against the ``onex-dev`` serving RDS
``omninode-dev-postgres`` (physical database ``omnidash_analytics``), reached
through the dev-system cluster EC2 ``i-06169517a92b45f86``. The read path in use
there today is ``role_omnidash``, which OWNS its relations -- so it can DROP,
ALTER and TRUNCATE them, and Postgres exempts a relation's owner from RLS
unconditionally. The argument for this principal is not missing access; it is
that the access in use is far wider than validation requires.

This module pins the DECLARATION half. Three properties, each of which has its
own way of going wrong:

* **Shape.** ``NOBYPASSRLS`` is the point of the role, not an inherited default.
  A validation principal that could bypass RLS would mask exactly the defect
  class OMN-17298 and OMN-17422 turn on -- it would read rows the tenant
  isolation policies exist to withhold and report the database clean.
* **Reach, and nothing beyond it.** The declared privilege set is CONNECT plus
  schema USAGE. Any write privilege, any ``CREATE``, any ``TEMPORARY`` in this
  principal's block is a defect: a role that can create a relation OWNS it, and
  an owner is exempt from RLS, which reopens the bypass for every future table.
* **Instance parity, and what actually scopes the role.** The instance schema
  has no include mechanism, so
  ``test_application_database_table_grants.py::test_shipped_instances_declare_identical_grants``
  requires the three shipped copies not to drift. The principal is therefore
  declared in all three, and the first draft of this module -- which asserted
  ``onex-dev`` alone -- was wrong about the platform rather than about the
  intent. What scopes the role is not the declaration:
  ``omninode_infra scripts/provision-cluster-roles.sh`` walks every
  ``principals`` block in the instance it is HANDED, and its only caller is the
  "Provision topology cluster roles (OMN-17347)" step of
  ``deploy-onex-staging.yml``, which passes ``onex-dev``. ``deploy-onex-prod.yml``
  has no such step. Even where the role is created it is created NOLOGIN and is
  inert until the deployment-owned credential attach runs for that lane.

The principal deliberately has NO binding. ``application_database.py`` checks
binding-set EQUALITY against ``_EXPECTED_BINDING_PRINCIPALS`` and principal-set
CONTAINMENT, so an extra principal with no binding is structurally permitted and
an extra BINDING is not. That asymmetry is what lets a non-workload principal be
declared here at all, and it is asserted rather than assumed.

Ticket: OMN-17792 (AC6, RDS half). Precedent: OMN-15425 / OMN-17301 / OMN-17347.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology import load_topology_profile

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTANCE_ROOT = REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
PROJECTION_ROOT = REPO_ROOT / "docker" / "catalog" / "database-topology"

ROLE = "validator_ro"
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")

# Every schema the onex-dev application database declares. The role is a
# read-only validation identity across the whole application database, so its
# USAGE set is the declared schema set rather than a subset chosen by taste.
EXPECTED_SCHEMAS = ("omninode_internal", "platform_catalog", "public", "tenant")

pytestmark = pytest.mark.unit


def _instance(name: str) -> dict[str, object]:
    document = yaml.safe_load((INSTANCE_ROOT / f"{name}.yaml").read_text("utf-8"))
    assert isinstance(document, dict)
    return document


def _principals(name: str) -> dict[str, object]:
    document = _instance(name)
    databases = document["databases"]
    assert isinstance(databases, dict)
    application = databases["application"]
    assert isinstance(application, dict)
    principals = application["principals"]
    assert isinstance(principals, dict)
    return principals


def test_validator_ro_is_declared_on_onex_dev() -> None:
    assert ROLE in _principals("onex-dev"), (
        f"{ROLE} is not declared in the onex-dev application principals. "
        "omninode_infra scripts/provision-cluster-roles.sh reads that block and "
        "nothing else -- an undeclared principal is never created on the RDS."
    )


def test_validator_ro_carries_the_isolation_relevant_shape() -> None:
    spec = _principals("onex-dev")[ROLE]
    assert isinstance(spec, dict)
    assert spec["bypass_rls"] is False, (
        "a validation principal that can bypass RLS reads rows the tenant "
        "isolation policies withhold and reports the database clean"
    )
    # `login: true` is the END state the deployment-owned credential attach
    # produces, and the typed model (ModelDeploymentTopologyDatabasePrincipal)
    # accepts no other value. The role is CREATEd NOLOGIN by the provisioning
    # seam; see migration 104 for why NOLOGIN is never re-asserted.
    assert spec["login"] is True


def test_validator_ro_declares_connect_and_schema_usage_and_nothing_else() -> None:
    spec = _principals("onex-dev")[ROLE]
    assert isinstance(spec, dict)
    grants = spec["grants"]
    assert isinstance(grants, list)

    database_grants = [g for g in grants if g["object_type"] == "DATABASE"]
    assert len(database_grants) == 1
    assert list(database_grants[0]["privileges"]) == ["CONNECT"]
    assert database_grants[0].get("schema") is None
    assert not database_grants[0].get("objects")

    schema_grants = [g for g in grants if g["object_type"] == "SCHEMA"]
    assert tuple(sorted(g["schema"] for g in schema_grants)) == EXPECTED_SCHEMAS
    for grant in schema_grants:
        assert list(grant["privileges"]) == ["USAGE"], (
            "USAGE only: CREATE on a schema makes this role the OWNER of every "
            "relation it creates, and an owner is exempt from RLS"
        )

    assert len(database_grants) + len(schema_grants) == len(grants), (
        "validator_ro declares a grant that is neither DATABASE CONNECT nor "
        "SCHEMA USAGE -- read-only reach is the whole point of the principal"
    )


def test_validator_ro_holds_no_write_or_ddl_privilege_anywhere() -> None:
    spec = _principals("onex-dev")[ROLE]
    assert isinstance(spec, dict)
    granted = {
        privilege for grant in spec["grants"] for privilege in grant["privileges"]
    }
    assert granted <= {"CONNECT", "USAGE"}, (
        f"validator_ro declares {sorted(granted - {'CONNECT', 'USAGE'})}; a "
        "validation role must not be able to change what it is validating"
    )


def test_validator_ro_has_no_binding_and_the_topology_still_validates() -> None:
    # The extra-principal/no-binding asymmetry is load-bearing: it is the only
    # reason a non-workload principal can be declared at all. If a future edit
    # tightens principal-set containment into equality, this fails here rather
    # than at RDS provisioning time.
    topology = load_topology_profile("onex-dev")
    database = topology.databases["application"]
    assert ROLE in database.principals
    assert ROLE not in {binding.principal for binding in database.bindings.values()}, (
        "validator_ro must never be bound: a binding is a runtime connection "
        "pool, and OMN-16911 attests current_user on every one of them"
    )


def test_validator_ro_is_declared_identically_in_all_three_instances() -> None:
    """The shipped instances have no include mechanism; the copies must not drift.

    This is the platform's own rule (``test_shipped_instances_declare_identical_grants``),
    and it is asserted here too so a future edit that adds the principal to one
    instance fails naming THIS principal rather than as a whole-block inequality.
    """
    blocks = {
        instance: _principals(instance).get(ROLE) for instance in _SHIPPED_INSTANCES
    }
    assert all(block is not None for block in blocks.values()), (
        f"{ROLE} is missing from {sorted(k for k, v in blocks.items() if v is None)}"
    )
    reference = blocks["onex-dev"]
    for instance, block in blocks.items():
        assert block == reference, f"{ROLE} drifts between onex-dev and {instance}"


def test_validator_ro_reaches_a_cluster_only_where_the_seam_is_invoked() -> None:
    """Declaration is not provisioning -- asserted against the real workflows.

    ``provision-cluster-roles.sh`` walks every ``principals`` block in the
    instance it is handed, so "which instances declare it" does not bound blast
    radius; "which workflow invokes the seam, with which --instance" does. If a
    prod deploy ever gains that step, this fails and the decision gets made
    deliberately instead of arriving as a side effect of a parity rule.
    """
    # omninode_infra is a sibling clone, not a dependency of this package, so
    # this check is best-effort by construction: it runs on any machine that has
    # the clone (every workspace checkout does) and skips on a CI runner that
    # does not. It is NOT the enforcement surface -- the prod decision is
    # recorded in the topology comment and on OMN-17792. It is here so that a
    # workspace lane changing the seam's caller set trips over this immediately.
    candidates = [REPO_ROOT.parent / "omninode_infra"]
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home:
        candidates.append(Path(omni_home) / "omninode_infra")
    for candidate in candidates:
        workflows = candidate / ".github" / "workflows"
        if workflows.is_dir():
            break
    else:
        pytest.skip(
            "omninode_infra is not checked out beside this repo or under OMNI_HOME"
        )

    callers = {
        path.name
        for path in sorted(workflows.glob("*.yml"))
        if "provision-cluster-roles.sh" in path.read_text(encoding="utf-8")
    }
    assert callers == {"deploy-onex-staging.yml"}, (
        "the set of workflows invoking the cluster-role provisioning seam "
        f"changed: {sorted(callers)}. validator_ro is declared in all three "
        "instances by the parity rule, so the seam's caller set IS its blast "
        "radius -- re-read the prod decision before letting this through."
    )


def test_instance_files_carry_no_indented_comment() -> None:
    """A rationale comment inside a principals block is silently destroyed.

    ``scripts/generate_application_database_table_grants.py`` re-serialises the
    whole document through ``yaml.dump`` and rebuilds the file as
    ``(column-0 # lines) + (dumped body)``. An indented comment survives neither
    ``--write`` nor the byte comparison ``--check`` makes, so it does not read as
    "a comment was lost" -- it reads as **derivation drift on all three
    instances**, on a PR that changed no grant. That is what it did here: the
    first revision of this change put a 37-line rationale block inside the
    ``validator_ro`` principal and reddened
    ``Application Database Domain Enforcement (OMN-15361)`` with a message
    telling the author to regenerate.

    The generator's own ``--check`` catches this, but only in the CI job that
    has the cross-repo omnimarket checkout. This runs everywhere, costs nothing,
    and says what is actually wrong.
    """
    for instance in _SHIPPED_INSTANCES:
        offending = [
            (number, line)
            for number, line in enumerate(
                (INSTANCE_ROOT / f"{instance}.yaml").read_text("utf-8").splitlines(), 1
            )
            if line.startswith((" ", "\t")) and line.lstrip().startswith("#")
        ]
        assert offending == [], (
            f"{instance}.yaml carries an indented comment at line(s) "
            f"{[number for number, _ in offending]}. This file is machine-rendered; "
            "only column-0 '#' lines survive. Put the rationale in the migration "
            "that provisions the principal and in its test module."
        )


def test_rendered_projection_carries_the_principal() -> None:
    # The rendered catalog is what docker/compose consumers read; the unit
    # suite fails closed on drift, and this names WHICH principal drifted.
    projection = yaml.safe_load((PROJECTION_ROOT / "onex-dev.yaml").read_text("utf-8"))
    assert ROLE in projection["databases"]["application"]["principals"]
