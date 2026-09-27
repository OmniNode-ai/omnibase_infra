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

The five it cannot declare were first recorded here as five open operator
questions. That framing was WRONG and OMN-15418 retires it. Which service owns
each of those databases, which identity connects to it, and on which lanes it
exists are all recorded in committed deployment sources -- compose files, k8s
overlays, provisioning scripts and service ownership manifests -- so they are
DERIVED below, each carrying the file and line it was read from. An operator
question is what remains when no source answers; none of these five is that.

What the derivation found, and it is the same answer for all five: every one of
them is blocked from becoming a ``ModelDeploymentTopology`` database entry by a
MODEL constraint, not by a missing decision. ``ModelDeploymentTopologyDatabase``
requires a ``checksum_ledgers`` entry whose ``stream_column``, ``domain_column``,
``version_column`` and ``checksum_column`` are four DISTINCT columns
(``omnibase_core`` ``model_deployment_topology_database_migration_ledger.py``,
``ledger_columns_are_distinct``). Two of the five are vendored databases carrying
a vendor's own ledger, and the other three carry an OmniNode ledger with at most
two of the four columns. Writing a conforming ledger for any of them would be
inventing a fact -- the same mistake the earlier "invent a principal" framing was
right to refuse. So the derivation is recorded and the topology entry is not
written, and the reason is a named code constraint a reader can go and check.

A lane row carrying NO principal is a derived finding, not an omission. Measured
against the 2026-09-14 read-only pre-change ACL snapshot of the .201 dev lane and
its `pg_stat_activity` sample: `omniclaude`, `omnimemory` and `omninode_cloud`
exist there owned by the bootstrap superuser with no other grantee, their
`role_*` logins do not exist on that lane at all, and nothing connected to any of
them for the whole sample. `umami` is not present on that lane in any form.
`keycloak` is, and the only identity observed reaching it is the superuser the
compose file configures Keycloak with. The sample is a 5h22m window, not the
full day the principal inventory needs, so "nothing observed" here bounds what
was seen and is never read as proof a database is unused.

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

from dataclasses import dataclass
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
    _SQL_IDENTIFIER,
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


@dataclass(frozen=True)
class UndeclaredDeploymentDatabase:
    """One in-scope physical database the typed topology does not declare.

    Every field is DERIVED from a committed deployment source, never asserted.
    ``provenance`` carries the exact ``repo:path:line`` each fact was read from,
    and ``blocked_by`` names the model constraint that stops the derivation from
    becoming a ``ModelDeploymentTopology`` entry.
    """

    physical_name: str
    owner_service: str
    # Lane -> the role names declared or observed to connect there. Role NAMES
    # only; no credential, DSN or secret reference appears anywhere in here.
    connecting_principals: tuple[tuple[str, tuple[str, ...]], ...]
    absent_from_lanes: tuple[str, ...]
    migration_ledger: str
    blocked_by: str
    provenance: tuple[str, ...]

    @property
    def principal_names(self) -> frozenset[str]:
        """Return every role name this database is derived to be reached by."""
        return frozenset(
            role for _, roles in self.connecting_principals for role in roles
        )


# Repositories a provenance citation may name. A citation into anything else is
# a typo or a source this repo has no business reading, and fails structurally.
_PROVENANCE_REPOS = frozenset(
    {"omnibase_infra", "omnibase_core", "omninode_infra", "omniclaude", "omnimemory"}
)

# Provenance citations that live in THIS repository, so the test can open the
# file and prove the cited line still says what the derivation claims. A
# cross-repo citation cannot be read from omnibase_infra CI and is pinned by its
# quoted text instead -- stated plainly rather than dressed up as an assertion.
_IN_REPO_PROVENANCE: dict[str, str] = {
    "omnibase_infra:docker/docker-compose.infra.yml:885": "KC_DB_USERNAME: postgres",
    "omnibase_infra:docker/postgres/init/02-keycloak-db.sql:4": (
        "CREATE DATABASE keycloak"
    ),
}

# The five, derived. This table REPLACES the five operator questions that stood
# here. Every entry is falsifiable: open the cited file at the cited line.
_UNDECLARED_DERIVATION: tuple[UndeclaredDeploymentDatabase, ...] = (
    UndeclaredDeploymentDatabase(
        physical_name="keycloak",
        owner_service="keycloak (vendored identity provider, Liquibase-managed)",
        connecting_principals=(
            # The compose lanes point Keycloak at the bootstrap superuser; the
            # cloud plane gives it a dedicated owning role of the same name.
            (
                "compose (.201 dev): configured carrier, observed connecting",
                ("postgres",),
            ),
            (
                "cloud (RDS omninode-dev-postgres, auth namespace): declared",
                ("keycloak",),
            ),
        ),
        absent_from_lanes=("onex-lab",),
        migration_ledger="vendor-owned: Keycloak Liquibase (databasechangelog)",
        blocked_by=(
            "vendored database -- create-shared-owner-role.sh excludes it from "
            "OmniNode DDL ownership by name, and it carries no OmniNode "
            "four-column checksum ledger"
        ),
        provenance=(
            "omnibase_infra:docker/docker-compose.infra.yml:885",
            "omnibase_infra:docker/postgres/init/02-keycloak-db.sql:4",
            "omninode_infra:k8s/auth/keycloak-values.yaml:32",
            "omninode_infra:scripts/init-databases.sh:603",
            "omninode_infra:scripts/create-shared-owner-role.sh:24",
        ),
    ),
    UndeclaredDeploymentDatabase(
        physical_name="omniclaude",
        owner_service="omniclaude",
        connecting_principals=(
            (
                "compose (.201 dev): database present, owned by the bootstrap superuser, no service role exists, nothing observed connecting",
                (),
            ),
            ("public-cluster dev namespace: declared", ("role_omniclaude",)),
        ),
        absent_from_lanes=("onex-lab", "onex-dev", "onex-prod"),
        migration_ledger="public.schema_migrations(filename, applied_at)",
        blocked_by=(
            "ledger carries neither a stream, domain, version nor checksum "
            "column, so no conforming checksum_ledgers entry exists"
        ),
        provenance=(
            "omninode_infra:k8s/dev/postgres/init-databases-cm.yaml:90",
            "omninode_infra:scripts/create-shared-owner-role.sh:114",
            "omniclaude:scripts/init-db.sh:77",
        ),
    ),
    UndeclaredDeploymentDatabase(
        physical_name="omnimemory",
        owner_service="omnimemory",
        connecting_principals=(
            (
                "compose (.201 dev): database present, owned by the bootstrap superuser, no service role exists, nothing observed connecting",
                (),
            ),
            ("public-cluster dev namespace: declared", ("role_omnimemory",)),
        ),
        absent_from_lanes=("onex-lab", "onex-dev", "onex-prod"),
        migration_ledger="none -- flat .sql corpus with no ledger relation",
        blocked_by=(
            "the service applies raw migration files and records nothing, so "
            "there is no ledger relation to declare at all"
        ),
        provenance=(
            "omninode_infra:k8s/dev/postgres/init-databases-cm.yaml:91",
            "omninode_infra:scripts/create-shared-owner-role.sh:114",
            "omnimemory:deployment/database/migrations/001_create_subscription_tables.sql:1",
        ),
    ),
    UndeclaredDeploymentDatabase(
        physical_name="omninode_cloud",
        owner_service="onex_api (cloud control plane)",
        connecting_principals=(
            (
                "compose (.201 dev): database present, owned by the bootstrap superuser, no service role exists, nothing observed connecting",
                (),
            ),
            ("onex-lab: declared and owning", ("role_omninode_cloud",)),
            (
                "public-cluster dev namespace: declared",
                ("role_omninode_cloud",),
            ),
        ),
        absent_from_lanes=(),
        migration_ledger="public.schema_migrations(version, applied_at, checksum)",
        blocked_by=(
            "ledger carries two of the four required columns -- no stream and "
            "no domain column -- so ledger_columns_are_distinct cannot be met"
        ),
        provenance=(
            "omninode_infra:db/migrations/application-relation-ownership.yaml:6",
            "omninode_infra:k8s/migrations/omninode-cloud-migrate.yaml:354",
            "omninode_infra:k8s/onex-lab/substitutions/postgres.yaml:355",
            "omninode_infra:db/migrations/00000000_migrations_tracking.sql:70",
        ),
    ),
    UndeclaredDeploymentDatabase(
        physical_name="umami",
        owner_service="umami (vendored web analytics, Prisma-managed)",
        connecting_principals=(
            ("onex-dev, onex-prod, onex-public: declared and owning", ("umami",)),
        ),
        # Measured, not assumed: the 2026-09-14 pre-change ACL snapshot of the
        # .201 dev lane enumerates every non-template database and umami is not
        # among them, so a compose-lane CONNECT proof cannot cover it.
        absent_from_lanes=(
            "compose (.201 dev)",
            "onex-lab",
            "public-cluster dev namespace",
        ),
        migration_ledger="vendor-owned: Prisma (_prisma_migrations)",
        blocked_by=(
            "vendored database -- create-shared-owner-role.sh excludes it from "
            "OmniNode DDL ownership by name, and it carries no OmniNode "
            "four-column checksum ledger"
        ),
        provenance=(
            "omninode_infra:k8s/onex-dev/umami/deployment.yaml:88",
            "omninode_infra:k8s/onex-prod/umami/deployment.yaml:81",
            "omninode_infra:scripts/create-shared-owner-role.sh:24",
        ),
    ),
)

# Derived from the table above, never written twice: a database can only leave
# the undeclared set by gaining a topology declaration, and it can only enter by
# someone adding a derivation row that says why it has none.
_UNDECLARED_DEPLOYMENT_DATABASES = frozenset(
    entry.physical_name for entry in _UNDECLARED_DERIVATION
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


# ---------------------------------------------------------------------------
# OMN-15418: the derivation that retired the five operator questions.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entry", _UNDECLARED_DERIVATION, ids=lambda e: e.physical_name)
def test_every_undeclared_database_carries_a_complete_derivation(
    entry: UndeclaredDeploymentDatabase,
) -> None:
    """No entry may be a name with a shrug attached.

    An entry that named the database and nothing else is exactly the open
    question this table replaced, so each of the four derived facts -- owner
    service, connecting principals, ledger, and the reason the topology cannot
    hold it -- is required rather than optional.
    """
    assert entry.owner_service, f"{entry.physical_name}: no owner service derived"
    assert entry.connecting_principals, (
        f"{entry.physical_name}: no connecting principal derived. A database "
        "nothing connects to is a finding in its own right and must be recorded "
        "as one, not left empty."
    )
    assert entry.migration_ledger, f"{entry.physical_name}: no ledger recorded"
    assert entry.blocked_by, (
        f"{entry.physical_name}: no reason recorded for why this derivation is "
        "not a ModelDeploymentTopology entry. Without it the row reads as an "
        "unanswered question again."
    )
    assert entry.provenance, f"{entry.physical_name}: no provenance cited"


@pytest.mark.parametrize("entry", _UNDECLARED_DERIVATION, ids=lambda e: e.physical_name)
def test_every_derived_principal_is_a_plain_role_name(
    entry: UndeclaredDeploymentDatabase,
) -> None:
    """Role names only: this table must never become a place a DSN can hide."""
    for principal in entry.principal_names:
        assert _SQL_IDENTIFIER.fullmatch(principal), (
            f"{entry.physical_name}: derived principal {principal!r} is not a "
            "bare SQL role name. A URL, host or credential must never appear in "
            "this table."
        )


@pytest.mark.parametrize("entry", _UNDECLARED_DERIVATION, ids=lambda e: e.physical_name)
def test_every_provenance_citation_is_resolvable(
    entry: UndeclaredDeploymentDatabase,
) -> None:
    """Each citation names an allowlisted repo, a relative path, and a line.

    A citation is the whole value of this table -- it is what makes a derived
    fact checkable instead of asserted -- so a malformed one fails here rather
    than misleading the next reader.
    """
    for citation in entry.provenance:
        repo, _, remainder = citation.partition(":")
        path, _, line = remainder.rpartition(":")
        assert repo in _PROVENANCE_REPOS, (
            f"{entry.physical_name}: citation {citation!r} names repository "
            f"{repo!r}, which is not one of {sorted(_PROVENANCE_REPOS)}"
        )
        assert path and not path.startswith("/"), (
            f"{entry.physical_name}: citation {citation!r} must carry a "
            "repository-relative path, never an absolute one"
        )
        assert line.isdigit() and int(line) > 0, (
            f"{entry.physical_name}: citation {citation!r} must end in a line number"
        )


def test_in_repo_provenance_still_says_what_the_derivation_claims() -> None:
    """Open the in-repo cited lines and read them.

    This is the only half of the provenance that omnibase_infra CI can actually
    verify; the cross-repo citations are recorded text, checked by a reader, and
    the next test pins that limit rather than leaving it implied.
    """
    assert _IN_REPO_PROVENANCE, "the in-repo provenance pin cannot be empty"

    cited = {
        citation
        for entry in _UNDECLARED_DERIVATION
        for citation in entry.provenance
        if citation.startswith("omnibase_infra:")
    }
    assert cited == set(_IN_REPO_PROVENANCE), (
        "every omnibase_infra citation must be pinned to its expected text, and "
        "every pin must be cited by a derivation entry; drifted="
        f"{sorted(cited ^ set(_IN_REPO_PROVENANCE))!r}"
    )

    for citation, expected in _IN_REPO_PROVENANCE.items():
        _, _, remainder = citation.partition(":")
        path, _, line = remainder.rpartition(":")
        lines = (_ROOT / path).read_text(encoding="utf-8").splitlines()
        assert int(line) <= len(lines), (
            f"{citation}: file has only {len(lines)} lines, so the citation has rotted"
        )
        assert expected in lines[int(line) - 1], (
            f"{citation}: line {line} no longer contains {expected!r}. Either "
            "the source moved and the citation needs updating, or the derived "
            f"fact changed. Line reads: {lines[int(line) - 1]!r}"
        )


def test_positive_control_the_provenance_reader_can_fail() -> None:
    """The zero above is real: the same read, given a wrong line, disagrees.

    Without this, a reader that silently matched everything would make the pin
    above pass for a citation that had rotted completely.
    """
    citation, expected = next(iter(_IN_REPO_PROVENANCE.items()))
    _, _, remainder = citation.partition(":")
    path, _, line = remainder.rpartition(":")
    lines = (_ROOT / path).read_text(encoding="utf-8").splitlines()

    wrong = [text for text in lines if expected not in text]
    assert wrong, (
        f"positive control cannot run: every line of {path} contains "
        f"{expected!r}, so a passing pin proves nothing"
    )

    # The pin must be LINE-sensitive, not merely file-sensitive: the neighbour
    # of the cited line must not satisfy it. A pin that any line could satisfy
    # would pass for a citation whose line number was wrong by a hundred.
    neighbour = lines[int(line) - 2]
    assert expected not in neighbour, (
        f"positive control failed: line {int(line) - 1} of {path} also contains "
        f"{expected!r}, so the pin does not discriminate between lines"
    )


@pytest.mark.parametrize("entry", _UNDECLARED_DERIVATION, ids=lambda e: e.physical_name)
def test_a_derived_database_is_still_absent_from_every_topology_instance(
    entry: UndeclaredDeploymentDatabase,
) -> None:
    """The derivation is a record of a gap, so the gap must still be open.

    The day a database here gains a real topology declaration, its row stops
    being true and has to be deleted rather than left to contradict the
    topology. That deletion is what shrinking the undeclared set means.
    """
    for instance in _TOPOLOGY_INSTANCES:
        declared = {
            database.physical_name
            for database in _load_topology(instance).databases.values()
        }
        assert entry.physical_name not in declared, (
            f"{instance}: {entry.physical_name} now HAS a topology declaration, "
            "so its derivation row is stale. Delete the row and pin its CONNECT "
            "allowlist in _DECLARED_CONNECT instead."
        )


def test_the_derivation_covers_the_undeclared_set_exactly() -> None:
    """The table and the gap are the same set, in both directions.

    A derivation row for a database that is not in the deployment scope is dead
    weight; a scope database with no row is the unanswered question returning.
    """
    derived = {entry.physical_name for entry in _UNDECLARED_DERIVATION}
    assert derived == set(_UNDECLARED_DEPLOYMENT_DATABASES)
    assert derived <= set(_DEPLOYMENT_CONNECT_DATABASES), (
        "a derivation row names a database outside the approved deployment "
        f"CONNECT scope: {sorted(derived - set(_DEPLOYMENT_CONNECT_DATABASES))!r}"
    )
    assert len({entry.physical_name for entry in _UNDECLARED_DERIVATION}) == len(
        _UNDECLARED_DERIVATION
    ), "a database is derived twice"
