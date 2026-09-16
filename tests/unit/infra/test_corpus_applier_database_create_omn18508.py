# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18508: a role that applies its own database's corpus needs CREATE on it.

``omninode_cloud`` is the one database on the compose lanes whose migration
corpus is applied by something other than the superuser:
``docker-compose.dev-lane.yml``'s ``cloud-migration`` one-shot sets
``DB_USER: role_omninode``, and ``docker/migrations/cloud/run-cloud-migrations.sh``
passes that through to ``psql -U`` with ``ON_ERROR_STOP=1``.

The corpus opens with ``CREATE EXTENSION IF NOT EXISTS pgcrypto``. ``pgcrypto``
is ``trusted = t``, but **trusted only waives the superuser requirement, not the
privilege requirement**: a non-superuser still needs ``CREATE`` on the current
DATABASE to install it. ``CREATE`` on schema ``public`` is a different privilege
and does not cover it.

Measured on the .201 compose dev lane, Postgres 16.15, 2026-09-16, read-only::

    has_database_privilege('role_omninode','omninode_cloud','CREATE')  -> f
    has_database_privilege('postgres',     'omninode_cloud','CREATE')  -> t   <- positive control
    has_database_privilege('role_omninode','omninode_cloud','CONNECT') -> t
    has_schema_privilege  ('role_omninode','public','CREATE')          -> t
    tables in omninode_cloud.public                                     -> 0
    tables in omnidash_analytics.public                                 -> 71  <- positive control

Every database on that lane is owned by ``postgres``; none is owned by its
service role. So the fix is a database-scoped ``GRANT CREATE``, not an ownership
change -- ownership would additionally confer ``DROP DATABASE`` on a principal
deliberately pinned ``NOSUPERUSER NOBYPASSRLS NOCREATEDB``, which is the identity
``onex-api`` itself connects as.

The grant has to land in BOTH provisioning seams. The bootstrap
(``000_create_multiple_databases.sh``) runs from ``/docker-entrypoint-initdb.d``
only when the data directory is empty, so it reaches fresh volumes only; the
warm path is ``scripts/run-forward-migrations.sh``, which the
``forward-migration`` one-shot runs on every compose up. A grant in one seam only
provisions half the lanes and does so silently.

These assertions are static by design -- they fire on hosts with no Docker and no
Postgres, which is where a silent revert would otherwise survive until the next
lane rediscovers it by hand.

Ticket: OMN-18508. Parent: OMN-18421. Peer: OMN-18438 (the warm seam this
extends), OMN-18475 (the migrate-image half of the same blocked apply).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]

DOCKER_DIR = REPO_ROOT / "docker"
BOOTSTRAP = DOCKER_DIR / "migrations" / "forward" / "000_create_multiple_databases.sh"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"

PRINCIPAL = "role_omninode"
TARGET_DATABASE = "omninode_cloud"
MAP_ENTRY = f"{TARGET_DATABASE}:{PRINCIPAL}"

# Both seams carry their own boundary markers so a slice cannot silently widen to
# the whole file and make every assertion below pass for the wrong reason.
BOOTSTRAP_BEGIN = "# ---- BEGIN corpus-applier database CREATE seam (OMN-18508) ----"
BOOTSTRAP_END = "# ---- END corpus-applier database CREATE seam (OMN-18508) ----"
RUNNER_BEGIN = "# ---- BEGIN corpus-applier database CREATE seam (OMN-18508) ----"
RUNNER_END = "# ---- END corpus-applier database CREATE seam (OMN-18508) ----"

# The shared grant helper every SERVICE_DB_MAP principal passes through. The
# database-level CREATE must NOT be added here: five of the six principals have
# their corpus applied by the superuser and need nothing.
SHARED_GRANT_FUNCTION = "grant_role_to_database() {"

# A compose environment value naming the superuser resolves to the cluster
# superuser, whatever the lane sets POSTGRES_USER to. Anything else is a
# distinct, non-superuser applying identity.
SUPERUSER_TOKENS = ("POSTGRES_USER", "postgres")

# Environment keys a migration one-shot names its applying identity with.
IDENTITY_KEYS = ("DB_USER", "POSTGRES_USER")
# ...and the keys it names the database(s) it applies to with.
DATABASE_KEYS = ("DB_NAME", "POSTGRES_DB", "NODE_POSTGRES_DB")

MIGRATION_ENTRYPOINT_RE = re.compile(r"run-[a-z0-9-]+-migrations\.sh")


def _load_compose(path: Path) -> dict[str, Any]:
    """Parse a compose file, tolerating compose's ``!override`` tag."""
    text = path.read_text(encoding="utf-8").replace("!override", "")
    return yaml.safe_load(text) or {}


def _slice(path: Path, begin: str, end: str) -> str:
    """The script text between a seam's own markers.

    Fails loudly rather than returning the whole file when a marker is missing.
    """
    text = path.read_text(encoding="utf-8")
    assert begin in text, (
        f"{path.name} carries no {begin!r} marker -- the OMN-18508 seam is "
        f"absent, so {PRINCIPAL} never gains CREATE on {TARGET_DATABASE} and the "
        "corpus apply dies on its first CREATE EXTENSION"
    )
    assert end in text, f"{path.name} carries no {end!r} marker"
    return text[text.index(begin) : text.index(end)]


def _map_entries(text: str) -> set[str]:
    """Every ``database:role`` pair quoted inside an OMN-18508 seam slice."""
    return set(re.findall(r'"([a-z0-9_]+:[a-z0-9_]+)"', text))


def _shared_grant_function_body() -> str:
    """The bootstrap's ``grant_role_to_database()``, which all six roles use."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert SHARED_GRANT_FUNCTION in text, (
        f"{BOOTSTRAP.name} no longer declares {SHARED_GRANT_FUNCTION!r} -- this "
        "test's narrowness assertion has lost its subject"
    )
    start = text.index(SHARED_GRANT_FUNCTION)
    # The next top-level function declaration ends it.
    rest = text[start + len(SHARED_GRANT_FUNCTION) :]
    next_fn = re.search(r"\n[a-z_]+\(\) \{", rest)
    return rest[: next_fn.start()] if next_fn else rest


# ---------------------------------------------------------------------------
# AC1 -- the fresh-volume seam, and its narrowness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_bootstrap_declares_the_corpus_applier_map() -> None:
    """AC1: the principals that apply their own corpus are named explicitly."""
    seam = _slice(BOOTSTRAP, BOOTSTRAP_BEGIN, BOOTSTRAP_END)

    assert MAP_ENTRY in _map_entries(seam), (
        f"{BOOTSTRAP.name}'s OMN-18508 seam does not carry {MAP_ENTRY!r} -- "
        f"{PRINCIPAL} applies the {TARGET_DATABASE} corpus and must hold CREATE "
        "on that database"
    )


@pytest.mark.unit
def test_bootstrap_grants_database_create_and_reads_it_back() -> None:
    """AC1/AC2: grant, then prove the grant took.

    A ``GRANT`` issued without grant option on the object warns and returns
    success rather than raising -- the trap OMN-18060's and OMN-18438's seams
    both document. The readback is the only thing separating a real grant from a
    silent no-op.
    """
    seam = _slice(BOOTSTRAP, BOOTSTRAP_BEGIN, BOOTSTRAP_END)

    assert re.search(r'GRANT\s+CREATE\s+ON\s+DATABASE\s+"?\$', seam), (
        f"{BOOTSTRAP.name}'s OMN-18508 seam issues no GRANT CREATE ON DATABASE"
    )
    assert "has_database_privilege" in seam and "'CREATE'" in seam, (
        f"{BOOTSTRAP.name}'s OMN-18508 seam does not read the CREATE grant back "
        "-- an ungranted run would log success"
    )


@pytest.mark.unit
def test_bootstrap_does_not_widen_every_service_role() -> None:
    """AC1: least privilege -- five of the six principals must not gain this.

    ``grant_role_to_database()`` runs for every ``SERVICE_DB_MAP`` entry. The
    other five databases have their corpus applied by the superuser, so a
    database-level CREATE there widens five roles for nothing.
    """
    body = _shared_grant_function_body()
    executable = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )

    assert not re.search(r"GRANT\s+CREATE\s+ON\s+DATABASE", executable), (
        f"{BOOTSTRAP.name}'s shared grant_role_to_database() now grants CREATE "
        "on the database to EVERY service role -- issue it from the "
        "corpus-applier seam instead, which names only the principals that need it"
    )


# ---------------------------------------------------------------------------
# AC2 -- the warm-volume seam, and parity with the fresh one
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_runner_seam_grants_database_create_and_reads_it_back() -> None:
    """AC2: the warm path re-asserts the same grant on every compose up.

    The bootstrap runs only on an empty data directory. Every lane that exists
    today has a warm volume, so a bootstrap-only fix reaches none of them.
    """
    seam = _slice(RUNNER, RUNNER_BEGIN, RUNNER_END)

    assert re.search(r'GRANT\s+CREATE\s+ON\s+DATABASE\s+"?\$', seam), (
        f"{RUNNER.name}'s OMN-18508 seam issues no GRANT CREATE ON DATABASE -- "
        "no existing lane volume would ever gain the privilege"
    )
    assert "has_database_privilege" in seam and "'CREATE'" in seam, (
        f"{RUNNER.name}'s OMN-18508 seam does not read the CREATE grant back"
    )


@pytest.mark.unit
def test_runner_seam_is_idempotent_and_skips_an_absent_database_by_name() -> None:
    """AC2: re-running on a provisioned lane re-asserts; a lane without the
    database is a legitimate state, named rather than fatal.

    ``GRANT`` is idempotent in Postgres, so re-assertion is safe by construction;
    what is not safe is turning the migration gate UNHEALTHY on a lane whose
    volume predates the database, or on one that never provisioned the role.
    """
    seam = _slice(RUNNER, RUNNER_BEGIN, RUNNER_END)

    assert "pg_database" in seam, (
        f"{RUNNER.name}'s OMN-18508 seam does not check the database exists "
        "before granting on it"
    )
    assert "pg_roles" in seam or "pg_catalog.pg_roles" in seam, (
        f"{RUNNER.name}'s OMN-18508 seam does not check the role exists -- a "
        "lane with no credential provisioned would fail rather than skip"
    )
    assert "skip" in seam, f"{RUNNER.name}'s OMN-18508 seam has no named-skip branch"


@pytest.mark.unit
def test_the_two_seams_declare_the_same_principals() -> None:
    """AC1/AC2: parity, the same contract OMN-16993 pins for the login-only map.

    A principal in one seam and not the other is provisioned on fresh volumes
    and not on warm ones, or the reverse -- and the asymmetry is invisible until
    a lane rediscovers it.
    """
    bootstrap_entries = _map_entries(_slice(BOOTSTRAP, BOOTSTRAP_BEGIN, BOOTSTRAP_END))
    runner_entries = _map_entries(_slice(RUNNER, RUNNER_BEGIN, RUNNER_END))

    assert bootstrap_entries == runner_entries, (
        "the OMN-18508 corpus-applier maps have diverged: "
        f"{BOOTSTRAP.name} carries {sorted(bootstrap_entries)}, "
        f"{RUNNER.name} carries {sorted(runner_entries)}"
    )


@pytest.mark.unit
def test_runner_seam_sits_outside_the_login_only_credential_phase() -> None:
    """AC2: OMN-16993's test asserts that phase contains no GRANT at all."""
    text = RUNNER.read_text(encoding="utf-8")
    login_phase_end = text.index("Ensuring service migration ledger exists")

    assert text.index(RUNNER_BEGIN) > login_phase_end, (
        "the OMN-18508 seam starts inside the OMN-16993 credential phase, whose "
        "test asserts that phase contains no GRANT -- move it after that phase"
    )


# ---------------------------------------------------------------------------
# AC5 -- the invariant, derived rather than restated
# ---------------------------------------------------------------------------


def _migration_appliers() -> list[tuple[Path, str, str, list[str]]]:
    """Every compose migration one-shot, as ``(file, service, identity, dbs)``.

    Derived by PARSING each compose file: a service is a migration applier when
    its entrypoint or command runs a ``run-*-migrations.sh``. Its applying
    identity and its target databases are read off its own environment, never
    restated here -- a test that hardcoded them would keep passing after someone
    pointed a one-shot at a different role.
    """
    appliers: list[tuple[Path, str, str, list[str]]] = []

    for compose_file in sorted(DOCKER_DIR.glob("docker-compose*.yml")):
        compose = _load_compose(compose_file)
        for service_name, service in (compose.get("services") or {}).items():
            if not isinstance(service, dict):
                continue
            invocation_tokens: list[str] = []
            for key in ("entrypoint", "command"):
                declared = service.get(key)
                if declared is None:
                    continue
                if isinstance(declared, list):
                    invocation_tokens.extend(str(token) for token in declared)
                else:
                    invocation_tokens.append(str(declared))
            invocation = " ".join(invocation_tokens)
            if not MIGRATION_ENTRYPOINT_RE.search(invocation):
                continue

            environment = service.get("environment") or {}
            if not isinstance(environment, dict):
                continue

            identity = next(
                (str(environment[key]) for key in IDENTITY_KEYS if key in environment),
                "",
            )
            databases = [
                str(environment[key]) for key in DATABASE_KEYS if key in environment
            ]
            appliers.append((compose_file, service_name, identity, databases))

    return appliers


def _is_superuser_identity(identity: str) -> bool:
    """True when the declared identity resolves to the cluster superuser."""
    return any(token in identity for token in SUPERUSER_TOKENS)


@pytest.mark.unit
def test_the_derivation_finds_the_migration_one_shots() -> None:
    """AC5 control: a derivation that found nothing would pass everything.

    Without this, a regex that stopped matching would turn the invariant below
    into a vacuous green -- the exact false zero the sweep rules forbid.
    """
    appliers = _migration_appliers()

    assert len(appliers) >= 3, (
        "the compose parse found fewer than three migration one-shots "
        f"({[f'{f.name}:{s}' for f, s, _, _ in appliers]}) -- the derivation is "
        "broken, so the invariant below is not being checked at all"
    )
    assert any(
        service == "cloud-migration" and identity == PRINCIPAL
        for _, service, identity, _ in appliers
    ), (
        "the parse did not find cloud-migration applying as "
        f"{PRINCIPAL} -- either the one-shot was repointed or the parse is wrong"
    )


@pytest.mark.unit
def test_a_superuser_applier_is_present_and_needs_no_grant() -> None:
    """AC5 positive control: the invariant must distinguish, not pass everything.

    ``forward-migration`` and ``intelligence-migration`` apply as the superuser
    and correctly appear in neither seam's map. A test that demanded a grant for
    every applier, or granted one to every applier, would be indistinguishable
    from one that checks.
    """
    appliers = _migration_appliers()
    superuser_appliers = [
        (compose_file, service)
        for compose_file, service, identity, _ in appliers
        if _is_superuser_identity(identity)
    ]

    assert superuser_appliers, (
        "no migration one-shot applies as the superuser -- the control this "
        "test exists to provide is gone"
    )

    mapped_databases = {
        entry.split(":", 1)[0]
        for entry in _map_entries(_slice(BOOTSTRAP, BOOTSTRAP_BEGIN, BOOTSTRAP_END))
    }
    for compose_file, service, identity, databases in appliers:
        if not _is_superuser_identity(identity):
            continue
        for database in databases:
            assert database not in mapped_databases, (
                f"{compose_file.name}'s {service} applies {database} as the "
                f"superuser ({identity}), yet the corpus-applier map grants a "
                "role CREATE on it -- that widens a principal for nothing"
            )


@pytest.mark.unit
def test_every_non_superuser_applier_holds_database_create() -> None:
    """AC5: the invariant itself, across both seams.

    A database whose corpus is applied by a non-superuser identity must be
    provisioned with ``CREATE`` on that database for that identity, by the
    fresh-volume seam AND by the warm-volume seam.
    """
    bootstrap_entries = _map_entries(_slice(BOOTSTRAP, BOOTSTRAP_BEGIN, BOOTSTRAP_END))
    runner_entries = _map_entries(_slice(RUNNER, RUNNER_BEGIN, RUNNER_END))

    uncovered: list[str] = []
    checked = 0
    for compose_file, service, identity, databases in _migration_appliers():
        if _is_superuser_identity(identity) or not identity:
            continue
        for database in databases:
            checked += 1
            required = f"{database}:{identity}"
            if required not in bootstrap_entries:
                uncovered.append(
                    f"{compose_file.name}:{service} -> {required} missing from "
                    f"{BOOTSTRAP.name} (fresh volumes unprovisioned)"
                )
            if required not in runner_entries:
                uncovered.append(
                    f"{compose_file.name}:{service} -> {required} missing from "
                    f"{RUNNER.name} (every existing warm volume unprovisioned)"
                )

    assert checked, (
        "no non-superuser migration applier was found -- cloud-migration applies "
        f"as {PRINCIPAL}, so a zero here means the derivation stopped working"
    )
    assert not uncovered, (
        "a database is applied by a role that is never granted CREATE on it, so "
        "its corpus dies on CREATE EXTENSION:\n  " + "\n  ".join(uncovered)
    )


# ---------------------------------------------------------------------------
# AC4 -- the prose stops asserting something the owner table contradicts
# ---------------------------------------------------------------------------

# Files that described role_omninode as the database's owner before OMN-18508.
OWNERSHIP_PROSE_FILES = (
    DOCKER_DIR / "docker-compose.dev-lane.yml",
    DOCKER_DIR / "docker-compose.infra.yml",
    DOCKER_DIR / "docker-compose.judge.yml",
    DOCKER_DIR / "catalog" / "services" / "cloud-migration.yaml",
    DOCKER_DIR / "migrations" / "cloud" / "run-cloud-migrations.sh",
    BOOTSTRAP,
)

OWNERSHIP_CLAIM = "owning login"


@pytest.mark.unit
@pytest.mark.parametrize("path", OWNERSHIP_PROSE_FILES, ids=lambda p: p.name)
def test_ownership_is_mentioned_never_asserted(path: Path) -> None:
    """AC4: measured, every database on the lane is owned by ``postgres``.

    The phrase survives only inside quotation marks, where these files explain
    that the earlier description was wrong and what the role actually holds.
    An unquoted occurrence is the file asserting it again.
    """
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if OWNERSHIP_CLAIM not in line:
            continue
        quoted = re.search(rf'"[^"]*{OWNERSHIP_CLAIM}[^"]*"', line)
        assert quoted, (
            f"{path.name}:{number} asserts that a service role is the database's "
            f"{OWNERSHIP_CLAIM}: {line.strip()!r}. Measured on the .201 dev lane, "
            "every database is owned by postgres; the role holds CONNECT, schema "
            "USAGE + CREATE and database CREATE. State that instead."
        )


@pytest.mark.unit
def test_the_ownership_prose_control_has_a_subject() -> None:
    """AC4 control: a corpus with no occurrences would pass the test above
    without checking anything. At least one file must still explain the
    correction, so the assertion has something to discriminate on.
    """
    mentioning = [
        path.name
        for path in OWNERSHIP_PROSE_FILES
        if OWNERSHIP_CLAIM in path.read_text(encoding="utf-8")
    ]

    assert mentioning, (
        "no file records what the ownership claim was corrected from -- the "
        "correction was deleted rather than made, so the next reader has no way "
        "to tell the current prose is deliberate"
    )


# ---------------------------------------------------------------------------
# AC6 -- a grant that lands after its only consumer has exited is not a grant
# ---------------------------------------------------------------------------

# The service whose entrypoint runs the warm-volume provisioning seams.
PROVISIONING_ENTRYPOINT = "run-forward-migrations.sh"


def _provisioning_service() -> str:
    """The service that runs the warm-volume provisioning seams.

    Resolved across the whole compose set, not per file: a lane overlay
    redeclares this service to hand it more environment and inherits the
    entrypoint from the base file, so a per-file lookup finds it in the base
    and not in the overlay that carries the applier.
    """
    for compose_file in sorted(DOCKER_DIR.glob("docker-compose*.yml")):
        compose = _load_compose(compose_file)
        for service_name, service in (compose.get("services") or {}).items():
            if not isinstance(service, dict):
                continue
            declared = [service.get("entrypoint"), service.get("command")]
            if PROVISIONING_ENTRYPOINT in str(declared):
                return str(service_name)
    return ""


@pytest.mark.unit
def test_a_non_superuser_applier_waits_for_its_own_provisioning() -> None:
    """AC6: ordering, measured rather than assumed.

    ``cloud-migration`` declared ``postgres: service_healthy`` and nothing else
    that could order it against the seam which mints its identity's credential
    (OMN-18438) and grants its database CREATE (OMN-18508). Both one-shots then
    start as soon as Postgres reports healthy, and which of them wins is a
    scheduling coin flip.

    Measured on the .201 dev lane, 2026-09-16T22:13:58Z, on the first rebuild
    after the grant landed: the corpus apply raised ``permission denied to
    create extension "pgcrypto"`` at .122Z, and the grant it needed was issued
    at .416Z -- 294 ms later, in the same bring-up. The grant was correct and
    arrived after its only consumer had already exited 3.

    A superuser applier needs no such edge: nothing provisions the superuser.
    """
    provisioner = _provisioning_service()
    assert provisioner, (
        "no compose service runs "
        f"{PROVISIONING_ENTRYPOINT} -- the warm-volume seams have no host, so "
        "this test has nothing to order against"
    )

    checked = 0
    for compose_file, service_name, identity, _ in _migration_appliers():
        if _is_superuser_identity(identity) or not identity:
            continue

        service = (_load_compose(compose_file).get("services") or {})[service_name]
        checked += 1
        depends_on = service.get("depends_on") or {}
        assert isinstance(depends_on, dict), (
            f"{compose_file.name}'s {service_name} uses the list form of "
            "depends_on, which cannot express a completion condition"
        )
        declared = depends_on.get(provisioner)
        assert declared is not None, (
            f"{compose_file.name}'s {service_name} applies a corpus as "
            f"{identity} but does not wait for {provisioner}, which is what "
            f"provisions {identity}. Both start on postgres being healthy, so "
            "the apply can win the race and fail on a privilege that is granted "
            "milliseconds later."
        )
        assert declared.get("condition") == "service_completed_successfully", (
            f"{compose_file.name}'s {service_name} waits on {provisioner} with "
            f"condition {declared.get('condition')!r}. Only "
            "service_completed_successfully proves the seams ran; a started or "
            "healthy condition still races the grants they issue."
        )

    assert checked, (
        "no non-superuser applier was checked for ordering -- either the "
        "derivation or the provisioning-service lookup stopped working, and a "
        "zero here would read as a clean bill of health"
    )
