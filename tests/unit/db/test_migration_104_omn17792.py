# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 104 -- the ``validator_ro`` cluster role (OMN-17792).

Two halves, because the two failure modes this file guards against are not
detectable by the same instrument.

**The static half** pins DELIVERABILITY. 104 is the third file in this corpus to
provision a non-owner principal for ``omnidash_analytics``, and the two before it
both left a tombstone:

* ``096`` and ``097`` carry a ``\\connect omnidash_analytics``. The k8s Job that
  applies the flat corpus (``omninode_infra``,
  ``k8s/migrations/omnibase-infra-migrate.yaml``) gates its ``psql -f`` loop on
  ``directive_db == "$DB_NAME"``, where ``DB_NAME`` is ``omnibase_infra``. Both
  files were therefore UNREACHABLE on the RDS lane, and both accreted a false
  "applied" ledger row that masked it for months (OMN-15819 / OMN-15846).
* A NEW cross-database flat file is now a hard, fail-closed reject
  (``scripts/ci/check_flat_migration_foreign_connect.py``; the manifest is a
  closed ledger frozen to those five filenames), so the mistake cannot be
  repeated by accident -- but a reviewer reading 104 should be able to see that
  it was not attempted, and see WHERE the per-database half actually rides.

**The executed half** pins BEHAVIOUR, against a real PostgreSQL shaped like the
managed lane: two ``NOCREATEROLE`` migration identities, and
``omnidash_analytics`` owned by ``role_omnidash`` rather than by the flat loop's
``role_omnibase_infra``. That ownership split is what makes OMN-17301's D2 and D3
observable at all -- on a scratch database owned by the executing role, a GRANT
that would silently no-op on RDS succeeds and everything looks correct. A purely
textual assertion about this file could not catch either.

Ticket: OMN-17792 (AC6, RDS half). Class: OMN-17301 (D1/D2/D3), OMN-15343 (the
provisioning seam), OMN-15819 (the undeliverable-flat-file class).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
MIGRATION = FORWARD_DIR / "104_create_validator_ro_role.sql"
CROSS_DB_MANIFEST = FORWARD_DIR / "cross-database-flat-migrations.yaml"
ROLE = "validator_ro"
IMAGE = "postgres:16-alpine"
# Path INSIDE the ephemeral container, not on the host filesystem: the container
# is created and destroyed by the fixture and is not shared, so the usual
# predictable-temp-path attack has no reachable surface here.
CONTAINER_SQL = "/tmp/104.sql"  # noqa: S108 - path inside the ephemeral container


# ---------------------------------------------------------------------------
# Static: deliverability and least privilege
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_migration_exists() -> None:
    assert MIGRATION.is_file(), f"{MIGRATION.name} is missing from the flat corpus"


@pytest.mark.unit
def test_migration_issues_no_connect_meta_command() -> None:
    """The 096/097 mistake, asserted as absence rather than assumed."""
    text = MIGRATION.read_text(encoding="utf-8")
    offending = [
        line
        for line in text.splitlines()
        if re.match(r"^\s*\\connect\b", line) is not None
    ]
    assert offending == [], (
        "104 carries a \\connect meta-command. A flat migration whose \\connect "
        "names a database other than omnibase_infra has NO execution path in the "
        "k8s Job that applies this corpus -- that is exactly what made 096 and 097 "
        "undeliverable, and what their tombstones record. The per-database half "
        "belongs in the node-owned loop, which connects to omnidash_analytics "
        "directly."
    )


@pytest.mark.unit
def test_migration_is_not_on_the_cross_database_manifest() -> None:
    manifest = yaml.safe_load(CROSS_DB_MANIFEST.read_text(encoding="utf-8"))
    listed = {entry["file"] for entry in manifest["entries"]}
    assert MIGRATION.name not in listed, (
        "104 was added to the cross-database manifest. That manifest is a CLOSED "
        "ledger of pre-existing undeliverable files; adding an entry is not a way "
        "to authorise a new one."
    )


@pytest.mark.unit
def test_migration_creates_the_role_with_every_flag_pinned_off() -> None:
    text = MIGRATION.read_text(encoding="utf-8")
    create = re.search(
        rf"CREATE\s+ROLE\s+{ROLE}\s+WITH([^;]*);", text, flags=re.IGNORECASE
    )
    assert create is not None, f"104 does not CREATE ROLE {ROLE}"
    attributes = create.group(1).upper()
    for flag in (
        "NOLOGIN",
        "NOSUPERUSER",
        "NOBYPASSRLS",
        "NOCREATEDB",
        "NOCREATEROLE",
        "NOREPLICATION",
    ):
        assert flag in attributes, f"CREATE ROLE {ROLE} does not pin {flag}"


@pytest.mark.unit
def test_migration_grants_connect_and_no_other_privilege() -> None:
    text = MIGRATION.read_text(encoding="utf-8")
    grants = re.findall(r"^\s*GRANT\s+(.+?)\s+ON\b", text, flags=re.MULTILINE)
    assert grants, "104 issues no GRANT at all"
    for privileges in grants:
        assert privileges.strip().upper() == "CONNECT", (
            f"104 confers {privileges!r}. Only the database-level CONNECT "
            "privilege is cluster-wide and therefore deliverable by this file. "
            "Schema and relation privileges are per-database and unreachable "
            "from the omnibase_infra connection this file runs on."
        )


def _statements() -> str:
    """The file with `--` comment lines stripped.

    The header discusses the credential attach and the RDS master secret by
    name, so a whole-file substring scan for those words would fail on its own
    documentation. What must be absent is a credential in an EXECUTED statement.
    """
    return "\n".join(
        line
        for line in MIGRATION.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("--")
    ).upper()


@pytest.mark.unit
def test_migration_carries_no_credential_material() -> None:
    statements = _statements()
    for forbidden in ("PASSWORD", "ENCRYPTED"):
        assert forbidden not in statements, (
            f"104 executes a statement mentioning {forbidden}. LOGIN + password "
            "is a deployment-owned attach by reference; no credential material "
            "ever lives in a migration, and the holder is given ACCESS to the "
            "store path rather than the value."
        )


@pytest.mark.unit
def test_migration_grants_no_login() -> None:
    statements = _statements()
    assert (
        re.search(r"\bALTER\s+ROLE\s+VALIDATOR_RO\s+[^;]*\bLOGIN\b", statements) is None
    ), (
        "104 grants LOGIN. That is the deployment-owned attach step, not a "
        "migration's to take."
    )


@pytest.mark.unit
def test_migration_never_reasserts_nologin() -> None:
    # 094's finding: re-asserting NOLOGIN on a pre-existing role REVOKES the
    # deployment-owned credential attach as a side effect of recording a
    # migration. NOLOGIN is a CREATE-time default here and nothing else.
    text = MIGRATION.read_text(encoding="utf-8")
    alters = re.findall(rf"ALTER\s+ROLE\s+{ROLE}\s+([^;]*);", text, flags=re.IGNORECASE)
    for clause in alters:
        assert "NOLOGIN" not in clause.upper(), (
            "104 re-asserts NOLOGIN in an ALTER ROLE, which would revoke the "
            "credential attach that makes the role usable (094's invariant)."
        )


@pytest.mark.unit
def test_migration_distinguishes_an_explicit_grant_from_publics_default() -> None:
    # OMN-17301 D3: has_database_privilege(...,'CONNECT') is TRUE via PUBLIC's
    # default on any database that has not revoked it, so it cannot detect a
    # GRANT that silently did nothing (D2).
    text = MIGRATION.read_text(encoding="utf-8")
    assert "aclexplode" in text, (
        "104 does not read datacl through aclexplode, so it cannot tell an "
        "explicit role grant from PUBLIC's default -- the vacuous-assertion "
        "defect OMN-17301 D3 names."
    )


# ---------------------------------------------------------------------------
# Executed: the file applied against a managed-lane-shaped PostgreSQL
# ---------------------------------------------------------------------------


def _docker() -> str:
    binary = shutil.which("docker")
    if binary is None:
        pytest.skip("docker is not available; this module executes real SQL")
    return binary


@pytest.fixture(scope="module")
def pg() -> Iterator[str]:
    """A scratch PostgreSQL shaped like the managed lane. Yields the container."""
    docker = _docker()
    name = f"omn17792-{uuid.uuid4().hex[:10]}"
    try:
        subprocess.run(
            [
                docker,
                "run",
                "-d",
                "--name",
                name,
                "-e",
                "POSTGRES_PASSWORD=scratch",
                IMAGE,
            ],
            check=True,
            capture_output=True,
            timeout=180,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        pytest.skip(f"could not start {IMAGE}: {exc}")

    try:
        for _ in range(60):
            probe = subprocess.run(
                [docker, "exec", name, "pg_isready", "-U", "postgres"],
                capture_output=True,
                timeout=30,
                check=False,  # not-ready is the expected loop condition
            )
            if probe.returncode == 0:
                break
            time.sleep(1)
        else:  # pragma: no cover
            pytest.skip("scratch postgres never became ready")

        for statement in (
            "CREATE ROLE role_omnibase_infra LOGIN PASSWORD 'scratch' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE;",
            "CREATE ROLE role_omnidash LOGIN PASSWORD 'scratch' "
            "NOSUPERUSER NOCREATEDB NOCREATEROLE;",
            "CREATE DATABASE omnibase_infra OWNER role_omnibase_infra;",
            "CREATE DATABASE omnidash_analytics OWNER role_omnidash;",
        ):
            subprocess.run(
                [
                    docker,
                    "exec",
                    "-i",
                    name,
                    "psql",
                    "-U",
                    "postgres",
                    "-q",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "-c",
                    statement,
                ],
                check=True,
                capture_output=True,
                timeout=60,
            )

        subprocess.run(
            [docker, "cp", str(MIGRATION), f"{name}:{CONTAINER_SQL}"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        yield name
    finally:
        subprocess.run(
            [docker, "rm", "-f", name],
            capture_output=True,
            timeout=120,
            check=False,  # best-effort teardown must never mask a test failure
        )


def _psql(
    container: str, sql: str, *, user: str = "postgres", db: str = "postgres"
) -> str:
    result = subprocess.run(
        [
            _docker(),
            "exec",
            "-i",
            "-e",
            "PGPASSWORD=scratch",
            container,
            "psql",
            "-h",
            "127.0.0.1",
            "-U",
            user,
            "-d",
            db,
            "-tAc",
            sql,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return result.stdout.strip()


def _apply(container: str, *, user: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _docker(),
            "exec",
            "-i",
            "-e",
            "PGPASSWORD=scratch",
            container,
            "psql",
            "-h",
            "127.0.0.1",
            "-U",
            user,
            "-d",
            "omnibase_infra",
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            CONTAINER_SQL,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # the failing path is asserted by the caller
    )


def _drop_role(container: str) -> None:
    # `DROP ROLE` fails while any database ACL still names the role -- including
    # the CONNECT grant this very migration issues -- so the dependent
    # privileges are dropped per database first, exactly as the 103 fixture does.
    for db in ("omnibase_infra", "omnidash_analytics", "postgres"):
        subprocess.run(
            [
                _docker(),
                "exec",
                "-i",
                container,
                "psql",
                "-U",
                "postgres",
                "-q",
                "-d",
                db,
                "-c",
                f"DROP OWNED BY {ROLE} CASCADE;",
            ],
            capture_output=True,
            timeout=60,
            check=False,  # the role may not exist or may own nothing; both fine
        )
    subprocess.run(
        [
            _docker(),
            "exec",
            "-i",
            container,
            "psql",
            "-U",
            "postgres",
            "-q",
            "-c",
            f"DROP ROLE IF EXISTS {ROLE};",
        ],
        capture_output=True,
        timeout=60,
        check=False,  # IF EXISTS makes absence a success
    )


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.postgres
def test_absent_role_fails_with_a_named_remediation_not_a_raw_permission_error(
    pg: str,
) -> None:
    """OMN-17301 D1, replayed for 104.

    The migrate Job runs at migration-order 1 of 6, before overlay-apply and the
    runtime digest pin, so an abort here blocks EVERY staging deploy on every
    trigger. It still aborts -- refusing to record itself over a missing
    principal is deliberate -- but it must say what to do about it.
    """
    _drop_role(pg)
    result = _apply(pg, user="role_omnibase_infra")

    assert result.returncode != 0, (
        "104 recorded itself while validator_ro did not exist. A silently-absent "
        "principal is the OMN-14950 masking outcome, strictly worse than an abort."
    )
    combined = result.stdout + result.stderr
    assert "provision-cluster-roles.sh" in combined, (
        "the failure does not name the provisioning seam that holds CREATEROLE; "
        f"got: {combined[-600:]}"
    )
    assert "OMN-17792" in combined


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.postgres
def test_applies_cleanly_once_the_seam_has_provisioned_the_role(pg: str) -> None:
    """The real managed-lane steady state: the seam creates it, 104 is a no-op.

    This is the disposition OMN-15343's runner branch was written to let
    through, and it is the only one that lets a deploy succeed on RDS.
    """
    _drop_role(pg)
    # Exactly what omninode_infra scripts/provision-cluster-roles.sh does.
    _psql(
        pg,
        f"CREATE ROLE {ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS "
        "NOCREATEDB NOCREATEROLE NOREPLICATION;",
    )
    _psql(pg, f"GRANT CONNECT ON DATABASE omnidash_analytics TO {ROLE};")

    result = _apply(pg, user="role_omnibase_infra")
    assert result.returncode == 0, (
        "104 failed against a correctly-provisioned role under the ordinary, "
        f"non-CREATEROLE migration identity: {(result.stdout + result.stderr)[-800:]}"
    )
    assert "explicit grant" in (result.stdout + result.stderr)


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.postgres
def test_an_escalated_flag_is_fatal_and_names_the_flag(pg: str) -> None:
    """NOBYPASSRLS is the whole point of the role, so an observed escalation is
    fatal whether or not the executing identity can correct it."""
    _drop_role(pg)
    _psql(pg, f"CREATE ROLE {ROLE} NOLOGIN NOSUPERUSER BYPASSRLS;")
    _psql(pg, f"GRANT CONNECT ON DATABASE omnidash_analytics TO {ROLE};")

    result = _apply(pg, user="role_omnibase_infra")
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "BYPASSRLS" in combined.upper()

    _psql(pg, f"ALTER ROLE {ROLE} NOBYPASSRLS;")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.postgres
def test_the_role_can_read_nothing_until_the_per_database_half_lands(pg: str) -> None:
    """The honest boundary of this file, executed rather than asserted in prose.

    104 delivers a cluster role and CONNECT. It delivers NO schema USAGE and NO
    relation SELECT, because those are per-database and this file runs on the
    omnibase_infra connection. Anyone reading "the read-only role migration
    landed" should be able to see here that reading still requires the
    node-owned half.
    """
    _drop_role(pg)
    _psql(
        pg,
        f"CREATE ROLE {ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS "
        "NOCREATEDB NOCREATEROLE NOREPLICATION;",
    )
    _psql(pg, f"GRANT CONNECT ON DATABASE omnidash_analytics TO {ROLE};")
    _psql(
        pg,
        "CREATE TABLE IF NOT EXISTS public.omn17792_probe (id int);",
        user="role_omnidash",
        db="omnidash_analytics",
    )

    assert _apply(pg, user="role_omnibase_infra").returncode == 0

    readable = _psql(
        pg,
        f"SELECT has_table_privilege('{ROLE}', 'public.omn17792_probe', 'SELECT');",
        db="omnidash_analytics",
    )
    assert readable == "f", (
        "104 granted a relation privilege. It cannot: it is connected to "
        "omnibase_infra. If this passes, the file has acquired a \\connect and is "
        "undeliverable on the managed lane."
    )
    usable = _psql(
        pg,
        f"SELECT has_schema_privilege('{ROLE}', 'public', 'USAGE');",
        db="omnidash_analytics",
    )
    # PUBLIC holds USAGE on `public` by default on PostgreSQL 16, so this reads
    # `t` for reasons that have nothing to do with 104. Asserted as the *source*
    # instead, the same way the CONNECT readback distinguishes the two.
    assert usable == "t"
    # S608 below is suppressed: the only interpolation is the ROLE constant.
    explicit = _psql(
        pg,
        "SELECT count(*) FROM pg_namespace n, aclexplode(n.nspacl) a "  # noqa: S608
        f"WHERE n.nspname = 'public' AND a.grantee = '{ROLE}'::regrole::oid;",
        db="omnidash_analytics",
    )
    assert explicit == "0", (
        "104 issued an explicit schema grant on omnidash_analytics, which it "
        "cannot reach from the omnibase_infra connection"
    )
