# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration 103 executed against a CREATEROLE-less role (OMN-17301).

These tests run the REAL migration file against a REAL PostgreSQL, as a role
shaped the way the managed lane's migration identity is shaped: ``LOGIN
NOSUPERUSER NOCREATEDB NOCREATEROLE``. A static assertion about the file's text
could not have caught any of the three defects this module pins, because all
three are about what PostgreSQL *does* with the statements rather than what the
statements say:

* **D1** ``CREATE ROLE`` aborts the file with a raw ``permission denied to
  create role`` (SQLSTATE 42501). This is the defect that stalled
  ``Deploy onex-staging`` run 33341217605 at migration-order 1 of 6, before
  overlay-apply and the runtime digest pin, blocking every staging deploy on
  every trigger.
* **D2** ``GRANT CONNECT`` issued by a role that holds no grant option on the
  target database does not raise -- PostgreSQL emits
  ``WARNING: no privileges were granted`` and returns success. On the managed
  lane ``omnidash_analytics`` is owned by ``role_omnidash``, not by the flat
  loop's ``role_omnibase_infra``, so the grant silently did nothing.
* **D3** ``has_database_privilege(..., 'CONNECT')`` is TRUE via PUBLIC's default
  grant on any database that has not revoked it, so the file's own assertion
  could not detect D2. That is why D2 survived review.

The fixture reproduces the managed lane's OWNERSHIP SPLIT, not just its
privileges: ``omnidash_analytics`` is owned by ``role_omnidash`` while the
migration runs as ``role_omnibase_infra``. Without that split D2 and D3 are
invisible -- on a scratch database owned by the executing role the GRANT
succeeds and everything looks correct.

Ticket: OMN-17301. Class: OMN-16759 / OMN-16249 (database-level half),
OMN-15343 (the provisioning seam).
"""

from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "103_create_tenant_projection_writer_role.sql"
)
ROLE = "tenant_projection_writer"
IMAGE = "postgres:16-alpine"
# Path INSIDE the ephemeral container, not on the host filesystem: the
# container is created and destroyed by the fixture and is not shared, so
# the usual predictable-temp-path attack has no reachable surface here.
CONTAINER_SQL = "/tmp/103.sql"  # noqa: S108 - path inside the ephemeral container

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.postgres]


def _docker() -> str:
    binary = shutil.which("docker")
    if binary is None:
        pytest.skip("docker is not available; this module executes real SQL")
    return binary


@pytest.fixture(scope="module")
def pg() -> Iterator[str]:
    """A scratch PostgreSQL shaped like the managed lane. Yields the container."""
    docker = _docker()
    name = f"omn17301-{uuid.uuid4().hex[:10]}"
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
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:  # pragma: no cover
        pytest.skip(f"could not start {IMAGE}: {exc}")

    try:
        for _ in range(60):
            probe = subprocess.run(
                [docker, "exec", name, "pg_isready", "-U", "postgres"],
                capture_output=True,
                timeout=30,
                check=False,  # not-ready is the expected loop condition, not an error
            )
            if probe.returncode == 0:
                break
            time.sleep(1)
        else:  # pragma: no cover
            pytest.skip("scratch postgres never became ready")

        # The managed lane's shape: two NOCREATEROLE migration identities, and
        # omnidash_analytics owned by role_omnidash rather than by the flat
        # loop's role. The ownership split is what makes D2/D3 observable.
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


def _apply_migration(container: str, *, user: str) -> subprocess.CompletedProcess[str]:
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
        check=False,  # the failing path is asserted by the caller, not raised here
    )


def _drop_role(container: str) -> None:
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
                "-c",
                f"DROP OWNED BY {ROLE} CASCADE;",
                "-d",
                db,
            ],
            capture_output=True,
            timeout=60,
            check=False,  # the role may not exist or may own nothing; both are fine
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
        check=False,  # IF EXISTS makes absence a success, not an error
    )


def _role_exists(container: str) -> bool:
    # S608 below is suppressed because ROLE is a module-level constant here,
    # never external input.
    query = f"SELECT count(*) FROM pg_roles WHERE rolname = '{ROLE}'"  # noqa: S608
    return _psql(container, query) == "1"


def test_absent_role_and_no_createrole_fails_with_a_named_remediation(pg: str) -> None:
    """D1: the managed-lane condition must name the privilege and the seam.

    Before OMN-17301 this produced, verbatim and with nothing else::

        ERROR:  permission denied to create role
        DETAIL:  Only roles with the CREATEROLE attribute may create roles.

    which tells an operator neither which principal is missing nor where it is
    provisioned. The migration must still FAIL -- reporting success over an
    absent principal is the OMN-14950 masking outcome -- but it must fail
    legibly.
    """
    _drop_role(pg)
    assert not _role_exists(pg), "precondition: the role must be absent"

    result = _apply_migration(pg, user="role_omnibase_infra")

    assert result.returncode != 0, (
        "the migration reported SUCCESS while tenant_projection_writer does not "
        "exist. topology/application_database.py binds tenant_projection to this "
        "principal and OMN-16911 attests current_user on every projection "
        "connection, so a silent skip resurfaces as total DLQ loss on the tenant "
        "projections. Failing here is correct; masking is not."
    )
    combined = result.stdout + result.stderr
    assert "permission denied to create role" not in combined, (
        "the raw PostgreSQL privilege error reached the operator unhandled -- the "
        f"insufficient_privilege handler did not fire. Output:\n{combined}"
    )
    assert ROLE in combined
    assert "CREATEROLE" in combined
    assert "provision-cluster-roles.sh" in combined, (
        "the failure must name the provisioning seam that can actually create "
        f"the role. Output:\n{combined}"
    )
    assert "OMN-17301" in combined
    assert not _role_exists(pg), "a failed apply must not leave the role behind"


def test_role_provisioned_at_the_seam_makes_the_migration_a_no_op(pg: str) -> None:
    """AC2/AC3: once the seam has provisioned the role, the file is a clean no-op."""
    _drop_role(pg)
    _psql(
        pg,
        f"CREATE ROLE {ROLE} WITH NOLOGIN NOSUPERUSER NOBYPASSRLS "
        "NOCREATEDB NOCREATEROLE NOREPLICATION",
    )

    result = _apply_migration(pg, user="role_omnibase_infra")

    assert result.returncode == 0, (
        "with the role already provisioned the migration has no privileged work "
        f"left to do and must succeed. Output:\n{result.stdout}{result.stderr}"
    )
    assert _role_exists(pg)


def test_the_no_op_path_is_idempotent(pg: str) -> None:
    """A recorded migration is re-run by the compose runner on every bring-up."""
    _drop_role(pg)
    _psql(
        pg,
        f"CREATE ROLE {ROLE} WITH NOLOGIN NOSUPERUSER NOBYPASSRLS "
        "NOCREATEDB NOCREATEROLE NOREPLICATION",
    )

    first = _apply_migration(pg, user="role_omnibase_infra")
    second = _apply_migration(pg, user="role_omnibase_infra")

    assert first.returncode == 0
    assert second.returncode == 0, (
        f"second apply failed:\n{second.stdout}{second.stderr}"
    )


def test_connect_readback_reports_public_rather_than_claiming_a_grant(pg: str) -> None:
    """D2 + D3: the ineffective GRANT must be reported, not silently passed over.

    ``role_omnibase_infra`` holds no grant option on ``omnidash_analytics``
    (``role_omnidash`` owns it), so the GRANT is a no-op that PostgreSQL reports
    only as a WARNING. The pre-OMN-17301 assertion could not see that, because
    ``has_database_privilege`` is satisfied by PUBLIC's default CONNECT. The
    migration must now say WHICH grant carries the privilege.
    """
    _drop_role(pg)
    _psql(
        pg,
        f"CREATE ROLE {ROLE} WITH NOLOGIN NOSUPERUSER NOBYPASSRLS "
        "NOCREATEDB NOCREATEROLE NOREPLICATION",
    )

    result = _apply_migration(pg, user="role_omnibase_infra")
    combined = result.stdout + result.stderr

    assert result.returncode == 0
    assert "via PUBLIC's default grant" in combined, (
        "the migration must state that CONNECT is carried by PUBLIC and that no "
        "explicit grant landed, so 'granted' is never inferred from 'can "
        f"connect'. Output:\n{combined}"
    )

    explicit = _psql(
        pg,
        # S608 below is suppressed: the only interpolation is the ROLE constant.
        "SELECT EXISTS (SELECT 1 FROM pg_database d, aclexplode(d.datacl) a "  # noqa: S608
        "WHERE d.datname = 'omnidash_analytics' "
        f"AND a.grantee = '{ROLE}'::regrole::oid AND a.privilege_type = 'CONNECT')",
    )
    assert explicit == "f", (
        "this test's whole premise is that the explicit grant does NOT land for "
        "a non-owner grantor; if it now does, the fixture no longer reproduces "
        "the managed lane's ownership split and D2 is untested"
    )


def test_superuser_lane_still_creates_the_role(pg: str) -> None:
    """The compose lanes run as postgres and must keep working unchanged."""
    _drop_role(pg)
    assert not _role_exists(pg)

    result = _apply_migration(pg, user="postgres")

    assert result.returncode == 0, f"{result.stdout}{result.stderr}"
    assert _role_exists(pg)
    attributes = _psql(
        pg,
        "SELECT rolsuper::text || ',' || rolbypassrls::text || ',' || "  # noqa: S608 - ROLE is a module constant
        f"rolcreaterole::text || ',' || rolcreatedb::text FROM pg_roles WHERE rolname = '{ROLE}'",
    )
    assert attributes == "false,false,false,false", (
        "the RLS-relevant attributes must be pinned off: this principal is the "
        "identity the OMN-14894 tenant_isolation policies are enforced against, "
        f"got {attributes}"
    )
