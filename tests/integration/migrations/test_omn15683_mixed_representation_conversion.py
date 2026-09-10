# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execution proof for the OMN-15683 mixed-representation conversion class.

THE DEFECT, IN ONE LINE
    Migration 0034 resolves tenant identity with a single predicate --
    ``m.tenant_slug = d.tenant_id`` -- and has no branch for a ``tenant_id``
    that is ALREADY the canonical UUID. Write-time UUID stamping (OMN-16804) is
    live, so the column is now MIXED, and 0034 aborts on every stamped row.

WHY THAT IS WORSE THAN A MISSING BRANCH
    0034's exception says the tenant-registry projection "HAS NOT CAUGHT UP".
    Measured read-only on onex-dev 2026-09-08, that sentence was FALSE for all
    26 rows it named: all three tenants were present in
    ``tenant_registry_mirror``, under ``tenant_uuid``. 0034 was written to
    replace ``contains null values`` -- a message that described a symptom
    instead of a cause -- and then asserted a cause it had not established.

WHY A REAL DATABASE
    Every claim here is about what PostgreSQL does with a ``LEFT JOIN``, a
    ``NOT NULL`` column, ``FORCE ROW LEVEL SECURITY`` and an ``UPDATE ... FROM``
    -- none of which can be observed through a mock. The standing evidence for
    what happens when this class is reasoned about instead of measured is
    OMN-16493, where two fail-closed guards were green in review and RLS-blinded
    on the lane. Every migration below is applied as the REAL vendored bytes,
    through the SAME ``psql -v ON_ERROR_STOP=1 -f <file>`` invocation the
    deploy-time runner uses.

THE FIXTURE IS THE ONEX-DEV CENSUS, NOT AN INVENTION
    229 rows: ``beta-business-proof`` 151 (slug) and 21 (its canonical UUID),
    ``omninode`` 43, two more canonical-UUID values at 4 and 1, three tail slugs
    at 1 each, and the two SEED fixtures at 3 each. That is the enumeration
    recorded in the rolling work ledger for 2026-09-08T18:40Z, reconciled there
    against ``pg_stat_user_tables.n_live_tup``. The same tenant appearing under
    BOTH representations is the whole point: it is what makes 0034 abort and
    what 0036 has to collapse onto one identity.

SERVER SOURCE, and the OMN-16412 contamination guard
    An already-running server named by ``OMNIBASE_INFRA_DB_URL`` /
    ``POSTGRES_HOST``, else a hermetic ephemeral cluster from a local
    ``initdb``. A NON-LOOPBACK ambient host is a hard error unless
    ``OMN15683_ALLOW_REMOTE_PG=1``: this suite creates and drops throwaway
    databases AND cluster-wide roles, and a persistent dev-shell env var has
    silently pointed a sibling suite at the live .201 stability-test Postgres
    before (OMN-16412, 35 leftover throwaway databases). Skips cleanly when
    neither source exists -- unless ``OMN15683_REQUIRE_PG=1``, which turns every
    skip into a hard failure so a job that owns a Postgres cannot go vacuously
    green.

Run: uv run pytest tests/integration/migrations/test_omn15683_mixed_representation_conversion.py -v

Ticket: OMN-15683 (this class, and 0036), OMN-16930 / OMN-17288 / OMN-17316
(the chain), OMN-16804 (the stamping that produced the mixture).
"""

from __future__ import annotations

import ipaddress
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

REPO_ROOT = Path(__file__).resolve().parents[3]
_MIGRATIONS = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_delegation"
)
_PREDECESSOR = (
    _MIGRATIONS / "0034_delegation_events_uuid_via_registry_role_set_guard.sql"
)
# 0036: the FIRST successor. It converts correctly and cannot RUN on onex-dev,
# because it reads tenant_registry_mirror after switching role. It is a RED
# control here, on the privilege axis, exactly as 0034 is on the data axis.
_SUPERSEDED_BY_PRIVILEGE = (
    _MIGRATIONS / "0036_delegation_events_uuid_mixed_representation.sql"
)
_SUCCESSOR = (
    _MIGRATIONS
    / "0037_delegation_events_uuid_mixed_representation_guard_before_set_role.sql"
)
_READINESS = (
    REPO_ROOT / "scripts" / "ci" / "check_delegation_tenant_conversion_readiness.py"
)

# The canonical UUIDs that onex-dev's OMN-16804 stamping had already written
# into the still-text column. All three ARE registry tenants -- that is what
# makes 0034's "has not caught up" message wrong rather than merely unhelpful.
_STAMPED = {
    "beta-business-proof": ("91c74442-1233-4c97-b191-911a10346fdf", 151, 21),
    "second-stamped-tenant": ("f157eaa4-2b50-4df9-bdae-14f815815d66", 0, 4),
    "third-stamped-tenant": ("a64840b6-17c8-479c-823b-3cfef8e8a31e", 0, 1),
}
# Slug-only tenants, and the house tenant.
_SLUG_ONLY = {
    "omninode": ("820272f9-4aaf-5add-a2df-0af942852ab2", 43),
    "tail-slug-a": ("2f9e1c4a-1111-4a1a-9b2c-0d1e2f3a4b5c", 1),
    "tail-slug-b": ("3a8f2d5b-2222-4b2b-8c3d-1e2f3a4b5c6d", 1),
    "tail-slug-c": ("4b7e3c6c-3333-4c3c-9d4e-2f3a4b5c6d7e", 1),
}
# The pre-tenancy fixtures 0034/0036 delete by EXACT correlation_id. Their
# tenant values are NOT registry tenants and never will be.
_DEBRIS = {
    "SEED-A": "11111111-1111-1111-1111-111111111111",
    "SEED-B": "22222222-2222-2222-2222-222222222222",
}
_EXPECTED_TOTAL = 151 + 21 + 4 + 1 + 43 + 1 + 1 + 1 + 3 + 3  # 229
_EXPECTED_SURVIVING = _EXPECTED_TOTAL - 6  # 223

# 0034's misdirecting sentence, pinned verbatim. Its ARRIVAL is the defect.
_MISDIRECTION = "HAS NOT CAUGHT UP"
# The successors' distinctive wording. Matched on the phrase, not the ticket id: psql
# echoes the migration's absolute PATH on every diagnostic line, and a worktree
# named after the ticket would make an id match pass vacuously.
_SUCCESSOR_MARKER = "resolves under NEITHER form"


def _pg_bin(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for prefix in sorted(Path("/opt/homebrew/opt").glob("postgresql@*"), reverse=True):
        candidate = prefix / "bin" / name
        if candidate.exists():
            return str(candidate)
    return None


_PSQL = _pg_bin("psql")
_INITDB = _pg_bin("initdb")
_PG_CTL = _pg_bin("pg_ctl")

_REQUIRE_PG = os.environ.get("OMN15683_REQUIRE_PG") == "1"


def _unavailable(reason: str) -> None:
    if _REQUIRE_PG:
        raise AssertionError(f"OMN15683_REQUIRE_PG=1 but {reason}")
    pytest.skip(reason)


if _PSQL is None:  # pragma: no cover - environment dependent
    if _REQUIRE_PG:
        raise AssertionError("OMN15683_REQUIRE_PG=1 but psql is not available")
    pytest.skip("psql not available", allow_module_level=True)

PSQL: str = _PSQL


@dataclass(frozen=True)
class Server:
    host: str
    port: str
    user: str
    password: str

    def env(self) -> dict[str, str]:
        merged = dict(os.environ)
        merged["PGPASSWORD"] = self.password
        return merged


def _is_loopback_host(host: str) -> bool:
    if host.startswith("/"):
        return True
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _reject_unless_loopback_or_opted_in(env_var: str, host: str) -> None:
    if _is_loopback_host(host) or os.environ.get("OMN15683_ALLOW_REMOTE_PG") == "1":
        return
    raise AssertionError(
        f"{env_var}={host!r} resolves to a non-loopback host. Refusing to run "
        "the OMN-15683 conversion proof against it -- this suite creates and "
        "drops throwaway databases AND cluster-wide roles, and a stray ambient "
        "env var pointing at a shared/live Postgres is the OMN-16412 vector. "
        "Set OMN15683_ALLOW_REMOTE_PG=1 to opt in explicitly."
    )


def _server_from_env() -> Server | None:
    from urllib.parse import unquote, urlparse

    dsn = os.environ.get("OMNIBASE_INFRA_DB_URL", "")
    if dsn:
        parsed = urlparse(dsn)
        if parsed.hostname:
            _reject_unless_loopback_or_opted_in(
                "OMNIBASE_INFRA_DB_URL", parsed.hostname
            )
            return Server(
                host=parsed.hostname,
                port=str(parsed.port or 5432),
                user=unquote(parsed.username or "postgres"),
                password=unquote(parsed.password or ""),
            )
    host = os.environ.get("POSTGRES_HOST")
    if host:
        _reject_unless_loopback_or_opted_in("POSTGRES_HOST", host)
        return Server(
            host=host,
            port=os.environ.get("POSTGRES_PORT", "5432"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
        )
    return None


# PostgreSQL 16+: `GRANT ... WITH INHERIT ..., SET ...` and the 'SET' privilege
# for pg_has_role do not exist before 16, and 0034's carried-over ownership
# guard depends on both. Every .201 lane and the CI service are 16.x.
_MIN_SERVER_VERSION_NUM = 160000


def _require_pg16(srv: Server) -> None:
    probe = _psql(srv, "postgres", "-tAc", "SHOW server_version_num")
    assert probe.returncode == 0, f"could not read server_version_num: {probe.stderr!r}"
    if int(probe.stdout.strip()) < _MIN_SERVER_VERSION_NUM:
        _unavailable(
            f"server is PostgreSQL {probe.stdout.strip()} but the membership "
            "predicates these migrations carry were introduced in 16"
        )


def _psql(
    srv: Server, database: str, *args: str, user: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            PSQL,
            "-X",
            "-q",
            "-h",
            srv.host,
            "-p",
            srv.port,
            "-U",
            user or srv.user,
            "-d",
            database,
            *args,
        ],
        env=srv.env(),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    external = _server_from_env()
    if external is not None:
        if _psql(external, "postgres", "-tAc", "SELECT 1").returncode == 0:
            _require_pg16(external)
            yield external
            return

    if _INITDB is None or _PG_CTL is None:  # pragma: no cover
        _unavailable(
            "no reachable Postgres (OMNIBASE_INFRA_DB_URL/POSTGRES_HOST) and no "
            "local initdb to build an ephemeral cluster"
        )
        raise AssertionError("unreachable: _unavailable() skips or raises")
    initdb, pg_ctl = _INITDB, _PG_CTL

    root = Path(tempfile.mkdtemp(prefix="omn15683-pg-"))
    sock = root / "sock"
    sock.mkdir()
    data = root / "data"
    subprocess.run(
        [initdb, "-D", str(data), "-U", "postgres", "-A", "trust"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            pg_ctl,
            "-D",
            str(data),
            "-l",
            str(root / "postgres.log"),
            "-o",
            f"-k {sock} -h '' -c listen_addresses=''",
            "-w",
            "start",
        ],
        check=True,
        capture_output=True,
    )
    try:
        hermetic = Server(host=str(sock), port="5432", user="postgres", password="")
        _require_pg16(hermetic)
        yield hermetic
    finally:
        subprocess.run(
            [pg_ctl, "-D", str(data), "-m", "immediate", "stop"],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(root, ignore_errors=True)


def _scalar(srv: Server, database: str, sql: str, user: str | None = None) -> str:
    result = _psql(srv, database, "-tA", "-c", sql, user=user)
    assert result.returncode == 0, (
        f"psql did not run successfully (scalar {sql!r}): exit "
        f"{result.returncode}. This is a missing/failed prerequisite, NOT a "
        f"finding.\nstdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    return result.stdout.strip()


def _apply_as(
    srv: Server, database: str, migration: Path, user: str
) -> subprocess.CompletedProcess[str]:
    """Run a migration exactly the way the deploy-time runner does.

    ``psql -v ON_ERROR_STOP=1 -f <file>`` -- not a driver, not a wrapped
    transaction. The failure mode under test is about what the SERVER refuses
    and when, so the invocation has to be the real one.
    """
    return subprocess.run(
        [
            PSQL,
            "-X",
            "-q",
            "-h",
            srv.host,
            "-p",
            srv.port,
            "-U",
            user,
            "-d",
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-f",
            str(migration),
        ],
        env=srv.env(),
        capture_output=True,
        text=True,
        check=False,
    )


@dataclass(frozen=True)
class Lane:
    """A throwaway database carrying the onex-dev delegation_events shape."""

    database: str
    owner: str
    migrator: str
    mirror_owner: str
    mirror_reader_role: str


@pytest.fixture
def lane(server: Server) -> Iterator[Lane]:
    """The onex-dev shape: mixed text column, FORCE RLS, owner != migrator.

    AND THE GRANT TOPOLOGY, which the first revision of this fixture did not
    carry (OMN-15683, FRICTION 2026-09-08T22:25:10Z). It owned BOTH tables with
    one role, so the cross-owner mirror read that fails in production was free
    here and 0036 went green on every leg before aborting on onex-dev with
    ``permission denied for table tenant_registry_mirror``.

    Reproduced from the live onex-dev catalog readback:

        delegation_events        owner  role_omninode_owner
        tenant_registry_mirror   owner  role_omnidash
        tenant_registry_mirror   ACL    {role_omnidash=arwdDxt/role_omnidash,
                                         app_dashboard=r,
                                         omninode_runtime=arw,
                                         jake_ro=r}

    The migrate identity reaches the mirror the way it does there -- not by an
    ACL entry of its own (the live ACL has none) but by membership in a role
    that holds one. A reconstruction reproduces DATA for free; it reproduces
    OWNERSHIP, GRANTS and ROLE MEMBERSHIP only when it is told to.
    """
    tag = uuid.uuid4().hex[:12]
    database = f"omn15683_{tag}"
    owner = f"omn15683_owner_{tag}"
    migrator = f"omn15683_migrator_{tag}"
    mirror_owner = f"omn15683_mirror_owner_{tag}"
    mirror_reader = f"omn15683_mirror_reader_{tag}"

    assert (
        _psql(server, "postgres", "-c", f"CREATE DATABASE {database}").returncode == 0
    )
    if (
        _scalar(
            server, "postgres", "SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard'"
        )
        != "1"
    ):
        _psql(server, "postgres", "-c", "CREATE ROLE app_dashboard")
    _psql(server, "postgres", "-c", f"CREATE ROLE {owner}")
    _psql(server, "postgres", "-c", f"CREATE ROLE {mirror_owner}")
    _psql(server, "postgres", "-c", f"CREATE ROLE {mirror_reader}")
    _psql(server, "postgres", "-c", f"CREATE ROLE {migrator} LOGIN")
    # PostgreSQL 16 default membership confers both INHERIT and SET, which is
    # what the carried-over OMN-17316 guard requires.
    _psql(server, "postgres", "-c", f"GRANT {owner} TO {migrator}")
    # The migrate identity's ONLY route to the mirror: membership in a role that
    # holds SELECT. Revoking this one grant is what the privilege RED control
    # below does, and it is the only difference between a lane 0037 can convert
    # and a lane it refuses by name.
    _psql(server, "postgres", "-c", f"GRANT {mirror_reader} TO {migrator}")

    rows: list[str] = []
    mirror: list[str] = []
    for slug, (canonical, slug_rows, uuid_rows) in _STAMPED.items():
        mirror.append(f"('{slug}', '{canonical}'::uuid, 'active')")
        if slug_rows:
            rows.append(
                f"SELECT '{slug}-slug-' || g, '{slug}' FROM generate_series(1,{slug_rows}) g"  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
            )
        if uuid_rows:
            rows.append(
                f"SELECT '{slug}-uuid-' || g, '{canonical}' "  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
                f"FROM generate_series(1,{uuid_rows}) g"
            )
    for slug, (canonical, slug_rows) in _SLUG_ONLY.items():
        mirror.append(f"('{slug}', '{canonical}'::uuid, 'active')")
        rows.append(
            f"SELECT '{slug}-slug-' || g, '{slug}' FROM generate_series(1,{slug_rows}) g"  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
        )
    for prefix, literal in _DEBRIS.items():
        rows.append(f"SELECT '{prefix}-' || g, '{literal}' FROM generate_series(1,3) g")  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text

    setup = f"""
    CREATE TABLE tenant_registry_mirror (
        tenant_slug text PRIMARY KEY,
        tenant_uuid uuid NOT NULL UNIQUE,
        status      text NOT NULL,
        observed_at timestamptz NOT NULL DEFAULT now()
    );
    INSERT INTO tenant_registry_mirror (tenant_slug, tenant_uuid, status)
    VALUES {", ".join(mirror)};

    -- The migration 0022 shape: tenant_id TEXT NOT NULL DEFAULT 'omninode'.
    CREATE TABLE delegation_events (
        id             uuid NOT NULL DEFAULT gen_random_uuid(),
        correlation_id text NOT NULL,
        "timestamp"    timestamptz NOT NULL DEFAULT now(),
        tenant_id      text NOT NULL DEFAULT 'omninode'::text
    );
    INSERT INTO delegation_events (correlation_id, tenant_id)
    {" UNION ALL ".join(rows)};

    ALTER TABLE delegation_events OWNER TO {owner};
    -- THE TOPOLOGY. A DIFFERENT owner, and an ACL that does NOT name
    -- delegation_events' owner -- byte-for-byte the shape read back from
    -- onex-dev. `{owner}` therefore cannot read the mirror, which is the whole
    -- defect 0037 exists to route around.
    ALTER TABLE tenant_registry_mirror OWNER TO {mirror_owner};
    GRANT SELECT ON tenant_registry_mirror TO {mirror_reader};
    GRANT SELECT ON tenant_registry_mirror TO app_dashboard;
    GRANT SELECT ON delegation_events TO app_dashboard;

    -- Migration 0023's policy: TEXT compared to TEXT.
    CREATE POLICY tenant_isolation ON delegation_events
      FOR ALL
      USING (tenant_id = current_setting('app.tenant_id', true))
      WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
    ALTER TABLE delegation_events ENABLE ROW LEVEL SECURITY;
    ALTER TABLE delegation_events FORCE ROW LEVEL SECURITY;
    ANALYZE delegation_events;
    """  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
    prepared = _psql(server, database, "-v", "ON_ERROR_STOP=1", "-c", setup)
    assert prepared.returncode == 0, prepared.stderr

    try:
        yield Lane(
            database=database,
            owner=owner,
            migrator=migrator,
            mirror_owner=mirror_owner,
            mirror_reader_role=mirror_reader,
        )
    finally:
        _psql(server, "postgres", "-c", f"DROP DATABASE IF EXISTS {database}")
        _psql(server, "postgres", "-c", f"DROP ROLE IF EXISTS {migrator}")
        _psql(server, "postgres", "-c", f"DROP ROLE IF EXISTS {mirror_reader}")
        _psql(server, "postgres", "-c", f"DROP ROLE IF EXISTS {mirror_owner}")
        _psql(server, "postgres", "-c", f"DROP ROLE IF EXISTS {owner}")


def _column_type(server: Server, lane: Lane) -> str:
    return _scalar(
        server,
        lane.database,
        "SELECT atttypid::regtype::text FROM pg_attribute "
        "WHERE attrelid = 'delegation_events'::regclass "
        "AND attname = 'tenant_id' AND NOT attisdropped",
    )


def test_fixture_reproduces_the_onex_dev_census(server: Server, lane: Lane) -> None:
    """The positive control. Without it every assertion below could be vacuous.

    A fixture that failed to seed would make the predecessor "not abort" and the
    successor "convert cleanly" for the same uninteresting reason.
    """
    assert _scalar(server, lane.database, "SELECT count(*) FROM delegation_events") == (
        str(_EXPECTED_TOTAL)
    )
    assert _column_type(server, lane) == "text"
    # The mixture itself: at least one tenant present under BOTH forms.
    both = _scalar(
        server,
        lane.database,
        "SELECT count(*) FROM tenant_registry_mirror m "
        "WHERE EXISTS (SELECT 1 FROM delegation_events d WHERE d.tenant_id = m.tenant_slug) "
        "AND EXISTS (SELECT 1 FROM delegation_events d WHERE d.tenant_id = m.tenant_uuid::text)",
    )
    assert int(both) >= 1, (
        "the fixture does not carry a tenant under both representations, so it "
        "does not reproduce the class under test"
    )


def test_fixture_reproduces_the_onex_dev_grant_topology(
    server: Server, lane: Lane
) -> None:
    """The positive control ON THE PRIVILEGE AXIS.

    Without this the two privilege tests below could both pass for the
    uninteresting reason that the fixture never split the ownership -- which is
    exactly what the FIRST revision of this fixture did, and exactly why 0036
    reached onex-dev. A reconstruction that reproduces the row census and not
    the grant topology is silent on the axis that broke.
    """
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT pg_get_userbyid(relowner) FROM pg_class "
            "WHERE oid = 'delegation_events'::regclass",
        )
        == lane.owner
    )
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT pg_get_userbyid(relowner) FROM pg_class "
            "WHERE oid = 'tenant_registry_mirror'::regclass",
        )
        == lane.mirror_owner
    ), "the two relations share an owner; the cross-owner read is not reproduced"
    assert (
        _scalar(
            server,
            lane.database,
            f"SELECT has_table_privilege('{lane.owner}', "
            "'tenant_registry_mirror', 'SELECT')",
        )
        == "f"
    ), (
        "delegation_events' owner CAN read the mirror in this fixture, so the "
        "0036 failure cannot be reproduced here"
    )
    assert (
        _scalar(
            server,
            lane.database,
            f"SELECT has_table_privilege('{lane.migrator}', "
            "'tenant_registry_mirror', 'SELECT')",
        )
        == "t"
    ), "the migrate identity cannot read the mirror, so 0037 has nothing to use"


def test_red_control_0036_aborts_on_the_cross_owner_mirror_read(
    server: Server, lane: Lane
) -> None:
    """0036 reproduces the onex-dev abort, verbatim, and applies nothing.

    This is the leg the 0036 lab proof did not have. Its failure message is
    pinned on the SERVER's own wording rather than on anything this repo
    writes, because the point is that PostgreSQL refuses the read -- not that a
    guard noticed.
    """
    result = _apply_as(server, lane.database, _SUPERSEDED_BY_PRIVILEGE, lane.migrator)
    assert result.returncode != 0, (
        "0036 CONVERTED against a split-ownership topology. Either the fixture "
        "stopped reproducing the ACL split or the privilege axis was closed "
        "elsewhere -- either way this RED control measures nothing."
    )
    assert "permission denied for table tenant_registry_mirror" in result.stderr, (
        "0036 aborted for some OTHER reason; this control is no longer pinned "
        f"to the measured onex-dev failure. stderr={result.stderr!r}"
    )
    # It gets as far as the determinism guard -- past the blindness
    # reconciliation and the debris DELETE, exactly as on onex-dev. That
    # ordering is why the run looked healthy right up to the abort.
    assert "visibility reconciled" in result.stderr
    assert "debris row(s)" in result.stderr
    # And the whole transaction went with it.
    assert _column_type(server, lane) == "text"
    assert _scalar(
        server, lane.database, "SELECT count(*) FROM delegation_events"
    ) == str(_EXPECTED_TOTAL), (
        "0036's debris DELETE was not rolled back with its aborting block"
    )


def test_successor_refuses_by_name_when_the_migrate_identity_cannot_read_the_mirror(
    server: Server, lane: Lane
) -> None:
    """0037 does not grant itself the privilege; it refuses, early, by name.

    The alternative repair was a GRANT to delegation_events' owner as the
    migration's first statement. It is rejected in the file's header on
    evidence, and this test is what keeps the rejection honest: the refusal
    happens BEFORE the role switch and before any mutation, and it names the
    remedy as an operator act rather than performing it.
    """
    assert (
        _psql(
            server,
            "postgres",
            "-c",
            f"REVOKE {lane.mirror_reader_role} FROM {lane.migrator}",
        ).returncode
        == 0
    )
    result = _apply_as(server, lane.database, _SUCCESSOR, lane.migrator)
    assert result.returncode != 0, "0037 converted without being able to resolve"
    assert "holds no SELECT on tenant_registry_mirror" in result.stderr, (
        f"0037 aborted opaquely instead of refusing by name: {result.stderr!r}"
    )
    assert lane.migrator in result.stderr, "the refusal does not name the role"
    assert "GRANT SELECT ON tenant_registry_mirror TO" in result.stderr, (
        "the refusal does not state the remedy"
    )
    # Nothing was mutated: the refusal precedes even the NO FORCE.
    assert "visibility reconciled" not in result.stderr, (
        "0037 got past Phase A before refusing; the guard is in the wrong place"
    )
    assert _column_type(server, lane) == "text"
    assert _scalar(
        server, lane.database, "SELECT count(*) FROM delegation_events"
    ) == str(_EXPECTED_TOTAL)


def test_successor_creates_no_relation_and_leaves_no_privilege(
    server: Server, lane: Lane
) -> None:
    """The snapshot is a PL/pgSQL variable: nothing created, nothing granted.

    That is the reason the repair is a snapshot and not a GRANT. A GRANT would
    be a persistent, cross-owner widening of a role's reach, made to get one
    transaction through -- and the first revision of 0037 used a temp table,
    which the OMN-15361 application-database domain gate rejected, because a
    temp relation is authority the topology cannot account for. A variable is
    neither a relation nor a privilege object, so there is nothing for either
    concern to attach to; this test is what keeps that true.
    """
    assert _apply_as(server, lane.database, _SUCCESSOR, lane.migrator).returncode == 0
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT count(*) FROM pg_class "
            "WHERE relname LIKE 'omn15683%' OR relname LIKE '%mirror_snapshot%'",
        )
        == "0"
    ), "0037 left a relation behind"
    # And the mirror's ACL is untouched -- no privilege survives the file.
    assert (
        _scalar(
            server,
            lane.database,
            f"SELECT has_table_privilege('{lane.owner}', "
            "'tenant_registry_mirror', 'SELECT')",
        )
        == "f"
    ), "0037 left delegation_events' owner holding SELECT on the mirror"


def test_red_control_predecessor_aborts_on_the_mixed_column(
    server: Server, lane: Lane
) -> None:
    """0034 refuses -- and blames the projection for data the projection has.

    TWO axes, and the ORDER matters. On the true onex-dev topology 0034 does
    not reach its data-axis guard at all: it reads tenant_registry_mirror after
    switching role, so it dies on `permission denied` first, exactly as 0036
    did. That is asserted first, and then the mirror is granted to the owner to
    restore the SINGLE-OWNER shape in which the data defect is observable --
    which is precisely the shape the pre-0037 harness had, and the reason the
    privilege defect survived it. Naming it here keeps it a deliberate control
    rather than a fixture that quietly reverted.
    """
    privilege_first = _apply_as(server, lane.database, _PREDECESSOR, lane.migrator)
    assert privilege_first.returncode != 0
    assert (
        "permission denied for table tenant_registry_mirror" in privilege_first.stderr
    ), (
        "0034 reached its data-axis guard on a split-ownership topology; the "
        "fixture is no longer reproducing the onex-dev ACL split"
    )

    assert (
        _psql(
            server,
            lane.database,
            "-c",
            f"GRANT SELECT ON tenant_registry_mirror TO {lane.owner}",
        ).returncode
        == 0
    )
    result = _apply_as(server, lane.database, _PREDECESSOR, lane.migrator)
    assert result.returncode != 0, (
        "0034 CONVERTED a mixed-representation column. That would mean the "
        "defect this file exists to prove has been fixed elsewhere, or the "
        "fixture no longer reproduces it -- either way the RED control below "
        "is no longer measuring anything."
    )
    for canonical, _, _ in _STAMPED.values():
        assert canonical in result.stderr, (
            f"0034 aborted without naming {canonical}, one of the "
            "already-canonical values it cannot resolve"
        )
    assert _MISDIRECTION in result.stderr, (
        "0034's message no longer carries the misdirection this supersession "
        "is partly about; the RED control's second half is stale"
    )
    assert _column_type(server, lane) == "text", "0034 partially converted"
    assert _scalar(
        server, lane.database, "SELECT count(*) FROM delegation_events"
    ) == str(_EXPECTED_TOTAL), (
        "0034's debris DELETE was not rolled back with its aborting block"
    )


def test_successor_converts_the_mixed_column(server: Server, lane: Lane) -> None:
    """GREEN. Both representations collapse onto one canonical identity."""
    result = _apply_as(server, lane.database, _SUCCESSOR, lane.migrator)
    assert result.returncode == 0, f"0037 failed on the mixed column: {result.stderr!r}"
    assert _column_type(server, lane) == "uuid"
    assert _scalar(
        server, lane.database, "SELECT count(*) FROM delegation_events"
    ) == str(_EXPECTED_SURVIVING)
    # The six pre-tenancy fixtures are gone, by exact correlation_id.
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT count(*) FROM delegation_events WHERE correlation_id LIKE 'SEED-%'",
        )
        == "0"
    )
    # Every row resolves to a registry tenant. Nothing invented, nothing dropped.
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT count(*) FROM delegation_events d WHERE NOT EXISTS ("
            "SELECT 1 FROM tenant_registry_mirror m WHERE m.tenant_uuid = d.tenant_id)",
        )
        == "0"
    )


def test_already_canonical_rows_pass_through_unchanged(
    server: Server, lane: Lane
) -> None:
    """A row that already held the canonical UUID equals ITSELF afterwards.

    By resolution against the registry, not by an unchecked bypass -- which is
    why it is asserted per row rather than inferred from the column type.
    """
    assert _apply_as(server, lane.database, _SUCCESSOR, lane.migrator).returncode == 0
    for slug, (canonical, _, uuid_rows) in _STAMPED.items():
        if not uuid_rows:
            continue
        mismatched = _scalar(
            server,
            lane.database,
            "SELECT count(*) FROM delegation_events "  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
            f"WHERE correlation_id LIKE '{slug}-uuid-%' "
            f"AND tenant_id <> '{canonical}'::uuid",
        )
        assert mismatched == "0", (
            f"{mismatched} row(s) that already held {canonical} no longer do"
        )


def test_slug_rows_resolve_to_their_mirror_uuid(server: Server, lane: Lane) -> None:
    """And a slug row equals the mirror's uuid for that slug -- the SAME uuid a
    stamped row of the same tenant already carried."""
    assert _apply_as(server, lane.database, _SUCCESSOR, lane.migrator).returncode == 0
    for slug, (canonical, slug_rows, _) in _STAMPED.items():
        if not slug_rows:
            continue
        assert (
            _scalar(
                server,
                lane.database,
                "SELECT count(*) FROM delegation_events "  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
                f"WHERE correlation_id LIKE '{slug}-slug-%' "
                f"AND tenant_id <> '{canonical}'::uuid",
            )
            == "0"
        )
        # The collapse: slug rows and stamped rows now share one identity.
        assert _scalar(
            server,
            lane.database,
            "SELECT count(*) FROM delegation_events "  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
            f"WHERE tenant_id = '{canonical}'::uuid",
        ) == str(slug_rows + _STAMPED[slug][2])
    for slug, (canonical, slug_rows) in _SLUG_ONLY.items():
        assert (
            _scalar(
                server,
                lane.database,
                "SELECT count(*) FROM delegation_events "  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
                f"WHERE correlation_id LIKE '{slug}-slug-%' "
                f"AND tenant_id <> '{canonical}'::uuid",
            )
            == "0"
        )


def test_policy_and_force_rls_are_restored_as_uuid(server: Server, lane: Lane) -> None:
    """FORCE RLS back on, and the policy restated comparing uuid to uuid.

    Asserted because the type change requires DROPping the policy, and the
    OMN-17288 class is a table that commits with RLS on and no policy at all.
    """
    assert _apply_as(server, lane.database, _SUCCESSOR, lane.migrator).returncode == 0
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT relrowsecurity::text || ',' || relforcerowsecurity::text "
            "FROM pg_class WHERE oid = 'delegation_events'::regclass",
        )
        == "true,true"
    )
    policy = _scalar(
        server,
        lane.database,
        "SELECT pg_get_expr(polqual, polrelid) FROM pg_policy "
        "WHERE polrelid = 'delegation_events'::regclass AND polname = 'tenant_isolation'",
    )
    assert "::uuid" in policy, (
        f"tenant_isolation still compares against text after conversion: {policy!r}"
    )
    # The scratch resolution column never outlives the transaction.
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT count(*) FROM pg_attribute "
            "WHERE attrelid = 'delegation_events'::regclass "
            "AND attname = 'omn16930_resolved_tenant_uuid' AND NOT attisdropped",
        )
        == "0"
    )


def test_a_value_in_neither_form_still_fails_closed(server: Server, lane: Lane) -> None:
    """The NEGATIVE control. Resolving on two forms did not loosen the guard.

    The exception must name the value, its row count, and WHICH lookup failed --
    and must NOT assert that the projection is behind, which is the claim 0034
    made about data that was present.
    """
    foreign = "7c0ffee0-dead-4bee-9f00-000000000001"
    assert (
        _psql(
            server,
            lane.database,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "INSERT INTO delegation_events (correlation_id, tenant_id) VALUES "  # noqa: S608 - every interpolated value is a module constant or passes through _quote_literal; no caller input reaches the SQL text
            f"('foreign-1', '{foreign}'), ('foreign-2', '{foreign}')",
        ).returncode
        == 0
    )
    result = _apply_as(server, lane.database, _SUCCESSOR, lane.migrator)
    assert result.returncode != 0, "0037 converted a value it cannot resolve"
    assert foreign in result.stderr, "the refusal does not name the value"
    assert "2 row(s)" in result.stderr, "the refusal does not name the row count"
    assert _SUCCESSOR_MARKER in result.stderr
    assert "no tenant_registry_mirror row has tenant_uuid = this value" in (
        result.stderr
    ), "the refusal does not say WHICH of the two lookups failed"
    assert _MISDIRECTION not in result.stderr, (
        "0037 reproduced 0034's misdirection -- it asserts the projection is "
        "behind for a value it has not established anything about"
    )
    assert _column_type(server, lane) == "text", "0036 partially converted"
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT count(*) FROM delegation_events WHERE correlation_id LIKE 'SEED-%'",
        )
        == "6"
    ), "the debris DELETE was not rolled back with the aborting block"


def test_readiness_script_reconciles_under_force_rls(
    server: Server, lane: Lane
) -> None:
    """The operator-safe check passes where 0034's own header recipe inverts.

    Two measurements in one test, deliberately: the naive query that 0034's
    header prescribes returns the string that means PASS against a table it
    cannot see, and the script REFUSES to report PASS on the same connection
    unless its enumeration reconciles.
    """
    naive = _scalar(
        server,
        lane.database,
        "SELECT coalesce(string_agg(DISTINCT quote_literal(d.tenant_id), ', '), "
        "'ALL RESOLVE') FROM delegation_events d "
        "LEFT JOIN tenant_registry_mirror m ON m.tenant_slug = d.tenant_id "
        "WHERE m.tenant_slug IS NULL",
        user=lane.migrator,
    )
    visible = _scalar(
        server,
        lane.database,
        "SELECT count(*) FROM delegation_events",
        user=lane.migrator,
    )
    assert naive == "ALL RESOLVE" and visible == "0", (
        "the FORCE-RLS blindness this script exists to defeat is not "
        "reproduced by the fixture, so the assertion below proves nothing"
    )

    dsn_host = lane.database
    completed = subprocess.run(
        [
            "python3",
            str(_READINESS),
            "--database",
            dsn_host,
            "--psql-exec",
            _psql_exec_json(server, lane.migrator),
        ],
        env=server.env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        "the readiness script did not report PASS on a lane whose every value "
        f"resolves:\n{completed.stdout}\n{completed.stderr}"
    )
    assert "VERDICT: PASS" in completed.stdout
    assert f"enumerated total {_EXPECTED_TOTAL} == n_live_tup" in completed.stdout, (
        "PASS was printed without the reconciliation line that earns it"
    )


def _psql_exec_json(server: Server, user: str) -> str:
    import json

    return json.dumps([PSQL, "-h", server.host, "-p", server.port, "-U", user])
