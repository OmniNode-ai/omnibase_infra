# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Execution proof for the OMN-17316 role-membership guard class.

THE DEFECT, IN ONE LINE
    Migration 0033 guarded a ``SET ROLE`` with ``pg_has_role(..., 'USAGE')``.
    Since PostgreSQL 16, ``INHERIT`` and ``SET`` are INDEPENDENT membership
    options, so a membership created ``WITH INHERIT TRUE, SET FALSE`` passes
    that guard and the migration then aborts two statements later on a bare
    ``permission denied to set role "<owner>"``.

WHY THAT IS WORSE THAN A MISSING CHECK
    The guard exists precisely to convert an opaque late refusal into a named
    early one -- 0033's own comment says so ("This says the same thing earlier,
    and names the RLS-blindness consequence that made OMN-16493 cost a week").
    Under a ``SET FALSE`` membership it did the exact opposite of what it
    documents: it passed, and Postgres produced the opaque error anyway.

WHY A REAL DATABASE
    ``pg_has_role``'s three-valued behaviour under split INHERIT/SET membership
    is a property of the PostgreSQL 16 catalog, not of any code in this repo.
    It cannot be observed through a mock, and the standing evidence for what
    happens when this class is reasoned about instead of measured is OMN-16493,
    where two fail-closed guards were green in review and RLS-blinded on the
    lane. Every claim below is measured against the REAL vendored bytes through
    the SAME ``psql -v ON_ERROR_STOP=1 -f <file>`` invocation the deploy-time
    runner uses.

SERVER SOURCE, and the OMN-16412 contamination guard
    An already-running server named by ``OMNIBASE_INFRA_DB_URL`` /
    ``POSTGRES_HOST`` (this is how the CI ``migration-integration`` job supplies
    one), else a hermetic ephemeral cluster from a local ``initdb``. A
    NON-LOOPBACK ambient host is a hard error unless
    ``OMN17316_ALLOW_REMOTE_PG=1`` is also set: this suite creates and drops
    throwaway databases AND cluster-wide roles on whatever server it is handed,
    and a persistent dev-shell env var has silently pointed a sibling suite at
    the live .201 stability-test Postgres before (OMN-16412, 35 leftover
    throwaway databases). Skips cleanly when neither source exists -- unless
    ``OMN17316_REQUIRE_PG=1``, which turns every skip into a hard failure so a
    job that owns a Postgres cannot go vacuously green.

Run: uv run pytest tests/integration/migrations/test_omn17316_role_set_membership_guard.py -v

Ticket: OMN-17316 (this class), OMN-17288 (0033, the file under repair)
"""

from __future__ import annotations

import ipaddress
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

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
_DEFECTIVE = (
    _MIGRATIONS / "0033_delegation_events_uuid_via_registry_single_transaction.sql"
)
_REPAIRED = _MIGRATIONS / "0034_delegation_events_uuid_via_registry_role_set_guard.sql"

# The opaque refusal 0033 produces under a SET FALSE membership. Pinned to
# PostgreSQL's own wording: this string arriving instead of a named OMN-
# exception IS the defect.
_OPAQUE = "permission denied to set role"

# The repaired guard's distinctive wording. Used instead of the bare ticket id
# because psql echoes the migration's absolute PATH on every diagnostic line,
# and this ticket's worktree is named after the ticket -- matching on the id
# would match the path and pass vacuously.
_REPAIRED_MARKER = "pg_has_role SET is false"


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

_REQUIRE_PG = os.environ.get("OMN17316_REQUIRE_PG") == "1"


def _unavailable(reason: str) -> None:
    if _REQUIRE_PG:
        raise AssertionError(f"OMN17316_REQUIRE_PG=1 but {reason}")
    pytest.skip(reason)


if _PSQL is None:  # pragma: no cover - environment dependent
    if _REQUIRE_PG:
        raise AssertionError("OMN17316_REQUIRE_PG=1 but psql is not available")
    pytest.skip("psql not available", allow_module_level=True)

# Past the module-level skip above, psql is known to exist. Rebound to a
# non-Optional name so every subprocess argv below type-checks without a cast
# at each call site.
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


def _allow_remote_pg() -> bool:
    return os.environ.get("OMN17316_ALLOW_REMOTE_PG") == "1"


def _is_loopback_host(host: str) -> bool:
    """True only for hosts that never leave this machine."""
    if host.startswith("/"):  # unix socket path
        return True
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a real DNS name (container name, LAN host, ...)


def _reject_unless_loopback_or_opted_in(env_var: str, host: str) -> None:
    if _is_loopback_host(host) or _allow_remote_pg():
        return
    raise AssertionError(
        f"{env_var}={host!r} resolves to a non-loopback host. Refusing to run "
        "the OMN-17316 role-guard proof against it -- this suite creates and "
        "drops throwaway databases AND cluster-wide roles, and a stray ambient "
        "env var pointing at a shared/live Postgres is the OMN-16412 vector. "
        "Set OMN17316_ALLOW_REMOTE_PG=1 to opt in explicitly."
    )


def _server_from_env() -> Server | None:
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


# The whole class is PostgreSQL 16+. Before 16 there is no
# `GRANT ... WITH INHERIT ..., SET ...` syntax and no 'SET' privilege for
# pg_has_role, so an older server cannot express the defect OR the repair --
# it would fail with `syntax error at or near "INHERIT"` and read as a broken
# test rather than an inapplicable one. Found by running this suite against
# the hermetic fallback cluster on a dev box, which is Homebrew's PostgreSQL
# 14. Every .201 lane and the CI service are 16.x (verified 2026-08-31: dev
# and stability-test 16.15, prod and judge 16.14, CI postgres:16-alpine).
_MIN_SERVER_VERSION_NUM = 160000


def _require_pg16(srv: Server) -> None:
    probe = subprocess.run(
        [
            PSQL,
            "-X",
            "-h",
            srv.host,
            "-p",
            srv.port,
            "-U",
            srv.user,
            "-d",
            "postgres",
            "-tAc",
            "SHOW server_version_num",
        ],
        env=srv.env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, f"could not read server_version_num: {probe.stderr!r}"
    version_num = int(probe.stdout.strip())
    if version_num < _MIN_SERVER_VERSION_NUM:
        _unavailable(
            f"server is PostgreSQL {version_num} but the INHERIT/SET "
            "membership split this suite proves was introduced in 16 -- an "
            "older server rejects `GRANT ... WITH INHERIT TRUE, SET FALSE` "
            "outright, so there is nothing here it can measure"
        )


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    """A Postgres to talk to: the CI service if present, else a temp cluster."""
    external = _server_from_env()
    if external is not None:
        probe = subprocess.run(
            [
                PSQL,
                "-X",
                "-h",
                external.host,
                "-p",
                external.port,
                "-U",
                external.user,
                "-d",
                "postgres",
                "-tAc",
                "SELECT 1",
            ],
            env=external.env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
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

    root = Path(tempfile.mkdtemp(prefix="omn17316-pg-"))
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


def _scalar(srv: Server, database: str, sql: str) -> str:
    result = _psql(srv, database, "-tA", "-c", sql)
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


def _quote_literal(value: str) -> str:
    """Single-quote a value for SQL, doubling any embedded quote."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


@dataclass(frozen=True)
class Lane:
    """A throwaway database plus the two cluster roles the proof needs."""

    database: str
    owner: str
    migrator: str


@pytest.fixture
def lane(server: Server) -> Iterator[Lane]:
    """A lane whose delegation_events is owned by a role the migrator is NOT.

    ``tenant_id`` is created already ``uuid`` on purpose: that is 0033's
    already-converted branch, which reaches the ownership guard with the least
    fixture between the runner and the thing under test. The guard sits BEFORE
    the ``IF v_convert`` split, so this path exercises it identically to the
    conversion path.
    """
    tag = uuid.uuid4().hex[:12]
    database = f"omn17316_{tag}"
    owner = f"omn17316_owner_{tag}"
    migrator = f"omn17316_migrator_{tag}"

    created = _psql(server, "postgres", "-c", f"CREATE DATABASE {database}")
    assert created.returncode == 0, created.stderr

    # app_dashboard is environment-provisioned on every real lane (the GRANT at
    # the end of the migration targets it), and is cluster-wide, so a shared CI
    # server may already carry it.
    if (
        _scalar(
            server, "postgres", "SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard'"
        )
        != "1"
    ):
        _psql(
            server,
            "postgres",
            "-c",
            "CREATE ROLE app_dashboard NOLOGIN NOSUPERUSER NOBYPASSRLS",
        )

    # The migrator logs in as itself, so it must authenticate the same way the
    # fixture's own connections do. Giving it the SERVER's password (rather
    # than a literal) keeps one PGPASSWORD valid for every connection this
    # module makes -- on a CI service container with scram auth as much as on
    # the hermetic trust-auth cluster, where the clause is simply inert.
    password_clause = (
        f"PASSWORD {_quote_literal(server.password)}" if server.password else ""
    )
    bootstrap = _psql(
        server,
        database,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        f"""
        CREATE ROLE {owner} NOLOGIN NOSUPERUSER NOBYPASSRLS;
        CREATE ROLE {migrator} LOGIN NOSUPERUSER NOBYPASSRLS {password_clause};
        GRANT CREATE, USAGE ON SCHEMA public TO {owner};
        SET ROLE {owner};
        CREATE TABLE delegation_events (
            correlation_id TEXT,
            tenant_id UUID NOT NULL
                DEFAULT '820272f9-4aaf-5add-a2df-0af942852ab2'
        );
        ALTER TABLE delegation_events ENABLE ROW LEVEL SECURITY;
        CREATE POLICY tenant_isolation ON delegation_events FOR ALL
          USING (tenant_id = current_setting('app.tenant_id', true)::uuid)
          WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid);
        RESET ROLE;
        """,
    )
    assert bootstrap.returncode == 0, bootstrap.stderr

    try:
        yield Lane(database=database, owner=owner, migrator=migrator)
    finally:
        _psql(
            server, "postgres", "-c", f"DROP DATABASE IF EXISTS {database} WITH (FORCE)"
        )
        for role in (migrator, owner):
            _psql(server, "postgres", "-c", f"DROP ROLE IF EXISTS {role}")


def _grant_set_false(server: Server, lane: Lane) -> None:
    """The membership the whole class turns on: inherits, cannot SET ROLE."""
    granted = _psql(
        server,
        lane.database,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        f"GRANT {lane.owner} TO {lane.migrator} WITH INHERIT TRUE, SET FALSE",
    )
    assert granted.returncode == 0, granted.stderr


def _grant_default(server: Server, lane: Lane) -> None:
    """The PostgreSQL 16 default: confers BOTH INHERIT and SET."""
    granted = _psql(
        server,
        lane.database,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        f"GRANT {lane.owner} TO {lane.migrator}",
    )
    assert granted.returncode == 0, granted.stderr


# ---------------------------------------------------------------------------
# 1. The platform fact the whole finding rests on.
# ---------------------------------------------------------------------------


def test_usage_and_set_are_independent_predicates(server: Server, lane: Lane) -> None:
    """PostgreSQL 16: ``WITH INHERIT TRUE, SET FALSE`` splits the predicates.

    Measured, not cited. If a future PostgreSQL collapses these back into one,
    this test is where that is discovered -- before the guard built on the
    distinction silently becomes redundant or wrong.

    ``MEMBER`` is asserted too because it is the obvious "just use a different
    privilege string" fix, and it does NOT work: it is true under SET FALSE as
    well, so it is no more a proxy for SET ROLE than USAGE is.
    """
    _grant_set_false(server, lane)

    row = _scalar(
        server,
        lane.database,
        f"SELECT pg_has_role('{lane.migrator}','{lane.owner}','USAGE')::text "
        f"|| ',' || pg_has_role('{lane.migrator}','{lane.owner}','SET')::text "
        f"|| ',' || pg_has_role('{lane.migrator}','{lane.owner}','MEMBER')::text",
    )

    assert row == "true,false,true", (
        "expected USAGE=true SET=false MEMBER=true under "
        f"`WITH INHERIT TRUE, SET FALSE`; got {row!r}. The OMN-17316 guard is "
        "built on these three being independent."
    )


# ---------------------------------------------------------------------------
# 2. RED -- the defect, against 0033's real bytes.
# ---------------------------------------------------------------------------


def test_0033_passes_its_own_guard_and_then_aborts_opaquely(
    server: Server, lane: Lane
) -> None:
    """RED. 0033's guard admits an identity that cannot do what comes next.

    Two things are asserted, and the second is the point:

    * the migration FAILS (it cannot complete without the role switch), and
    * it fails with PostgreSQL's bare ``permission denied to set role``, from
      the ``set_config`` statement -- NOT with 0033's own named refusal.

    A guard whose failure message never appears is not a guard.
    """
    _grant_set_false(server, lane)

    result = _apply_as(server, lane.database, _DEFECTIVE, lane.migrator)
    combined = result.stdout + result.stderr

    assert result.returncode != 0, (
        "0033 unexpectedly SUCCEEDED under a SET FALSE membership -- it cannot "
        f"have performed the role switch.\n{combined}"
    )
    assert _OPAQUE in combined, (
        "expected the opaque PostgreSQL refusal that OMN-17316 reports; got:\n"
        f"{combined}"
    )
    assert "set_config" in combined, (
        "the abort should be attributed to the set_config('role', ...) "
        f"statement two statements past the guard; got:\n{combined}"
    )
    # NB: psql prefixes each diagnostic with the migration's absolute path, and
    # this ticket's worktree is named after the ticket -- so a bare
    # `"OMN-17316" in combined` matches the PATH and proves nothing. Every
    # assertion in this module keys on message text that cannot be a filename.
    assert _REPAIRED_MARKER not in combined, (
        "0033 must not already carry the repaired guard -- if it does, this "
        "RED test is measuring the wrong file."
    )


def test_0033_aborts_before_it_changes_anything(server: Server, lane: Lane) -> None:
    """The abort is inside the single DO block, so the transaction rolls back whole.

    This is the one piece of good news in the finding and it is worth pinning:
    the failure is loud and total, never a half-converted relation. If a future
    revision moves DDL ahead of the role switch, this is what catches it.
    """
    _grant_set_false(server, lane)

    _apply_as(server, lane.database, _DEFECTIVE, lane.migrator)

    assert (
        _scalar(
            server,
            lane.database,
            "SELECT atttypid::regtype::text FROM pg_attribute "
            "WHERE attrelid = 'delegation_events'::regclass AND attname = 'tenant_id'",
        )
        == "uuid"
    )
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT polname FROM pg_policy WHERE polrelid = 'delegation_events'::regclass",
        )
        == "tenant_isolation"
    )
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT count(*)::text FROM pg_attribute "
            "WHERE attrelid = 'delegation_events'::regclass "
            "AND attname = 'omn16930_resolved_tenant_uuid' AND NOT attisdropped",
        )
        == "0"
    ), "the scratch column outlived a failed run"


# ---------------------------------------------------------------------------
# 3. GREEN -- 0034 refuses early, by name, and says which predicate failed.
# ---------------------------------------------------------------------------


def test_0034_refuses_early_and_names_the_failing_predicate(
    server: Server, lane: Lane
) -> None:
    """GREEN. The same membership now produces the named refusal, before the switch.

    The message has to be actionable on its own -- an operator reading it in a
    deploy log has no access to this test -- so it is asserted to name the
    ticket, the predicate that failed, the fact that the membership carries
    SET FALSE, and the remediation.
    """
    _grant_set_false(server, lane)

    result = _apply_as(server, lane.database, _REPAIRED, lane.migrator)
    combined = result.stdout + result.stderr

    assert result.returncode != 0, (
        f"0034 must still refuse a SET FALSE membership.\n{combined}"
    )
    assert "OMN-17316: the migrate identity" in combined, (
        f"the refusal is not named:\n{combined}"
    )
    assert _REPAIRED_MARKER in combined, (
        f"the refusal does not say WHICH predicate failed:\n{combined}"
    )
    assert "SET FALSE" in combined, (
        f"the refusal does not name the membership option at fault:\n{combined}"
    )
    assert "WITH SET TRUE" in combined, (
        f"the refusal does not carry the remediation:\n{combined}"
    )
    assert _OPAQUE not in result.stderr.split(_REPAIRED_MARKER)[0], (
        "0034 still reached set_config before refusing -- the guard must come "
        f"FIRST:\n{combined}"
    )


def test_0034_still_refuses_a_non_member_and_names_the_other_predicate(
    server: Server, lane: Lane
) -> None:
    """The USAGE half is not lost to the repair.

    Adding the SET predicate would be a regression if it replaced the USAGE
    one: USAGE carries the RLS-blindness rationale (OMN-16493), SET carries the
    role switch. With NO membership at all, the USAGE arm must fire first.
    """
    result = _apply_as(server, lane.database, _REPAIRED, lane.migrator)
    combined = result.stdout + result.stderr

    assert result.returncode != 0, f"a non-member must be refused.\n{combined}"
    assert "OMN-16930" in combined and "USAGE is" in combined, (
        f"the USAGE arm did not fire for an identity with no membership:\n{combined}"
    )


def test_0034_completes_under_a_default_membership(server: Server, lane: Lane) -> None:
    """The guard must not over-refuse: the PG16 default GRANT still works.

    A guard that refuses everything is as useless as one that refuses nothing,
    and this is the path every real lane is on today -- which is exactly why
    the defect was latent and CI was green.
    """
    _grant_default(server, lane)

    result = _apply_as(server, lane.database, _REPAIRED, lane.migrator)
    combined = result.stdout + result.stderr

    assert result.returncode == 0, (
        f"0034 refused a DEFAULT (INHERIT+SET) membership.\n{combined}"
    )
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT polname FROM pg_policy WHERE polrelid = 'delegation_events'::regclass",
        )
        == "tenant_isolation"
    )
    assert (
        _scalar(
            server,
            lane.database,
            "SELECT has_table_privilege('app_dashboard','delegation_events','SELECT')::text",
        )
        == "true"
    ), "the OMN-14894 app_dashboard GRANT did not land"


# ---------------------------------------------------------------------------
# 4. Source-shape guards -- cheap, and they run without a Postgres.
# ---------------------------------------------------------------------------


def test_0034_checks_both_predicates_before_the_role_switch() -> None:
    """Ordering is the whole finding: a guard after the switch guards nothing."""
    body = _REPAIRED.read_text(encoding="utf-8")
    executable = body[body.index("DO $$") :]

    usage = executable.index("pg_has_role(current_user, v_owner, 'USAGE')")
    set_check = executable.index("pg_has_role(current_user, v_owner, 'SET')")
    switch = executable.index("set_config('role', v_owner::text, true)")

    assert usage < switch and set_check < switch, (
        "both membership predicates must be tested BEFORE "
        "set_config('role', ...); 0033's defect was testing one and "
        "exercising the other."
    )


def test_0033_is_left_untouched_and_still_carries_the_defect() -> None:
    """0034 SUPERSEDES 0033; it does not edit it.

    ``check_migration_append_only.py`` freezes 0033's bytes (it is declared in
    ``_ledger/application-migrations.tsv``), and the supersession row is the
    only escape it accepts. If someone "helpfully" repairs 0033 in place later,
    every lane that recorded its checksum breaks -- so the defect staying put
    is the correct end state, and is asserted rather than left to trust.
    """
    body = _DEFECTIVE.read_text(encoding="utf-8")
    executable = body[body.index("DO $$") :]

    assert "pg_has_role(current_user, v_owner, 'SET')" not in executable, (
        "0033 was edited in place. Its bytes are frozen: the repair belongs in "
        "0034 (OMN-16705 append-only ratchet, OMN-17316)."
    )


def test_0034_introduces_no_dynamic_sql() -> None:
    """OMN-15361. The role switch stays a VALUE, never composed SQL text."""
    body = _REPAIRED.read_text(encoding="utf-8")
    executable = body[body.index("DO $$") :]

    for forbidden in ("EXECUTE format(", "EXECUTE '", 'EXECUTE "', "quote_ident("):
        assert forbidden not in executable, (
            f"0034 composes SQL at runtime ({forbidden!r}); the OMN-15361 gate "
            "rejects dynamic SQL and set_config('role', <value>, true) is the "
            "static form that avoids it."
        )


def test_0034_keeps_the_app_dashboard_grant_in_file() -> None:
    """OMN-14894 ratchet: whoever (re)creates the policy grants in the same file."""
    executable = _REPAIRED.read_text(encoding="utf-8")
    assert re.search(
        r"^\s*GRANT SELECT ON delegation_events TO app_dashboard;",
        executable,
        re.M,
    ), "0034 dropped the OMN-14894 app_dashboard GRANT"
