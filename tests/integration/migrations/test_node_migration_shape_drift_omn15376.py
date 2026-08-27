# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Execution proof for the node-migration shape-drift class (OMN-15376).

Runs the REAL vendored SQL through the SAME psql invocation the deploy-time
runners use (``psql -v ON_ERROR_STOP=1 -f <file>``) against a REAL Postgres, on
two paths:

* **fresh** — empty database, the tables do not exist.
* **drifted** — every table the corpus creates already exists carrying ONLY its
  first declared column, which is the shape class the live failure evidences:
  ``relation "llm_cost_aggregates" already exists, skipping`` followed by
  ``ERROR: column "aggregation_key" does not exist`` (deploy-onex-dev run
  30418878385, ``0001_create_llm_cost_aggregates.sql:64``).

Three claims are proven by execution, not by inspection:

1. **RED** — the migration WITHOUT its reconciliation block fails on the drifted
   shape with the exact live error. The unfixed variant is derived by deleting
   the ``BEGIN/END OMN-15376 shape reconciliation`` region from the real file,
   so the RED is against "exists but wrong", never a hand-written surrogate.
2. **GREEN** — the real file succeeds on the same drifted shape.
3. **CONVERGENCE** — after the whole corpus applies on both paths, the two
   schemas are byte-identical (columns + types + nullability + defaults +
   constraints + indexes + views + triggers + RLS flags). A fix that merely
   stops erroring, while leaving the drifted table a different shape from a
   fresh one, would pass 1 and 2 and fail this.

Fenced ids (OMN-14974 / OMN-15313 / OMN-15335) are excluded exactly as both
runners exclude them; the list is read from ``scripts/run-forward-migrations.sh``
rather than restated.

Postgres source, in order: an already-running server named by
``OMNIBASE_INFRA_DB_URL`` / ``POSTGRES_HOST`` (this is how the CI
``migration-integration`` job supplies one), else a hermetic ephemeral cluster
from local ``initdb``. Skips only when neither exists.

OMN-16412: a non-loopback ambient ``OMNIBASE_INFRA_DB_URL``/``POSTGRES_HOST``
is a hard error unless ``OMN15376_ALLOW_REMOTE_PG=1`` is also set -- this
suite creates/drops throwaway databases on whatever host it is handed, and a
persistent dev-shell env var has silently pointed it at a live shared Postgres
before. See ``_reject_unless_loopback_or_opted_in`` below.

Run: uv run pytest tests/integration/migrations/test_node_migration_shape_drift_omn15376.py -v

Ticket: OMN-15376 (class), OMN-15302 (second live instance)
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

from tests.helpers.util_migration_shape import (
    fenced_migration_ids,
    guarded_create_tables,
    node_migration_files,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.serial]

RECONCILIATION_BEGIN = "-- ---- BEGIN OMN-15376 shape reconciliation:"
RECONCILIATION_END = "-- ---- END OMN-15376 shape reconciliation:"

# Roles the corpus GRANTs to. They are environment-provisioned in every real
# lane (forward migration 094 / the RDS bootstrap), so the fixture provisions
# them too -- otherwise this suite would prove role tolerance, not shape drift.
SEED_ROLES = ("app_dashboard", "role_omnidash", "omninodeadmin")

# The two live instances, pinned to the error text and file line the deploy
# printed. A change that moves either line without updating this is a signal.
LIVE_SIGNATURES = (
    pytest.param(
        "node_projection_cost_summary",
        "0001_create_llm_cost_aggregates.sql",
        "llm_cost_aggregates",
        "aggregation_key",
        id="OMN-15376-llm_cost_aggregates.aggregation_key",
    ),
    pytest.param(
        "node_projection_baselines",
        "0001_create_baselines_tables.sql",
        "baselines_comparisons",
        "snapshot_id",
        id="OMN-15302-baselines_comparisons.snapshot_id",
    ),
)


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

# When a Postgres has been handed to this suite deliberately (the CI
# migration-integration job sets this), a skip is a vacuous green, so every
# skip path below becomes a hard failure instead.
_REQUIRE_PG = os.environ.get("OMN15376_REQUIRE_PG") == "1"


def _unavailable(reason: str) -> None:
    if _REQUIRE_PG:
        raise AssertionError(f"OMN15376_REQUIRE_PG=1 but {reason}")
    pytest.skip(reason)


if _PSQL is None:  # pragma: no cover - environment dependent
    if os.environ.get("OMN15376_REQUIRE_PG") == "1":
        raise AssertionError("OMN15376_REQUIRE_PG=1 but psql is not available")
    pytest.skip("psql not available", allow_module_level=True)


@dataclass(frozen=True)
class Server:
    """Connection coordinates for whichever Postgres this run got."""

    host: str
    port: str
    user: str
    password: str

    def env(self) -> dict[str, str]:
        merged = dict(os.environ)
        merged["PGPASSWORD"] = self.password
        return merged


# OMN-16412: this suite creates and drops throwaway databases (`CREATE DATABASE
# omn15376_<uuid>`) on whatever server it is handed, so silently adopting an
# ambient OMNIBASE_INFRA_DB_URL/POSTGRES_HOST pointing at a shared/live host is
# a real contamination vector -- a persistent dev-shell env var leaked several
# local runs onto the live .201 stability-test Postgres with no indication
# anything unusual was happening (35 leftover throwaway DBs found and dropped,
# see the ticket). A loopback host (localhost/127.0.0.1/::1/a unix socket
# path) is always this process's own box and stays allowed with no opt-in.
# Anything else -- including a private-network address that LOOKS like it
# could be "just Docker" -- requires the explicit OMN15376_ALLOW_REMOTE_PG=1
# opt-in, because private-RFC1918 is exactly what the live .201 lane is too
# (192.168.86.0/24): there is no way to tell "CI's own ephemeral service
# container, reached via the Docker bridge gateway because this self-hosted
# runner executes inside a container and 127.0.0.1 doesn't reach a sibling
# container" apart from "someone's persistent dev-shell var pointing at a
# shared live lane" by host shape alone. CI sets the opt-in explicitly, once,
# in the one job/step that owns a Postgres it just spun up itself.
#
# Read fresh on every call (unlike ``_REQUIRE_PG`` below, which is frozen at
# import time) so tests can toggle it via monkeypatch.
def _allow_remote_pg() -> bool:
    return os.environ.get("OMN15376_ALLOW_REMOTE_PG") == "1"


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
        f"{env_var}={host!r} resolves to a non-loopback host. Refusing to "
        "silently run node-migration shape-drift tests (which CREATE/DROP "
        "throwaway databases) against it -- this is very likely a stray "
        "ambient env var pointing at someone's shared/live Postgres, not a "
        "sandbox meant for this suite (OMN-16412). If this is a genuinely "
        "intentional live-integration run, set OMN15376_ALLOW_REMOTE_PG=1 to "
        "opt in explicitly."
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


# OMN-16412: guard tests for the ambient-DSN host check. These are pure
# env-var/parsing tests -- they never touch a real Postgres and do not use the
# ``server`` fixture -- but they still live in this module (rather than in a
# separate always-collected file) because the whole file is skipped at import
# when ``psql`` is unavailable, matching every other test here.
def test_server_from_env_accepts_loopback_ambient_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("OMN15376_ALLOW_REMOTE_PG", raising=False)
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL",
        "postgresql://postgres:secret@127.0.0.1:5432/omnibase_infra",
    )

    srv = _server_from_env()

    assert srv is not None
    assert srv.host == "127.0.0.1"
    assert srv.port == "5432"


def test_server_from_env_accepts_loopback_ambient_postgres_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    monkeypatch.delenv("OMN15376_ALLOW_REMOTE_PG", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "localhost")

    srv = _server_from_env()

    assert srv is not None
    assert srv.host == "localhost"


def test_server_from_env_rejects_non_loopback_ambient_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-loopback ambient OMNIBASE_INFRA_DB_URL is a hard error, never a silent fallback.

    This is the exact contamination vector from OMN-16412: a persistent
    dev-shell OMNIBASE_INFRA_DB_URL pointed at the live .201 stability-test
    Postgres and the fixture adopted it with no warning.
    """
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.delenv("OMN15376_ALLOW_REMOTE_PG", raising=False)
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL",
        "postgresql://postgres:secret@192.168.86.201:5436/omnibase_infra",
    )

    with pytest.raises(AssertionError, match="OMNIBASE_INFRA_DB_URL"):
        _server_from_env()


def test_server_from_env_rejects_non_loopback_ambient_postgres_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The POSTGRES_HOST fallback path gets the identical guard as the DSN path."""
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    monkeypatch.delenv("OMN15376_ALLOW_REMOTE_PG", raising=False)
    monkeypatch.setenv("POSTGRES_HOST", "192.168.86.201")

    with pytest.raises(AssertionError, match="POSTGRES_HOST"):
        _server_from_env()


def test_server_from_env_allows_non_loopback_ambient_dsn_with_explicit_optin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OMN15376_ALLOW_REMOTE_PG=1 is the documented, explicit opt-in for a genuinely
    intentional live-integration run (e.g. this repo's own nightly-integration.yml,
    which resolves its dockerized e2e stack to a non-loopback Docker-bridge/DNS host)."""
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    monkeypatch.setenv("OMN15376_ALLOW_REMOTE_PG", "1")
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL",
        "postgresql://postgres:secret@192.168.86.201:5436/omnibase_infra",
    )

    srv = _server_from_env()

    assert srv is not None
    assert srv.host == "192.168.86.201"


def test_server_from_env_returns_none_when_nothing_ambient_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ambient var at all is untouched by the guard -- falls through to the
    ephemeral-cluster path exactly as before OMN-16412."""
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    monkeypatch.delenv("POSTGRES_HOST", raising=False)

    assert _server_from_env() is None


@pytest.mark.parametrize(
    "host", ["localhost", "127.0.0.1", "::1", "/var/run/postgresql/.s.PGSQL.5432"]
)
def test_is_loopback_host_accepts_local_forms(host: str) -> None:
    assert _is_loopback_host(host) is True


@pytest.mark.parametrize(
    "host",
    [
        "192.168.86.201",
        "10.0.0.5",
        "172.18.0.1",
        "example.com",
        "omnibase-infra-postgres",
    ],
)
def test_is_loopback_host_rejects_remote_forms(host: str) -> None:
    assert _is_loopback_host(host) is False


# OMN-16692: guard tests for the psql-output reader. Pure CompletedProcess
# handling -- no Postgres, no ``server`` fixture -- but kept in this module for
# the same reason as the OMN-16412 guards above.
def test_psql_stdout_fails_loudly_when_the_command_did_not_run() -> None:
    """A psql that exited non-zero is a prerequisite failure, never content.

    The live OMN-16692 shape: the binary was unavailable, stdout was empty, and
    the bare ``assert constraints.stdout.strip() == "<expected>"`` reported
    ``assert '' == 'delegation_routing_tenant_overlay_pkey:p,...'`` -- a
    schema-drift-shaped failure for a missing-psql cause.
    """
    failed = subprocess.CompletedProcess(
        args=["psql"],
        returncode=127,
        stdout="",
        stderr="psql: command not found",
    )

    with pytest.raises(AssertionError) as excinfo:
        _psql_stdout(failed, "constraint probe")

    message = str(excinfo.value)
    assert "psql did not run successfully (constraint probe)" in message
    assert "NOT schema drift" in message
    assert "psql: command not found" in message


def test_psql_stdout_returns_stripped_stdout_on_success() -> None:
    """The success path is unchanged: stripped stdout, ready to compare."""
    ok = subprocess.CompletedProcess(
        args=["psql"], returncode=0, stdout=" 2 \n", stderr=""
    )

    assert _psql_stdout(ok, "row count") == "2"


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    """A Postgres to talk to: the CI service if present, else a temp cluster."""
    external = _server_from_env()
    if external is not None:
        probe = subprocess.run(
            [
                _PSQL,
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
            yield external
            return

    if _INITDB is None or _PG_CTL is None:  # pragma: no cover
        _unavailable(
            "no reachable Postgres (OMNIBASE_INFRA_DB_URL/POSTGRES_HOST) and no "
            "local initdb to build an ephemeral cluster"
        )

    root = Path(tempfile.mkdtemp(prefix="omn15376-pg-"))
    sock = root / "sock"
    sock.mkdir()
    data = root / "data"
    subprocess.run(
        [_INITDB, "-D", str(data), "-U", "postgres", "-A", "trust"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            _PG_CTL,
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
        yield Server(host=str(sock), port="5432", user="postgres", password="")
    finally:
        subprocess.run(
            [_PG_CTL, "-D", str(data), "-m", "immediate", "stop"],
            check=False,
            capture_output=True,
        )
        shutil.rmtree(root, ignore_errors=True)


def _psql(srv: Server, database: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            _PSQL,
            "-X",
            "-q",
            "-h",
            srv.host,
            "-p",
            srv.port,
            "-U",
            srv.user,
            "-d",
            database,
            *args,
        ],
        env=srv.env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _psql_stdout(result: subprocess.CompletedProcess[str], what: str) -> str:
    """Stripped stdout of a psql invocation that MUST have succeeded (OMN-16692).

    A psql that did not RUN -- binary missing or broken, server unreachable,
    role/permission error, SQL error under ``ON_ERROR_STOP=1`` -- exits
    non-zero with EMPTY stdout. Asserting straight on ``.stdout`` therefore
    converts a missing prerequisite into a content mismatch that reads as real
    schema drift: the live case was
    ``assert '' == 'delegation_routing_tenant_overlay_pkey:p,...'`` on a runner
    whose CI log had already said ``psql client unavailable``.

    Every read of psql output goes through here so the failure names the
    invocation and carries psql's own diagnostics instead of an empty string.
    """
    assert result.returncode == 0, (
        f"psql did not run successfully ({what}): exit {result.returncode}. "
        f"This is a missing/failed prerequisite, NOT schema drift.\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    return result.stdout.strip()


def _psql_script(
    srv: Server, database: str, sql: str, **variables: str
) -> subprocess.CompletedProcess[str]:
    """Run SQL from stdin so psql :'var' interpolation applies.

    ``psql -c`` sends its argument to the server verbatim and does NOT expand
    ``:'var'`` (verified: ``syntax error at or near ":"``), so anything that
    needs a value substituted has to arrive on stdin.
    """
    args = [
        _PSQL,
        "-X",
        "-q",
        "-t",
        "-A",
        "-h",
        srv.host,
        "-p",
        srv.port,
        "-U",
        srv.user,
        "-d",
        database,
    ]
    for name, value in variables.items():
        args += ["-v", f"{name}={value}"]
    args += ["-f", "-"]
    return subprocess.run(
        args, input=sql, env=srv.env(), capture_output=True, text=True, check=False
    )


def _new_database(srv: Server) -> str:
    name = f"omn15376_{uuid.uuid4().hex[:12]}"
    created = _psql(srv, "postgres", "-c", f"CREATE DATABASE {name}")
    assert created.returncode == 0, created.stderr
    # Several node-owned migrations assert the operator-provisioned
    # omninode_internal schema instead of creating it, matching the managed RDS
    # lane where CREATE SCHEMA is not available to the migration role.
    schema = _psql(
        srv,
        name,
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "CREATE SCHEMA IF NOT EXISTS omninode_internal;",
    )
    assert schema.returncode == 0, schema.stderr
    for role in SEED_ROLES:
        # Roles are CLUSTER-wide; a shared CI server may already have them.
        # Passed as a psql variable rather than interpolated into the SQL text.
        exists = _psql_script(
            srv,
            "postgres",
            "SELECT 1 FROM pg_roles WHERE rolname = :'role'",
            role=role,
        )
        if "1" not in _psql_stdout(exists, f"pg_roles probe for {role}"):
            _psql_script(
                srv,
                "postgres",
                'CREATE ROLE :"role" NOSUPERUSER NOBYPASSRLS;',
                role=role,
            )
    return name


def _drop_database(srv: Server, name: str) -> None:
    _psql(srv, "postgres", "-c", f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")


# FENCE-PARITY GAP, found by running this suite (OMN-15379):
# omninode_infra's k8s Job runner fences SEVEN node ids; this repo's
# scripts/run-forward-migrations.sh fences SIX -- it is missing
# node:node_projection_registration:0002_node_service_registry_tenant_rls.sql.
# 0002 ALTERs node_service_registry, which only the FENCED 0000 creates, so on
# any lane where that table does not already exist the compose runner skips 0000
# and then dies on 0002 with `relation "node_service_registry" does not exist`.
# That is a real wall and a separate ticket; it is NOT the shape-drift class and
# must not be what makes this suite red. Excluded here with its ticket, not
# silently: the fence itself is operator-gated and is not edited from this lane.
_K8S_ONLY_FENCED = frozenset(
    {"node:node_projection_registration:0002_node_service_registry_tenant_rls.sql"}
)


def _corpus() -> list[tuple[str, Path]]:
    fenced = fenced_migration_ids() | _K8S_ONLY_FENCED
    return [
        (migration_id, path)
        for migration_id, path in node_migration_files()
        if migration_id not in fenced
    ]


def _strip_reconciliation(sql: str) -> str:
    """Delete every BEGIN/END reconciliation region -- the pre-fix variant."""
    pattern = re.compile(
        re.escape(RECONCILIATION_BEGIN)
        + r".*?"
        + re.escape(RECONCILIATION_END)
        + r"[^\n]*\n",
        re.S,
    )
    stripped, count = pattern.subn("", sql)
    assert count > 0, "no reconciliation region found — RED would be vacuous"
    return stripped


def _drift_seed_statements() -> list[str]:
    """``CREATE TABLE`` for every corpus table, carrying only its first column."""
    statements: list[str] = []
    for _migration_id, path in _corpus():
        for table in guarded_create_tables(path.read_text(encoding="utf-8")):
            # This proof snapshots public-schema drift. Non-public tables carry
            # separate topology/operator preconditions and assertions.
            if "." in table.qualified_name and not table.qualified_name.startswith(
                "public."
            ):
                continue
            if not table.columns or table.columns[0].generated:
                continue
            first = table.columns[0]
            statements.append(
                f"CREATE TABLE IF NOT EXISTS {table.qualified_name} "
                f"({first.seed_ddl_fragment()});"
            )
    return statements


def _apply_corpus(srv: Server, database: str) -> list[tuple[str, str]]:
    return _apply_corpus_files(srv, database, _corpus())


def _apply_corpus_files(
    srv: Server, database: str, files: list[tuple[str, Path]]
) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for migration_id, path in files:
        result = _psql(srv, database, "-v", "ON_ERROR_STOP=1", "-f", str(path))
        if result.returncode != 0:
            output = result.stdout + result.stderr
            line = next(
                (ln for ln in output.splitlines() if "ERROR:" in ln), output[-400:]
            )
            failures.append((migration_id, line))
    return failures


_SCHEMA_SNAPSHOT_SQL = """
SELECT 'COL|'||c.relname||'|'||a.attname||'|'
       ||format_type(a.atttypid, a.atttypmod)||'|'||a.attnotnull::text||'|'
       ||coalesce(pg_get_expr(d.adbin, d.adrelid), '')||'|'||a.attgenerated::text
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
WHERE c.relkind IN ('r', 'p', 'v', 'm')
UNION ALL
SELECT 'CON|'||c.relname||'|'||con.contype::text||'|'
       ||pg_get_constraintdef(con.oid)
FROM pg_constraint con
JOIN pg_class c ON c.oid = con.conrelid
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
UNION ALL
SELECT 'IDX|'||tablename||'|'||indexdef FROM pg_indexes WHERE schemaname = 'public'
UNION ALL
SELECT 'VIEW|'||viewname||'|'||md5(definition) FROM pg_views WHERE schemaname = 'public'
UNION ALL
SELECT 'TRG|'||c.relname||'|'||t.tgname
FROM pg_trigger t
JOIN pg_class c ON c.oid = t.tgrelid
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
WHERE NOT t.tgisinternal
UNION ALL
SELECT 'RLS|'||c.relname||'|'||c.relrowsecurity::text||'|'
       ||c.relforcerowsecurity::text
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
WHERE c.relkind = 'r'
UNION ALL
SELECT 'POL|'||tablename||'|'||policyname||'|'||coalesce(qual, '')
FROM pg_policies WHERE schemaname = 'public'
ORDER BY 1
"""


def _schema_snapshot(srv: Server, database: str) -> list[str]:
    result = _psql(
        srv, database, "-t", "-A", "-v", "ON_ERROR_STOP=1", "-c", _SCHEMA_SNAPSHOT_SQL
    )
    snapshot = _psql_stdout(result, f"schema snapshot of {database}")
    return sorted(line for line in snapshot.splitlines() if line.strip())


_CREATE_TYPE_RE = re.compile(r"CREATE\s+TYPE\s+([A-Za-z0-9_]+)\s+AS\s+", re.I)


def _migration_path(node: str, filename: str) -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "docker"
        / "migrations"
        / "forward"
        / "nodes"
        / node
        / filename
    )


def _drifted_table_ddl(path: Path, table: str, omit_column: str) -> str:
    """DDL for a pre-existing table that predates ``path``.

    Carries every declared column except the witness, and except any column
    whose type this very migration defines -- a table that predates the
    migration cannot be using an enum the migration itself creates.
    """
    sql = path.read_text(encoding="utf-8")
    own_types = {name.lower() for name in _CREATE_TYPE_RE.findall(sql)}
    target = {t.bare_name: t for t in guarded_create_tables(sql)}[table]
    kept = [
        column
        for column in target.columns
        if column.name.strip('"') != omit_column
        and column.type_text.split("(")[0].strip().lower() not in own_types
        and not column.generated
    ]
    assert kept, f"{table}: drift seed would be empty"
    fragments = ", ".join(column.seed_ddl_fragment() for column in kept)
    return f"CREATE TABLE {target.qualified_name} ({fragments});"


@pytest.mark.parametrize(("node", "filename", "table", "column"), LIVE_SIGNATURES)
def test_unfixed_migration_is_red_on_the_drifted_shape(
    server: Server, node: str, filename: str, table: str, column: str
) -> None:
    """RED: strip the reconciliation, seed the drift, get the live error back."""
    path = _migration_path(node, filename)
    declared = {
        c.name.strip('"')
        for t in guarded_create_tables(path.read_text("utf-8"))
        if t.bare_name == table
        for c in t.columns
    }
    assert column in declared, f"{column} is not declared by {table} - premise moved"

    database = _new_database(server)
    try:
        seeded = _psql(
            server,
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            _drifted_table_ddl(path, table, column),
        )
        assert seeded.returncode == 0, seeded.stderr

        unfixed = Path(tempfile.mkdtemp(prefix="omn15376-red-")) / filename
        unfixed.write_text(
            _strip_reconciliation(path.read_text("utf-8")), encoding="utf-8"
        )
        result = _psql(server, database, "-v", "ON_ERROR_STOP=1", "-f", str(unfixed))
        output = result.stdout + result.stderr

        assert result.returncode != 0, (
            f"the UNFIXED {node}/{filename} succeeded against a table missing "
            f"{column} — the RED premise is vacuous"
        )
        assert f'column "{column}" does not exist' in output, output
        assert f'relation "{table}" already exists, skipping' in output, output
    finally:
        _drop_database(server, database)


@pytest.mark.parametrize(("node", "filename", "table", "column"), LIVE_SIGNATURES)
def test_fixed_migration_is_green_on_the_same_drifted_shape(
    server: Server, node: str, filename: str, table: str, column: str
) -> None:
    """GREEN: the real file converges the same drifted table and exits 0."""
    path = _migration_path(node, filename)
    database = _new_database(server)
    try:
        seeded = _psql(
            server,
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            _drifted_table_ddl(path, table, column),
        )
        assert seeded.returncode == 0, seeded.stderr
        result = _psql(server, database, "-v", "ON_ERROR_STOP=1", "-f", str(path))
        assert result.returncode == 0, result.stdout + result.stderr

        # Passed as psql variables, never interpolated into the SQL text.
        present = _psql_script(
            server,
            database,
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = :'tbl' AND column_name = :'col';",
            tbl=table,
            col=column,
        )
        assert _psql_stdout(present, f"{table}.{column} presence probe") == "1", (
            present.stdout
        )
    finally:
        _drop_database(server, database)


def test_reconciliation_preserves_pre_existing_rows(server: Server) -> None:
    """A drifted table with data keeps every row; nothing is dropped."""
    path = _migration_path(
        "node_projection_cost_summary", "0001_create_llm_cost_aggregates.sql"
    )
    database = _new_database(server)
    try:
        created = _psql(
            server,
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            _drifted_table_ddl(path, "llm_cost_aggregates", "aggregation_key"),
        )
        assert created.returncode == 0, created.stderr
        seeded = _psql(
            server,
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "INSERT INTO llm_cost_aggregates (total_cost_usd, total_tokens, "
            "call_count) VALUES (1.5, 10, 2), (2.5, 20, 4);",
        )
        assert seeded.returncode == 0, seeded.stderr

        # aggregation_key is NOT NULL with no DEFAULT: on a POPULATED drifted
        # table it cannot be converged without inventing data, so the migration
        # must refuse LOUDLY and name the conflict rather than guess.
        result = _psql(server, database, "-v", "ON_ERROR_STOP=1", "-f", str(path))
        output = result.stdout + result.stderr
        assert result.returncode != 0, output
        assert "OMN-15376" in output, output
        assert "aggregation_key" in output, output
        assert "data ruling" in output, output

        rows = _psql(
            server,
            database,
            "-t",
            "-A",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "SELECT count(*) FROM llm_cost_aggregates",
        )
        assert _psql_stdout(rows, "llm_cost_aggregates row count") == "2", rows.stdout
    finally:
        _drop_database(server, database)


def test_delegation_routing_overlay_reconciliation_enforces_declared_shape(
    server: Server,
) -> None:
    """OMN-15631: drifted overlay rows converge before constraints are enforced."""
    path = _migration_path(
        "node_delegation_routing_reducer",
        "0001_create_delegation_routing_tenant_overlay.sql",
    )
    database = _new_database(server)
    try:
        seeded = _psql(
            server,
            database,
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            """
            CREATE TABLE delegation_routing_tenant_overlay (
                tenant_id TEXT,
                task_type TEXT
            );
            INSERT INTO delegation_routing_tenant_overlay (tenant_id, task_type)
            VALUES
                (NULL, NULL),
                ('tenant-a', 'summarize'),
                ('tenant-a', 'summarize');
            """,
        )
        assert seeded.returncode == 0, seeded.stderr

        result = _psql(server, database, "-v", "ON_ERROR_STOP=1", "-f", str(path))
        assert result.returncode == 0, result.stdout + result.stderr

        shape = _psql(
            server,
            database,
            "-t",
            "-A",
            "-F",
            "|",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            """
            SELECT
                count(*) FILTER (WHERE id IS NULL),
                count(*) FILTER (WHERE tenant_id IS NULL),
                count(*) FILTER (WHERE task_type IS NULL),
                count(*) FILTER (WHERE backend_id IS NULL),
                count(*) FILTER (WHERE endpoint_url IS NULL),
                count(*) FILTER (WHERE model_name IS NULL),
                count(*) FILTER (WHERE created_at IS NULL),
                count(*) FILTER (WHERE updated_at IS NULL),
                count(*) - count(DISTINCT (tenant_id, task_type))
            FROM delegation_routing_tenant_overlay;
            """,
        )
        assert (
            _psql_stdout(shape, "delegation_routing_tenant_overlay shape probe")
            == "0|0|0|0|0|0|0|0|0"
        ), shape.stdout

        constraints = _psql(
            server,
            database,
            "-t",
            "-A",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            """
            SELECT string_agg(conname || ':' || contype, ',' ORDER BY conname)
            FROM pg_constraint
            WHERE conrelid = 'delegation_routing_tenant_overlay'::regclass
              AND conname IN (
                'delegation_routing_tenant_overlay_pkey',
                'delegation_routing_tenant_overlay_tenant_task_uq'
              );
            """,
        )
        assert _psql_stdout(
            constraints, "delegation_routing_tenant_overlay constraint probe"
        ) == (
            "delegation_routing_tenant_overlay_pkey:p,"
            "delegation_routing_tenant_overlay_tenant_task_uq:u"
        ), constraints.stdout
    finally:
        _drop_database(server, database)


def test_whole_corpus_converges_from_drifted_shapes(server: Server) -> None:
    """The load-bearing claim: fresh and drifted end at the SAME schema."""
    corpus = _corpus()
    seeds = _drift_seed_statements()
    assert len(corpus) >= 60, len(corpus)
    assert len(seeds) >= 40, len(seeds)

    fresh_db = _new_database(server)
    drift_db = _new_database(server)
    try:
        fresh_failures = _apply_corpus(server, fresh_db)
        assert not fresh_failures, fresh_failures

        for statement in seeds:
            seeded = _psql(server, drift_db, "-v", "ON_ERROR_STOP=1", "-c", statement)
            assert seeded.returncode == 0, f"{statement}\n{seeded.stderr}"
        drift_failures = _apply_corpus(server, drift_db)
        assert not drift_failures, drift_failures

        fresh_schema = _schema_snapshot(server, fresh_db)
        drift_schema = _schema_snapshot(server, drift_db)
        only_fresh = [row for row in fresh_schema if row not in set(drift_schema)]
        only_drift = [row for row in drift_schema if row not in set(fresh_schema)]
        assert not only_fresh and not only_drift, (
            f"fresh-only={only_fresh[:20]}\ndrift-only={only_drift[:20]}"
        )
    finally:
        _drop_database(server, fresh_db)
        _drop_database(server, drift_db)


def test_fresh_path_schema_is_unchanged_by_the_reconciliation(server: Server) -> None:
    """The reconciliation is a strict no-op on an empty database.

    This is the claim that keeps the blast radius honest: 46 vendored migrations
    gained ~5k lines of guarded DDL, and every one of those lines must be
    inert when the table is being created fresh. Proven by applying the corpus
    twice on two empty databases -- once with the BEGIN/END reconciliation
    regions stripped out, once as committed -- and asserting the resulting
    schemas are identical. Nothing downstream of a projection table (golden
    chains, handler column expectations, dashboard reads) can have moved if
    this holds.
    """
    corpus = _corpus()
    stripped_dir = Path(tempfile.mkdtemp(prefix="omn15376-nofix-"))
    stripped: list[tuple[str, Path]] = []
    stripped_count = 0
    for migration_id, path in corpus:
        original = path.read_text(encoding="utf-8")
        target = stripped_dir / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if RECONCILIATION_BEGIN in original:
            target.write_text(_strip_reconciliation(original), encoding="utf-8")
            stripped_count += 1
        else:
            target.write_text(original, encoding="utf-8")
        stripped.append((migration_id, target))
    assert stripped_count >= 40, stripped_count

    with_fix_db = _new_database(server)
    without_fix_db = _new_database(server)
    try:
        assert not _apply_corpus(server, with_fix_db)
        assert not _apply_corpus_files(server, without_fix_db, stripped)
        assert _schema_snapshot(server, with_fix_db) == _schema_snapshot(
            server, without_fix_db
        )
    finally:
        _drop_database(server, with_fix_db)
        _drop_database(server, without_fix_db)
        shutil.rmtree(stripped_dir, ignore_errors=True)
