# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17886 AC2: the forward migration that revokes omninode_runtime's grants
on the ``omninode_internal.gateway_link_health_status`` view.

``nodes/node_gateway_link_health_write_effect/0002_revoke_omninode_runtime_gateway_link_health_status.sql``
is applied with the shipped ``scripts/run-forward-migrations.sh``, connecting as
``role_omnidash``: a NOSUPERUSER login that owns ``omnidash_analytics``, the
identity the k8s lanes' node loop runs as. The live-ACL gate only ever applies
it as ``postgres``, a superuser, for which a REVOKE always takes; the case that
matters on a managed lane is the one where it does not.

What differs from the lanes, stated so nobody reads more into a green:

* The runner is given a scoped corpus: the real ``_ledger/bootstrap.sql``, the
  real fence and force-RLS manifests, the real ``application-migrations.tsv``
  row for this file and nothing else, and this file byte for byte. The table and
  view are created by the test in 0001's shape, not by 0001.
* The compose runner uses one identity for both databases, so ``role_omnidash``
  also owns the fixture's ``omnibase_infra``, which it does not on any lane.

Where the Postgres comes from:

* ``OMN17886_GATE_DB_URL`` (a superuser URL) names a running server, as in the
  live-ACL gate, which runs after this module on the same server in CI. The test
  creates the two databases and three roles below and drops them afterwards, so
  it refuses a server where any of them already exists.
* Otherwise the shared ``ephemeral_postgres`` fixture starts a throwaway
  cluster: initdb/pg_ctl where they work, else a ``postgres:16-alpine``
  container. The module gates on ``EPHEMERAL_POSTGRES_UNAVAILABLE``.
* ``OMN17886_REQUIRE_PG=1`` turns "no Postgres available" into a failure, and is
  passed on to the shared fixture as ``ONEX_MIGRATION_PROOF_REQUIRE_PG=1``.

Where it runs: the module is marked ``slow`` (each case builds a database and
runs the forward runner, about two seconds), so the PR test splits, which select
``not slow``, deselect it rather than collect it as a skip on runners that have
no Postgres; ``config/skip_count_baseline.yaml`` is shrink-only (OMN-19677). The
migration-integration job runs it by path, with ``OMN17886_REQUIRE_PG=1``,
against its own empty server, so it cannot skip there.
"""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from tests.integration.migrations.conftest import EPHEMERAL_POSTGRES_UNAVAILABLE

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# ruff: noqa: S608 -- every interpolated name is a literal defined in this file;
# no query here carries untrusted input.

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
NODE = "node_gateway_link_health_write_effect"
FILENAME = "0002_revoke_omninode_runtime_gateway_link_health_status.sql"
ARTIFACT = f"nodes/{NODE}/{FILENAME}"
VERSION = f"node:{NODE}:{FILENAME}"
MIGRATION = FORWARD / ARTIFACT

INFRA_DB = "omnibase_infra"
NODE_DB = "omnidash_analytics"
RUNNER_ROLE = "role_omnidash"
RUNTIME_ROLE = "omninode_runtime"
FOREIGN_OWNER = "omn17886_foreign_owner"
VIEW = "omninode_internal.gateway_link_health_status"
_REQUIRE_PG = os.environ.get("OMN17886_REQUIRE_PG") == "1"

_EMPTY_LEDGER_FILES = (
    "application-migration-blocks.tsv",
    "legacy-node-migrations.tsv",
    "verified-checksum-adoptions.tsv",
    "verified-divergent-adoptions.tsv",
    "verified-cross-source-adoptions.tsv",
    "verified-canonical-adoptions.tsv",
    "cloud-migration-aliases.tsv",
)
# Variables the runner reads that would change its path (a slot, a lane
# release, a role to provision). None may leak in from the caller's shell.
_RUNNER_ENV_EXACT = frozenset(
    {
        "MIGRATIONS_DIR",
        "NODE_MIGRATIONS_DIR",
        "NODE_POSTGRES_DB",
        "OMNINODE_CLOUD_HISTORY_DB",
        "FORWARD_MIGRATION_LOCK_ID",
        "MIGRATION_LOCK_WAIT_SECONDS",
    }
)


def _clean_env() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PG", "POSTGRES_", "ONEX_"))
        and not key.endswith("_PASSWORD")
        and key not in _RUNNER_ENV_EXACT
    }


@dataclass(frozen=True)
class _Server:
    host: str
    port: int
    user: str
    password: str

    def psql(
        self,
        *args: str,
        dbname: str = "postgres",
        user: str | None = None,
        password: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = _clean_env()
        env["PGPASSWORD"] = self.password if user is None else (password or "")
        return subprocess.run(
            [
                "psql",
                "-X",
                "-h",
                self.host,
                "-p",
                str(self.port),
                "-U",
                user or self.user,
                "-d",
                dbname,
                "-v",
                "ON_ERROR_STOP=1",
                *args,
            ],
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

    def query(self, sql: str, *, dbname: str = "postgres") -> str:
        result = self.psql("-At", "-c", sql, dbname=dbname)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()


def _unavailable(reason: str) -> None:
    if _REQUIRE_PG:
        raise AssertionError(f"OMN17886_REQUIRE_PG=1 but {reason}")
    pytest.skip(reason)


@pytest.fixture
def server(request: pytest.FixtureRequest) -> Iterator[_Server]:
    url = os.environ.get("OMN17886_GATE_DB_URL", "")
    if url:
        parsed = urlparse(url)
        yield _Server(
            host=parsed.hostname or "localhost",
            port=parsed.port or 5432,
            user=unquote(parsed.username or "postgres"),
            password=unquote(parsed.password or ""),
        )
        return
    if EPHEMERAL_POSTGRES_UNAVAILABLE:
        _unavailable(
            "no OMN17886_GATE_DB_URL, and no ephemeral Postgres: psql is not on "
            "PATH, or neither initdb/pg_ctl nor a reachable Docker daemon is"
        )
    if _REQUIRE_PG:
        # The shared fixture skips a container that will not start unless it is
        # told a backend is required (conftest.py, _unavailable).
        request.getfixturevalue("monkeypatch").setenv(
            "ONEX_MIGRATION_PROOF_REQUIRE_PG", "1"
        )
    pg = request.getfixturevalue("ephemeral_postgres")
    yield _Server(host=pg.socket_dir, port=pg.port, user="postgres", password="")


def _scoped_corpus(root: Path) -> Path:
    forward = root / "forward"
    ledger = forward / "_ledger"
    ledger.mkdir(parents=True)
    shutil.copy(FORWARD / "_ledger" / "bootstrap.sql", ledger / "bootstrap.sql")
    for name in _EMPTY_LEDGER_FILES:
        (ledger / name).write_text("", encoding="utf-8")
    manifest = (FORWARD / "_ledger" / "application-migrations.tsv").read_text(
        encoding="utf-8"
    )
    declared = [
        line for line in manifest.splitlines() if line.split("\t", 1)[0] == ARTIFACT
    ]
    (ledger / "application-migrations.tsv").write_text(
        "".join(f"{line}\n" for line in declared), encoding="utf-8"
    )
    for name in (
        "fenced-node-migrations.yaml",
        "grandfathered-force-rls-migrations.yaml",
    ):
        shutil.copy(FORWARD / name, forward / name)
    node_dir = forward / "nodes" / NODE
    node_dir.mkdir(parents=True)
    # A missing file is not a setup error: the cases then fail on the grants the
    # database still holds, which is the failure this test exists to show.
    if MIGRATION.is_file():
        shutil.copy(MIGRATION, node_dir / FILENAME)
    return forward


@dataclass(frozen=True)
class _Lane:
    server: _Server
    runner_password: str
    corpus: Path

    def superuser(self, sql: str) -> None:
        result = self.server.psql("-c", sql, dbname=NODE_DB)
        assert result.returncode == 0, result.stderr

    def as_runner(self, sql: str) -> None:
        result = self.server.psql(
            "-c", sql, dbname=NODE_DB, user=RUNNER_ROLE, password=self.runner_password
        )
        assert result.returncode == 0, result.stderr

    def create_view(self) -> None:
        # 0001's shape, created by whoever the session is: the table and the
        # read-surface view over it.
        self.as_runner(
            "CREATE TABLE omninode_internal.gateway_link_health"
            " (tenant_id text PRIMARY KEY, last_seen_at timestamptz);"
            " CREATE VIEW omninode_internal.gateway_link_health_status AS"
            " SELECT tenant_id, last_seen_at FROM omninode_internal.gateway_link_health;"
        )

    def run_runner(self) -> subprocess.CompletedProcess[str]:
        env = _clean_env()
        env.update(
            {
                "POSTGRES_USER": RUNNER_ROLE,
                "POSTGRES_PASSWORD": self.runner_password,
                "POSTGRES_HOST": self.server.host,
                "POSTGRES_PORT": str(self.server.port),
                "POSTGRES_DB": INFRA_DB,
                "NODE_POSTGRES_DB": NODE_DB,
                "MIGRATIONS_DIR": str(self.corpus),
                "NODE_MIGRATIONS_DIR": str(self.corpus / "nodes"),
            }
        )
        return subprocess.run(
            ["sh", str(RUNNER)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=REPO_ROOT,
            timeout=300,
        )

    def apply_file_as_runner(self) -> subprocess.CompletedProcess[str]:
        # The node loop's own apply command, in both runners.
        return self.server.psql(
            "-f",
            str(self.corpus / "nodes" / NODE / FILENAME),
            dbname=NODE_DB,
            user=RUNNER_ROLE,
            password=self.runner_password,
        )

    def runtime_privileges(self) -> set[str]:
        # From the view's own ACL: information_schema would hide a grant that
        # does not involve the session's roles.
        rows = self.server.query(
            "SELECT a.privilege_type FROM pg_catalog.pg_class c"
            " JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace,"
            " aclexplode(c.relacl) a"
            " WHERE n.nspname = 'omninode_internal'"
            " AND c.relname = 'gateway_link_health_status'"
            f" AND a.grantee = '{RUNTIME_ROLE}'::regrole",
            dbname=NODE_DB,
        )
        return set(rows.split())

    def ledger(self) -> list[str]:
        rows = self.server.query(
            "SELECT migration_stream || '|' || domain || '|' || checksum "
            "FROM platform_catalog.schema_migrations "
            f"WHERE version = '{VERSION}'",
            dbname=NODE_DB,
        )
        return rows.splitlines()


@pytest.fixture
def lane(server: _Server, tmp_path: Path) -> Iterator[_Lane]:
    names = ", ".join(f"'{name}'" for name in (INFRA_DB, NODE_DB))
    roles = ", ".join(
        f"'{name}'" for name in (RUNNER_ROLE, RUNTIME_ROLE, FOREIGN_OWNER)
    )
    present = server.query(
        f"SELECT string_agg(datname, ',') FROM pg_database WHERE datname IN ({names})"
    ) + server.query(
        f"SELECT string_agg(rolname, ',') FROM pg_roles WHERE rolname IN ({roles})"
    )
    assert not present, (
        f"refusing to build over existing {present}; point OMN17886_GATE_DB_URL at a "
        "server of its own"
    )
    password = secrets.token_hex(16)
    setup = (
        f"CREATE ROLE {RUNNER_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
        f"NOCREATEROLE NOREPLICATION PASSWORD '{password}'",
        # As flat 099 creates it.
        f"CREATE ROLE {RUNTIME_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
        "NOCREATEROLE NOREPLICATION",
        f"CREATE ROLE {FOREIGN_OWNER} NOLOGIN",
        f"CREATE DATABASE {INFRA_DB} OWNER {RUNNER_ROLE}",
        f"CREATE DATABASE {NODE_DB} OWNER {RUNNER_ROLE}",
    )
    try:
        for statement in setup:
            server.query(statement)
        # The sentinel table the runner clears and sets, created by the compose
        # bootstrap on a real lane.
        for database in (INFRA_DB, NODE_DB):
            result = server.psql(
                "-c",
                "CREATE TABLE public.db_metadata ("
                " id BOOLEAN PRIMARY KEY DEFAULT TRUE,"
                " migrations_complete BOOLEAN NOT NULL DEFAULT FALSE,"
                " runner_completed_at TIMESTAMPTZ,"
                " updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());"
                " INSERT INTO public.db_metadata (id) VALUES (TRUE);",
                dbname=database,
                user=RUNNER_ROLE,
                password=password,
            )
            assert result.returncode == 0, result.stderr
        result = server.psql(
            "-c",
            f"CREATE SCHEMA omninode_internal AUTHORIZATION {RUNNER_ROLE}",
            dbname=NODE_DB,
        )
        assert result.returncode == 0, result.stderr
        # Positive control: the identity under test cannot bypass ownership.
        assert (
            server.query(
                f"SELECT rolsuper FROM pg_roles WHERE rolname = '{RUNNER_ROLE}'"
            )
            == "f"
        )
        yield _Lane(
            server=server, runner_password=password, corpus=_scoped_corpus(tmp_path)
        )
    finally:
        for database in (NODE_DB, INFRA_DB):
            server.psql("-c", f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
        for role in (RUNNER_ROLE, RUNTIME_ROLE, FOREIGN_OWNER):
            dropped = server.psql("-c", f"DROP ROLE IF EXISTS {role}")
            assert dropped.returncode == 0, dropped.stderr


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _expected_ledger_row() -> str:
    checksum = hashlib.sha256(MIGRATION.read_bytes()).hexdigest()
    return f"node:{NODE}|omninode_internal|{checksum}"


def _post_condition() -> str:
    executable = "\n".join(
        line.split("--", 1)[0]
        for line in MIGRATION.read_text(encoding="utf-8").splitlines()
    )
    statements = [s.strip() for s in executable.split(";") if s.strip()]
    matches = [
        s
        for s in statements
        if "gateway_link_health_status_runtime_revoked_assertion" in s
    ]
    assert len(matches) == 1, statements
    return matches[0] + ";"


def test_a_default_rule_grants_on_a_runner_owned_view_are_revoked(
    lane: _Lane,
) -> None:
    # The from-empty shape on a k8s lane: the runner creates the view, and a
    # default-privilege rule for the runner's objects confers the grants, as
    # flat 099's rule does for its owner.
    lane.as_runner(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal "
        f"GRANT SELECT, INSERT, UPDATE ON TABLES TO {RUNTIME_ROLE}"
    )
    lane.create_view()
    assert lane.runtime_privileges() == {"SELECT", "INSERT", "UPDATE"}

    result = lane.run_runner()

    assert result.returncode == 0, _output(result)
    assert lane.runtime_privileges() == set()
    # The owner keeps its own access: the file revokes from one grantee only.
    assert (
        lane.server.query(
            f"SELECT has_table_privilege('{RUNNER_ROLE}', '{VIEW}', 'SELECT')",
            dbname=NODE_DB,
        )
        == "t"
    )
    assert lane.ledger() == [_expected_ledger_row()]


def test_b_view_with_no_runtime_grant_is_a_no_op(lane: _Lane) -> None:
    # The shape of a lane with no runtime default rule.
    lane.create_view()
    assert lane.runtime_privileges() == set()

    result = lane.run_runner()

    assert result.returncode == 0, _output(result)
    assert lane.runtime_privileges() == set()
    # An empty set alone would also follow from the file never running.
    assert lane.ledger() == [_expected_ledger_row()]


def test_c_grant_the_runner_cannot_revoke_stops_the_loop_unledgered(
    lane: _Lane,
) -> None:
    # The view and the runtime grants belong to a role the runner is not, and
    # the runner holds only SELECT on it. Its REVOKE removes nothing and only
    # warns; the post-condition must stop the file so the loop does not ledger
    # a revoke that did not happen.
    lane.superuser(
        "CREATE TABLE omninode_internal.gateway_link_health"
        " (tenant_id text PRIMARY KEY, last_seen_at timestamptz);"
        " CREATE VIEW omninode_internal.gateway_link_health_status AS"
        " SELECT tenant_id, last_seen_at FROM omninode_internal.gateway_link_health;"
        f" ALTER TABLE omninode_internal.gateway_link_health OWNER TO {FOREIGN_OWNER};"
        f" ALTER VIEW omninode_internal.gateway_link_health_status OWNER TO {FOREIGN_OWNER};"
        f" GRANT USAGE ON SCHEMA omninode_internal TO {FOREIGN_OWNER};"
        f" SET ROLE {FOREIGN_OWNER};"
        f" GRANT SELECT, INSERT, UPDATE ON {VIEW} TO {RUNTIME_ROLE};"
        f" GRANT SELECT ON {VIEW} TO {RUNNER_ROLE};"
        " RESET ROLE;"
    )
    assert lane.runtime_privileges() == {"SELECT", "INSERT", "UPDATE"}

    result = lane.run_runner()

    assert result.returncode != 0, _output(result)
    assert "no privileges could be revoked" in _output(result)
    assert "division by zero" in _output(result)
    assert lane.runtime_privileges() == {"SELECT", "INSERT", "UPDATE"}
    assert lane.ledger() == []


def test_d_rerun_after_the_revoke_is_a_no_op(lane: _Lane) -> None:
    lane.as_runner(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal "
        f"GRANT SELECT, INSERT, UPDATE ON TABLES TO {RUNTIME_ROLE}"
    )
    lane.create_view()
    first = lane.run_runner()
    assert first.returncode == 0, _output(first)
    assert lane.runtime_privileges() == set()

    second = lane.run_runner()

    assert second.returncode == 0, _output(second)
    assert f"skip  {VERSION} (already applied)" in second.stdout
    # The SQL itself, not only the ledger, is safe to run again: a lane whose
    # ledger does not carry this id (the k8s Job keeps its own) re-applies it.
    again = lane.apply_file_as_runner()
    assert again.returncode == 0, _output(again)
    assert lane.runtime_privileges() == set()
    assert lane.ledger() == [_expected_ledger_row()]


def test_the_post_condition_refuses_a_remaining_runtime_grant(lane: _Lane) -> None:
    check = _post_condition()
    lane.create_view()
    lane.as_runner(f"GRANT SELECT ON {VIEW} TO {RUNTIME_ROLE}")

    present = lane.server.psql("-At", "-c", check, dbname=NODE_DB)

    assert present.returncode != 0, present.stdout
    assert "division by zero" in present.stderr

    lane.as_runner(f"REVOKE SELECT ON {VIEW} FROM {RUNTIME_ROLE}")
    absent = lane.server.psql("-At", "-c", check, dbname=NODE_DB)
    assert absent.returncode == 0, absent.stderr
    assert absent.stdout.strip() == "1"
