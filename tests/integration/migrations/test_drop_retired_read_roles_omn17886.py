# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17886 AC2 step 3: removing the hand-made ``jake_ro`` and ``lakshman_ro``.

Two forward migrations do it, applied here by the shipped
``scripts/run-forward-migrations.sh`` exactly as a lane applies them:

* flat ``108_drop_owned_by_retired_read_roles.sql`` runs first, in
  ``omnibase_infra``, and refuses before dropping anything when either role
  still depends on an object in a database neither loop reaches;
* ``nodes/node_projection_live_events/0003_drop_retired_read_roles.sql`` runs in
  ``omnidash_analytics``, clears it, and drops both roles once ``pg_shdepend``
  records nothing left.

Two runner identities are exercised, because the lanes differ:

* ``postgres``, a superuser, is what the compose lanes run as;
* ``role_omnidash``, NOSUPERUSER and NOCREATEROLE, owns both databases here and
  stands for the managed lane's migrate Job. Flats 104 and 107 were retired
  because they refused on that lane (``_ledger/retired-flat-migrations.tsv``),
  so the case that must hold there is the one where the roles are absent.

What differs from the lanes, stated so nobody reads more into a green:

* The runner is given a scoped corpus: the real ``_ledger/bootstrap.sql``, the
  real fence and force-RLS manifests, the real ``application-migrations.tsv``
  row for the node file and nothing else, and the two files byte for byte.
* The roles' grants are made by the test in the shapes the .201 readback of
  2026-09-25 found (table SELECT, database CONNECT, a default-privilege entry),
  not copied from a lane.
* The compose runner uses one identity for both databases; on the managed lane
  the flat and node loops run as different roles.

Where the Postgres comes from, and where it runs, as in the other OMN-17886
modules: ``OMN17886_GATE_DB_URL`` or the shared ``ephemeral_postgres`` fixture;
``OMN17886_REQUIRE_PG=1`` makes a missing server a failure; the module is
marked ``slow``, so the PR test splits deselect it rather than collect it as a
skip (``config/skip_count_baseline.yaml`` is shrink-only, OMN-19677), and the
migration-integration job runs it by path.
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
FLAT = "108_drop_owned_by_retired_read_roles.sql"
NODE = "node_projection_live_events"
NODE_FILE = "0003_drop_retired_read_roles.sql"
NODE_ARTIFACT = f"nodes/{NODE}/{NODE_FILE}"
NODE_VERSION = f"node:{NODE}:{NODE_FILE}"
FLAT_MIGRATION = FORWARD / FLAT
NODE_MIGRATION = FORWARD / NODE_ARTIFACT

INFRA_DB = "omnibase_infra"
NODE_DB = "omnidash_analytics"
ELSEWHERE_DB = "omn17886_elsewhere"
MANAGED_RUNNER = "role_omnidash"
RUNTIME_ROLE = "omninode_runtime"
RETIRED = ("jake_ro", "lakshman_ro")
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

    def run(self, sql: str, *, dbname: str) -> None:
        result = self.psql("-c", sql, dbname=dbname)
        assert result.returncode == 0, result.stderr


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
        line
        for line in manifest.splitlines()
        if line.split("\t", 1)[0] == NODE_ARTIFACT
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
    # A missing file is not a setup error: the cases then fail on the roles
    # the cluster still holds, which is the failure this test exists to show.
    if FLAT_MIGRATION.is_file():
        shutil.copy(FLAT_MIGRATION, forward / FLAT)
    if NODE_MIGRATION.is_file():
        shutil.copy(NODE_MIGRATION, node_dir / NODE_FILE)
    return forward


@dataclass(frozen=True)
class _Cluster:
    server: _Server
    managed_password: str
    corpus: Path

    def run_runner(self, *, managed: bool) -> subprocess.CompletedProcess[str]:
        env = _clean_env()
        env.update(
            {
                "POSTGRES_USER": MANAGED_RUNNER if managed else self.server.user,
                "POSTGRES_PASSWORD": (
                    self.managed_password if managed else self.server.password
                ),
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

    def give_the_retired_roles_their_readback_shape(self) -> None:
        # The shapes the .201 readback of 2026-09-25 found: table SELECT and a
        # default-privilege entry in both databases, and database CONNECT. The
        # runtime role gets the same kinds of grant beside them, so the cases
        # can show the removal leaves it alone.
        for role in RETIRED:
            self.server.run(
                f"CREATE ROLE {role} LOGIN PASSWORD NULL", dbname="postgres"
            )
        self.server.run(
            "CREATE TABLE public.infra_probe (id int);"
            " GRANT SELECT ON public.infra_probe TO jake_ro, lakshman_ro;"
            f" GRANT CONNECT ON DATABASE {INFRA_DB} TO jake_ro, lakshman_ro;"
            " ALTER DEFAULT PRIVILEGES IN SCHEMA public"
            " GRANT SELECT ON TABLES TO jake_ro;",
            dbname=INFRA_DB,
        )
        self.server.run(
            "CREATE TABLE omninode_internal.node_probe (id int);"
            " GRANT SELECT ON omninode_internal.node_probe TO jake_ro;"
            f" GRANT SELECT, INSERT, UPDATE ON omninode_internal.node_probe TO {RUNTIME_ROLE};"
            f" GRANT CONNECT ON DATABASE {NODE_DB} TO jake_ro, {RUNTIME_ROLE};"
            " ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal"
            f" GRANT SELECT ON TABLES TO jake_ro, {RUNTIME_ROLE};",
            dbname=NODE_DB,
        )

    def retired_present(self) -> set[str]:
        rows = self.server.query(
            "SELECT rolname FROM pg_catalog.pg_roles "
            "WHERE rolname IN ('jake_ro', 'lakshman_ro')"
        )
        return set(rows.split())

    def infra_grantees(self) -> set[str]:
        rows = self.server.query(
            "SELECT pg_get_userbyid(a.grantee) FROM pg_catalog.pg_class c,"
            " aclexplode(c.relacl) a WHERE c.oid = 'public.infra_probe'::regclass",
            dbname=INFRA_DB,
        )
        return set(rows.split())

    def runtime_state(self) -> tuple[set[str], bool, bool]:
        table = self.server.query(
            "SELECT a.privilege_type FROM pg_catalog.pg_class c,"
            " aclexplode(c.relacl) a"
            " WHERE c.oid = 'omninode_internal.node_probe'::regclass"
            f" AND a.grantee = '{RUNTIME_ROLE}'::regrole",
            dbname=NODE_DB,
        )
        default_rule = self.server.query(
            "SELECT count(*) > 0 FROM pg_catalog.pg_default_acl d,"
            " aclexplode(d.defaclacl) a"
            f" WHERE a.grantee = '{RUNTIME_ROLE}'::regrole",
            dbname=NODE_DB,
        )
        connect = self.server.query(
            f"SELECT has_database_privilege('{RUNTIME_ROLE}', '{NODE_DB}', 'CONNECT')"
        )
        return set(table.split()), default_rule == "t", connect == "t"

    def flat_ledger(self) -> list[str]:
        rows = self.server.query(
            "SELECT migration_id FROM public.schema_migrations "
            f"WHERE migration_id = 'docker/{FLAT}'",
            dbname=INFRA_DB,
        )
        return rows.splitlines()

    def node_ledger(self) -> list[str]:
        exists = self.server.query(
            "SELECT to_regclass('platform_catalog.schema_migrations') IS NOT NULL",
            dbname=NODE_DB,
        )
        if exists != "t":
            return []
        rows = self.server.query(
            "SELECT checksum FROM platform_catalog.schema_migrations "
            f"WHERE version = '{NODE_VERSION}'",
            dbname=NODE_DB,
        )
        return rows.splitlines()


@pytest.fixture
def cluster(server: _Server, tmp_path: Path) -> Iterator[_Cluster]:
    databases = (INFRA_DB, NODE_DB, ELSEWHERE_DB)
    roles = (MANAGED_RUNNER, RUNTIME_ROLE, *RETIRED)
    names = ", ".join(f"'{name}'" for name in databases)
    role_names = ", ".join(f"'{name}'" for name in roles)
    present = server.query(
        f"SELECT string_agg(datname, ',') FROM pg_database WHERE datname IN ({names})"
    ) + server.query(
        f"SELECT string_agg(rolname, ',') FROM pg_roles WHERE rolname IN ({role_names})"
    )
    assert not present, (
        f"refusing to build over existing {present}; point OMN17886_GATE_DB_URL at a "
        "server of its own"
    )
    password = secrets.token_hex(16)
    try:
        server.query(
            f"CREATE ROLE {MANAGED_RUNNER} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
            f"NOCREATEROLE NOREPLICATION PASSWORD '{password}'"
        )
        server.query(
            f"CREATE ROLE {RUNTIME_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
            "NOCREATEROLE NOREPLICATION"
        )
        for database in (INFRA_DB, NODE_DB):
            server.query(f"CREATE DATABASE {database} OWNER {MANAGED_RUNNER}")
            # The sentinel table the runner clears and sets, created by the
            # compose bootstrap on a real lane.
            server.run(
                "CREATE TABLE public.db_metadata ("
                " id BOOLEAN PRIMARY KEY DEFAULT TRUE,"
                " migrations_complete BOOLEAN NOT NULL DEFAULT FALSE,"
                " runner_completed_at TIMESTAMPTZ,"
                " updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW());"
                " INSERT INTO public.db_metadata (id) VALUES (TRUE);"
                f" ALTER TABLE public.db_metadata OWNER TO {MANAGED_RUNNER};",
                dbname=database,
            )
        server.run(
            f"CREATE SCHEMA omninode_internal AUTHORIZATION {MANAGED_RUNNER}",
            dbname=NODE_DB,
        )
        # Positive control: the managed identity cannot bypass anything.
        assert (
            server.query(
                "SELECT rolsuper::text || rolcreaterole::text FROM pg_roles "
                f"WHERE rolname = '{MANAGED_RUNNER}'"
            )
            == "falsefalse"
        )
        yield _Cluster(
            server=server, managed_password=password, corpus=_scoped_corpus(tmp_path)
        )
    finally:
        for database in (NODE_DB, INFRA_DB, ELSEWHERE_DB):
            server.psql("-c", f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
        for role in (*RETIRED, RUNTIME_ROLE, MANAGED_RUNNER):
            dropped = server.psql("-c", f"DROP ROLE IF EXISTS {role}")
            assert dropped.returncode == 0, dropped.stderr


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _post_condition() -> str:
    executable = "\n".join(
        line.split("--", 1)[0]
        for line in NODE_MIGRATION.read_text(encoding="utf-8").splitlines()
    )
    statements = [s.strip() for s in executable.split(";") if s.strip()]
    matches = [s for s in statements if "retired_read_roles_absent_assertion" in s]
    assert len(matches) == 1, statements
    return matches[0] + ";"


def test_a_compose_runner_removes_both_roles_and_leaves_the_runtime_alone(
    cluster: _Cluster,
) -> None:
    cluster.give_the_retired_roles_their_readback_shape()
    before = cluster.runtime_state()
    assert cluster.retired_present() == set(RETIRED)
    assert before == ({"SELECT", "INSERT", "UPDATE"}, True, True)

    result = cluster.run_runner(managed=False)

    assert result.returncode == 0, _output(result)
    assert cluster.retired_present() == set()
    assert cluster.runtime_state() == before
    # The flat ledger records the id; its checksum column is the runner's
    # literal "applied-by-runner", not a digest.
    assert cluster.flat_ledger() == [f"docker/{FLAT}"]
    assert cluster.node_ledger() == [_checksum(NODE_MIGRATION)]


def test_b_absent_roles_are_a_no_op_for_the_managed_identity(cluster: _Cluster) -> None:
    # The onex-dev case the merge waits on: neither role exists, and the
    # NOCREATEROLE, non-member runner must pass straight through.
    assert cluster.retired_present() == set()

    result = cluster.run_runner(managed=True)

    assert result.returncode == 0, _output(result)
    # The flat ledger records the id; its checksum column is the runner's
    # literal "applied-by-runner", not a digest.
    assert cluster.flat_ledger() == [f"docker/{FLAT}"]
    assert cluster.node_ledger() == [_checksum(NODE_MIGRATION)]


def test_c_a_dependency_elsewhere_stops_before_anything_is_dropped(
    cluster: _Cluster,
) -> None:
    cluster.give_the_retired_roles_their_readback_shape()
    cluster.server.query(f"CREATE DATABASE {ELSEWHERE_DB}")
    cluster.server.run(
        "CREATE TABLE public.elsewhere_probe (id int);"
        " GRANT SELECT ON public.elsewhere_probe TO jake_ro;",
        dbname=ELSEWHERE_DB,
    )

    result = cluster.run_runner(managed=False)

    assert result.returncode != 0, _output(result)
    assert f"database(s) {ELSEWHERE_DB}" in _output(result)
    assert cluster.retired_present() == set(RETIRED)
    # Nothing half-removed: the omnibase_infra grants the flat file would
    # have dropped first are all still there.
    assert {"jake_ro", "lakshman_ro"} <= cluster.infra_grantees()
    assert cluster.flat_ledger() == []
    assert cluster.node_ledger() == []


def test_d_present_roles_stop_the_managed_identity_unledgered(
    cluster: _Cluster,
) -> None:
    cluster.give_the_retired_roles_their_readback_shape()

    result = cluster.run_runner(managed=True)

    assert result.returncode != 0, _output(result)
    assert "cannot DROP OWNED BY jake_ro" in _output(result)
    assert cluster.retired_present() == set(RETIRED)
    assert {"jake_ro", "lakshman_ro"} <= cluster.infra_grantees()
    assert cluster.flat_ledger() == []


def test_e_rerun_after_the_removal_is_a_no_op(cluster: _Cluster) -> None:
    cluster.give_the_retired_roles_their_readback_shape()
    first = cluster.run_runner(managed=False)
    assert first.returncode == 0, _output(first)
    assert cluster.retired_present() == set()

    second = cluster.run_runner(managed=False)

    assert second.returncode == 0, _output(second)
    assert f"skip  {NODE_VERSION} (already applied)" in second.stdout
    # The SQL itself, not only the ledgers, is safe to run again: a lane whose
    # ledger does not carry these ids re-applies them.
    for path, database in ((FLAT_MIGRATION, INFRA_DB), (NODE_MIGRATION, NODE_DB)):
        again = cluster.server.psql("-f", str(path), dbname=database)
        assert again.returncode == 0, _output(again)
    assert cluster.retired_present() == set()


def test_the_post_condition_refuses_a_present_role(cluster: _Cluster) -> None:
    check = _post_condition()
    cluster.server.query("CREATE ROLE lakshman_ro NOLOGIN")

    present = cluster.server.psql("-At", "-c", check)

    assert present.returncode != 0, present.stdout
    assert "division by zero" in present.stderr

    cluster.server.query("DROP ROLE lakshman_ro")
    absent = cluster.server.psql("-At", "-c", check)
    assert absent.returncode == 0, absent.stderr
    assert absent.stdout.strip() == "1"
