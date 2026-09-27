# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17887: the forward migration that drops onex-lab's empty ``tenant`` schema.

The retire of the ``tenant`` schema is finished in code, but onex-lab still
carries the empty schema its bootstrap created at initdb, and that volume is
persistent. Jonah's ruling is to drop it by forward migration:
``nodes/node_projection_tenant_credentials/004_drop_empty_tenant_schema.sql``.

Each case builds a fresh ``omnidash_analytics`` shaped like onex-lab and applies
the migration with the shipped ``scripts/run-forward-migrations.sh``, connecting
as ``role_omnidash``: a NOSUPERUSER login that owns ``omnidash_analytics`` and is
a member of ``owner_onex_tenant``, as the lab bootstrap makes it.

What differs from the lanes, stated so nobody reads more into a green:

* The runner is given a scoped corpus: the real ``_ledger/bootstrap.sql``, the
  real fence and force-RLS manifests, the real ``application-migrations.tsv``
  row for this file and nothing else, and this file byte for byte. The other
  node migrations and the flat corpus are not applied.
* The compose lanes run this script as ``postgres``, a superuser, for which the
  ownership refusal in case (d) cannot happen. The k8s lanes (onex-lab,
  onex-dev) run omninode_infra's migrate Job as ``role_omnidash`` instead; its
  node loop applies each file with the same ``psql -U <role> -d
  omnidash_analytics -v ON_ERROR_STOP=1 -f <file>`` and ledgers it in
  ``node_schema_migrations`` only after psql exits 0. That Job is not run here.
* The compose runner uses one identity for both databases, so ``role_omnidash``
  also owns the fixture's ``omnibase_infra``, which it does not on any lane.

Where the Postgres comes from:

* ``OMN17887_GATE_DB_URL`` (a superuser URL) names a running server, as in the
  OMN-17886 gate. The test creates the two databases and four roles below and
  drops them afterwards, so it refuses a server where any of them already
  exists.
* Otherwise the shared ``ephemeral_postgres`` fixture starts a throwaway
  cluster: initdb/pg_ctl where they work, else a ``postgres:16-alpine``
  container. The module gates on ``EPHEMERAL_POSTGRES_UNAVAILABLE`` (psql
  missing, or neither initdb/pg_ctl nor a reachable Docker daemon), not on the
  native tools alone.
* ``OMN17887_REQUIRE_PG=1`` turns "no Postgres available" into a failure. It is
  passed on to the shared fixture as ``ONEX_MIGRATION_PROOF_REQUIRE_PG=1``, so a
  container that will not start fails too instead of skipping.
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

# The heavy marker deselects this module from ci.yml's Tests (Split n/m),
# whose filter is not heavy, so it never counts as a split skip against the
# shrink-only config/skip_count_baseline.yaml (OMN-19677). It executes in
# ci.yml's migration-integration job by explicit path with OMN17887_REQUIRE_PG=1,
# so a missing server fails there instead of skipping.
pytestmark = [pytest.mark.integration, pytest.mark.heavy]

# ruff: noqa: S608 -- every interpolated name is a literal defined in this file;
# no query here carries untrusted input.

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
FORWARD = REPO_ROOT / "docker" / "migrations" / "forward"
NODE = "node_projection_tenant_credentials"
FILENAME = "004_drop_empty_tenant_schema.sql"
ARTIFACT = f"nodes/{NODE}/{FILENAME}"
VERSION = f"node:{NODE}:{FILENAME}"
MIGRATION = FORWARD / ARTIFACT

INFRA_DB = "omnibase_infra"
NODE_DB = "omnidash_analytics"
RUNNER_ROLE = "role_omnidash"
TENANT_OWNER = "owner_onex_tenant"
UNRELATED_OWNER = "omn17887_unrelated_owner"
# The grantee of the one conditional grant still left on `tenant`:
# node_projection_delegation_inference_response/0004 runs
# `GRANT USAGE ON SCHEMA tenant TO tenant_projection_writer` when it exists.
WRITER_ROLE = "tenant_projection_writer"
_REQUIRE_PG = os.environ.get("OMN17887_REQUIRE_PG") == "1"

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
        raise AssertionError(f"OMN17887_REQUIRE_PG=1 but {reason}")
    pytest.skip(reason)


@pytest.fixture
def server(request: pytest.FixtureRequest) -> Iterator[_Server]:
    url = os.environ.get("OMN17887_GATE_DB_URL", "")
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
            "no OMN17887_GATE_DB_URL, and no ephemeral Postgres: psql is not on "
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
    # A missing file is not a setup error: every case then fails on what the
    # database still holds, which is the failure this test exists to show.
    if MIGRATION.is_file():
        shutil.copy(MIGRATION, node_dir / FILENAME)
    return forward


@dataclass(frozen=True)
class _Lab:
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

    def tenant(self) -> tuple[str, int] | None:
        row = self.server.query(
            "SELECT pg_get_userbyid(n.nspowner) || '|' || "
            "(SELECT count(*) FROM pg_catalog.pg_class c WHERE c.relnamespace = n.oid) "
            "FROM pg_catalog.pg_namespace n WHERE n.nspname = 'tenant'",
            dbname=NODE_DB,
        )
        if not row:
            return None
        owner, relations = row.split("|")
        return owner, int(relations)

    def tenant_acl(self) -> list[str]:
        acl = self.server.query(
            "SELECT coalesce(array_to_string(nspacl, ','), '') "
            "FROM pg_catalog.pg_namespace WHERE nspname = 'tenant'",
            dbname=NODE_DB,
        )
        return acl.split(",") if acl else []

    def schema_scoped_default_acls(self) -> int:
        # Rows keep their schema's oid, so a row the drop left behind still
        # counts here after `tenant` is gone.
        return int(
            self.server.query(
                "SELECT count(*) FROM pg_catalog.pg_default_acl "
                f"WHERE defaclrole = '{TENANT_OWNER}'::regrole "
                "AND defaclnamespace <> 0",
                dbname=NODE_DB,
            )
        )

    def ledger(self) -> list[str]:
        rows = self.server.query(
            "SELECT migration_stream || '|' || domain || '|' || checksum "
            "FROM platform_catalog.schema_migrations "
            f"WHERE version = '{VERSION}'",
            dbname=NODE_DB,
        )
        return rows.splitlines()


@pytest.fixture
def lab(server: _Server, tmp_path: Path) -> Iterator[_Lab]:
    names = ", ".join(f"'{name}'" for name in (INFRA_DB, NODE_DB))
    roles = ", ".join(
        f"'{name}'"
        for name in (RUNNER_ROLE, TENANT_OWNER, UNRELATED_OWNER, WRITER_ROLE)
    )
    present = server.query(
        f"SELECT string_agg(datname, ',') FROM pg_database WHERE datname IN ({names})"
    ) + server.query(
        f"SELECT string_agg(rolname, ',') FROM pg_roles WHERE rolname IN ({roles})"
    )
    assert not present, (
        f"refusing to build over existing {present}; point OMN17887_GATE_DB_URL at a "
        "server of its own"
    )
    password = secrets.token_hex(16)
    setup = (
        f"CREATE ROLE {RUNNER_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
        f"NOCREATEROLE NOREPLICATION PASSWORD '{password}'",
        f"CREATE ROLE {TENANT_OWNER} NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
        "NOCREATEROLE NOREPLICATION",
        f"CREATE ROLE {UNRELATED_OWNER} NOLOGIN",
        # As 0004 creates it.
        f"CREATE ROLE {WRITER_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
        "NOCREATEROLE NOREPLICATION",
        f"GRANT {TENANT_OWNER} TO {RUNNER_ROLE}",
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
        # Positive control: the identity under test cannot bypass ownership.
        assert (
            server.query(
                f"SELECT rolsuper FROM pg_roles WHERE rolname = '{RUNNER_ROLE}'"
            )
            == "f"
        )
        yield _Lab(
            server=server, runner_password=password, corpus=_scoped_corpus(tmp_path)
        )
    finally:
        for database in (NODE_DB, INFRA_DB):
            server.psql("-c", f"DROP DATABASE IF EXISTS {database} WITH (FORCE)")
        for role in (RUNNER_ROLE, TENANT_OWNER, UNRELATED_OWNER, WRITER_ROLE):
            dropped = server.psql("-c", f"DROP ROLE IF EXISTS {role}")
            assert dropped.returncode == 0, dropped.stderr


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _expected_ledger_row() -> str:
    checksum = hashlib.sha256(MIGRATION.read_bytes()).hexdigest()
    return f"node:{NODE}|tenant|{checksum}"


def _post_condition() -> str:
    executable = "\n".join(
        line.split("--", 1)[0]
        for line in MIGRATION.read_text(encoding="utf-8").splitlines()
    )
    statements = [s.strip() for s in executable.split(";") if s.strip()]
    matches = [s for s in statements if "tenant_schema_absent_assertion" in s]
    assert len(matches) == 1, statements
    return matches[0] + ";"


def test_a_absent_schema_is_a_no_op(lab: _Lab) -> None:
    assert lab.tenant() is None

    result = lab.run_runner()

    assert result.returncode == 0, _output(result)
    assert 'schema "tenant" does not exist, skipping' in _output(result)
    assert lab.tenant() is None
    # Absence alone would also follow from the file never running.
    assert lab.ledger() == [_expected_ledger_row()]


def test_b_empty_schema_owned_through_membership_is_dropped(lab: _Lab) -> None:
    # onex-lab's likely shape: the empty schema, the USAGE grant 0004 leaves on
    # it when it exists (applied by the node loop as the runner), and a
    # default-privileges entry scoped to it. Neither is an object in the
    # schema, so RESTRICT must not refuse on them, and both must go with it.
    lab.superuser(f"CREATE SCHEMA tenant AUTHORIZATION {TENANT_OWNER}")
    lab.as_runner(f"GRANT USAGE ON SCHEMA tenant TO {WRITER_ROLE}")
    lab.superuser(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {TENANT_OWNER} IN SCHEMA tenant "
        f"GRANT SELECT ON TABLES TO {WRITER_ROLE}"
    )
    assert lab.tenant() == (TENANT_OWNER, 0)
    assert f"{WRITER_ROLE}=U/{TENANT_OWNER}" in lab.tenant_acl()
    assert lab.schema_scoped_default_acls() == 1

    result = lab.run_runner()

    assert result.returncode == 0, _output(result)
    assert lab.tenant() is None, "tenant is still present after the runner ran"
    assert lab.schema_scoped_default_acls() == 0
    assert lab.ledger() == [_expected_ledger_row()]


def test_c_schema_holding_a_relation_fails_closed_and_keeps_it(lab: _Lab) -> None:
    lab.superuser(
        f"CREATE SCHEMA tenant AUTHORIZATION {TENANT_OWNER};"
        " CREATE TABLE tenant.tenant_leftover (id int);"
        " INSERT INTO tenant.tenant_leftover VALUES (7);"
    )

    result = lab.run_runner()

    assert result.returncode != 0, _output(result)
    assert "cannot drop schema tenant because other objects depend on it" in _output(
        result
    )
    assert "table tenant.tenant_leftover depends on schema tenant" in _output(result)
    assert lab.tenant() == (TENANT_OWNER, 1)
    assert (
        lab.server.query("SELECT id FROM tenant.tenant_leftover", dbname=NODE_DB) == "7"
    )
    assert lab.ledger() == []


def test_d_schema_owned_outside_the_runners_roles_fails_unledgered(lab: _Lab) -> None:
    lab.superuser(f"CREATE SCHEMA tenant AUTHORIZATION {UNRELATED_OWNER}")

    result = lab.run_runner()

    assert result.returncode != 0, _output(result)
    assert "must be owner of schema tenant" in _output(result)
    assert lab.tenant() == (UNRELATED_OWNER, 0)
    assert lab.ledger() == []


def test_e_rerun_after_the_drop_is_a_no_op(lab: _Lab) -> None:
    lab.superuser(f"CREATE SCHEMA tenant AUTHORIZATION {TENANT_OWNER}")
    first = lab.run_runner()
    assert first.returncode == 0, _output(first)
    assert lab.tenant() is None

    second = lab.run_runner()

    assert second.returncode == 0, _output(second)
    assert f"skip  {VERSION} (already applied)" in second.stdout
    # The SQL itself, not only the ledger, is safe to run again: a lane whose
    # ledger does not carry this id (the k8s Job keeps its own) re-applies it.
    again = lab.apply_file_as_runner()
    assert again.returncode == 0, _output(again)
    assert 'schema "tenant" does not exist, skipping' in _output(again)
    assert lab.tenant() is None
    assert lab.ledger() == [_expected_ledger_row()]


def test_the_post_condition_refuses_a_present_schema(lab: _Lab) -> None:
    check = _post_condition()
    lab.superuser(f"CREATE SCHEMA tenant AUTHORIZATION {TENANT_OWNER}")

    present = lab.server.psql("-At", "-c", check, dbname=NODE_DB)

    assert present.returncode != 0, present.stdout
    assert "division by zero" in present.stderr

    lab.superuser("DROP SCHEMA tenant RESTRICT")
    absent = lab.server.psql("-At", "-c", check, dbname=NODE_DB)
    assert absent.returncode == 0, absent.stderr
    assert absent.stdout.strip() == "1"
