# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17886 AC4: the live-ACL check as a gate over a from-empty build.

``omnidash_analytics`` is built from EMPTY by the real
``scripts/run-forward-migrations.sh`` over the real corpus, then
``scripts/validation/check_omninode_internal_live_acl.py`` diffs its
``omninode_internal`` ACLs against the onex-dev topology. So a migration that
confers a grant the topology does not declare, or omits one it does, fails CI.

The findings the corpus still produces today are listed, one by one, in
``config/omninode_internal_live_acl_allowlist.yaml``. The gate fails in both
directions: a finding not on that list, and an entry on it that no longer
reproduces. The second direction is what keeps the list shrink-only.

Where the Postgres comes from:

* ``OMN17886_GATE_DB_URL`` (a superuser URL) names an already-running server,
  which is how the migration-integration CI job runs it against its own
  ``services.postgres``. The server must not already hold ``omnibase_infra`` or
  ``omnidash_analytics``: the test creates both, builds them, and drops them.
* Otherwise an ephemeral cluster is started with initdb/pg_ctl.
* ``OMN17886_REQUIRE_PG=1`` turns "no Postgres available" into a failure, so a
  runner that should run this proof can never report it as a vacuous skip.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import yaml
from psycopg2.extensions import make_dsn

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl
from tests.integration.migrations.conftest import PG_TOOLS_MISSING

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
CLI = REPO_ROOT / "scripts" / "validation" / "check_omninode_internal_live_acl.py"
ALLOWLIST = REPO_ROOT / "config" / "omninode_internal_live_acl_allowlist.yaml"
PROFILE = "onex-dev"
# The real names, not throwaway ones: sixteen forward migrations run
# ``\connect omnidash_analytics``, so a build into any other name is not the
# production apply path. The fixture therefore refuses a server where either
# database already exists, and only ever drops databases it created itself.
INFRA_DB = "omnibase_infra"
NODE_DB = "omnidash_analytics"
_REQUIRE_PG = os.environ.get("OMN17886_REQUIRE_PG") == "1"


@dataclass(frozen=True)
class _Server:
    host: str
    port: int
    user: str
    password: str

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "PGHOST": self.host,
                "PGPORT": str(self.port),
                "PGUSER": self.user,
                "PGPASSWORD": self.password,
            }
        )
        return env

    def psql(
        self, sql: str, dbname: str = "postgres"
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["psql", "-v", "ON_ERROR_STOP=1", "-d", dbname, "-c", sql],
            capture_output=True,
            text=True,
            check=False,
            env=self.env(),
        )


def _unavailable(reason: str) -> None:
    if _REQUIRE_PG:
        raise AssertionError(f"OMN17886_REQUIRE_PG=1 but {reason}")
    pytest.skip(reason)


def _load_allowlist() -> list[dict[str, str]]:
    document = yaml.safe_load(ALLOWLIST.read_text(encoding="utf-8"))
    entries = document["entries"]
    assert isinstance(entries, list)
    return entries


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
    if PG_TOOLS_MISSING:
        _unavailable("no OMN17886_GATE_DB_URL and initdb/pg_ctl/psql are not on PATH")
    pg = request.getfixturevalue("ephemeral_postgres")
    yield _Server(host=pg.socket_dir, port=pg.port, user="postgres", password="")


def _existing_databases(server: _Server) -> set[str]:
    result = subprocess.run(
        ["psql", "-At", "-d", "postgres", "-c", "SELECT datname FROM pg_database"],
        capture_output=True,
        text=True,
        check=False,
        env=server.env(),
    )
    assert result.returncode == 0, result.stderr
    return set(result.stdout.split())


@pytest.fixture
def fresh_build(server: _Server) -> Iterator[_Server]:
    # From EMPTY, or not at all: never build over, or later drop, a database
    # this test did not create.
    present = _existing_databases(server) & {INFRA_DB, NODE_DB}
    assert not present, (
        f"refusing to build over existing database(s) {sorted(present)}; point "
        "OMN17886_GATE_DB_URL at a server of its own"
    )
    for db in (INFRA_DB, NODE_DB):
        result = server.psql(f"CREATE DATABASE {db}")
        assert result.returncode == 0, result.stderr
    try:
        env = server.env()
        env.update(
            {
                "POSTGRES_HOST": server.host,
                "POSTGRES_PORT": str(server.port),
                "POSTGRES_USER": server.user,
                "POSTGRES_PASSWORD": server.password,
                "POSTGRES_DB": INFRA_DB,
                "NODE_POSTGRES_DB": NODE_DB,
                "MIGRATIONS_DIR": str(MIGRATIONS_DIR),
            }
        )
        applied = subprocess.run(
            ["sh", str(RUNNER)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=REPO_ROOT,
            timeout=900,
        )
        assert applied.returncode == 0, (
            "the forward-migration corpus did not build from empty:\n"
            + applied.stdout[-4000:]
            + applied.stderr[-4000:]
        )
        yield server
    finally:
        for db in (NODE_DB, INFRA_DB):
            server.psql(f"DROP DATABASE IF EXISTS {db} WITH (FORCE)")


def test_allowlist_entries_are_unique_and_name_their_removal() -> None:
    entries = _load_allowlist()
    findings = [entry["finding"] for entry in entries]
    assert len(findings) == len(set(findings)), "duplicate allow-list entries"
    for entry in entries:
        assert set(entry) == {"finding", "removed_by"}, entry
        assert entry["finding"].strip(), entry
        assert entry["removed_by"].startswith("OMN-"), entry


def test_fresh_build_matches_the_topology_modulo_the_shrink_only_allowlist(
    fresh_build: _Server,
) -> None:
    env = fresh_build.env()
    # make_dsn quotes every value. A hand-built "password= dbname=x" parses as
    # password='dbname=x', and the check then reads the default database.
    env["OMN17886_GATE_NODE_DSN"] = make_dsn(
        host=fresh_build.host,
        port=fresh_build.port,
        user=fresh_build.user,
        password=fresh_build.password or None,
        dbname=NODE_DB,
    )
    result = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--profile",
            PROFILE,
            "--dsn-env",
            "OMN17886_GATE_NODE_DSN",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=REPO_ROOT,
    )
    assert result.returncode in (0, 1), result.stderr
    report = json.loads(result.stdout)

    # Positive control: the build produced the schema and the check saw every
    # relation the topology declares, so an empty findings list cannot come from
    # reading an empty or missing schema.
    declared_relations = declared_acl(load_topology_profile(PROFILE)).relations
    assert report["observed_relation_count"] >= len(declared_relations) > 0, report

    live = set(report["findings"])
    allowed = {entry["finding"] for entry in _load_allowlist()}
    new = sorted(live - allowed)
    stale = sorted(allowed - live)
    assert not new, (
        "live omninode_internal ACLs differ from the topology in ways the "
        "allow-list does not record; fix the migration or the declaration, "
        "do not list them:\n" + "\n".join(new)
    )
    assert not stale, (
        "these allow-list entries no longer reproduce; delete them from "
        f"{ALLOWLIST.relative_to(REPO_ROOT)} in this change:\n" + "\n".join(stale)
    )
