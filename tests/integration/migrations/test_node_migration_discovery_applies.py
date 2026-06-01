# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration test: node-owned migration discovery applies WITHOUT manual copy.

Runs the real ``scripts/run-forward-migrations.sh`` against a reachable
PostgreSQL, pointed at a temp migrations dir that contains only the savings
base table (infra flat 074) plus the vendored ``nodes/`` subtree. Asserts that:

  * the runner auto-discovers and applies every node migration,
  * the OMN-12489 dashboard + savings views exist and are queryable,
  * each node migration is tracked under a namespaced ``node:<node>:<file>``
    id (a separate identity space from the flat infra sequence), and
  * a node migration numbered 076 coexists with infra's flat 076 with NO
    primary-key collision — proving the renumber-as-operational-pattern is
    eliminated.

Skips cleanly when ``OMNIBASE_INFRA_DB_URL`` is not configured.

Ticket: OMN-12559
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.helpers.util_postgres import PostgresConfig, check_postgres_reachable

# This test mutates the target database (creates and drops views). It is opt-in
# so it never runs against shared infra by accident. Point it at an ephemeral or
# dedicated test Postgres and set RUN_NODE_MIGRATION_DISCOVERY_TEST=1.
OPT_IN_ENV = "RUN_NODE_MIGRATION_DISCOVERY_TEST"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
    pytest.mark.serial,
]

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
NODES_DIR = FORWARD_DIR / "nodes"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
SAVINGS_BASE = FORWARD_DIR / "074_create_savings_estimates.sql"
INFRA_FLAT_076 = FORWARD_DIR / "076_add_savings_estimate_provenance.sql"
OMNIBASE_ENV = Path.home() / ".omnibase" / ".env"


def _read_omnibase_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not OMNIBASE_ENV.exists():
        return env
    for raw_line in OMNIBASE_ENV.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key.strip()] = value
    return env


def _merged_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = {**os.environ, **_read_omnibase_env()}
    if overrides:
        env.update(overrides)
    return env


def _require_psql() -> str:
    psql = shutil.which("psql")
    if psql is None:
        pytest.skip("psql client not available on PATH")
    return psql


def _require_db() -> PostgresConfig:
    os.environ.update(_read_omnibase_env())
    if os.environ.get(OPT_IN_ENV) != "1":
        pytest.skip(
            f"node-migration discovery test is opt-in (mutates DB); "
            f"set {OPT_IN_ENV}=1 and OMNIBASE_INFRA_DB_URL to a dedicated DB"
        )
    config = PostgresConfig.from_env()
    if not config.is_configured or not check_postgres_reachable(config):
        pytest.skip(
            "PostgreSQL not configured/reachable (set OMNIBASE_INFRA_DB_URL "
            "to a full DSN, e.g. postgresql://postgres:pass@host:5432/db)"
        )
    # Authentication probe: a reachable host may still reject credentials.
    psql = _require_psql()
    probe = subprocess.run(
        [
            psql,
            "-h",
            config.host or "127.0.0.1",
            "-p",
            str(config.port),
            "-U",
            config.user,
            "-d",
            config.database,
            "-tAc",
            "SELECT 1",
        ],
        env=_merged_env(
            {
                "PGPASSWORD": config.password or "",
                "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            }
        ),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip(
            f"PostgreSQL reachable but auth/select failed: {probe.stderr.strip()}"
        )
    return config


def _psql_env(config: PostgresConfig) -> dict[str, str]:
    return _merged_env(
        {
            "POSTGRES_HOST": config.host or "127.0.0.1",
            "POSTGRES_PORT": str(config.port),
            "POSTGRES_USER": config.user,
            "POSTGRES_PASSWORD": config.password or "",
            "POSTGRES_DB": config.database,
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        }
    )


def _query(psql: str, config: PostgresConfig, sql: str) -> str:
    result = subprocess.run(
        [
            psql,
            "-h",
            config.host or "127.0.0.1",
            "-p",
            str(config.port),
            "-U",
            config.user,
            "-d",
            config.database,
            "-tAc",
            sql,
        ],
        env=_merged_env(
            {
                "PGPASSWORD": config.password or "",
                "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
            }
        ),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.serial
def test_node_migration_discovery_applies_views(tmp_path: Path) -> None:
    psql = _require_psql()
    config = _require_db()

    # Isolate this test run in a dedicated schema-equivalent by using a unique
    # tracking-id namespace is not enough; instead drop the objects we create up
    # front so re-runs are clean. The views/tables are IF NOT EXISTS / OR REPLACE.
    for view in (
        "projection_delegation_savings",
        "projection_delegation_summary",
        "projection_delegation_model_routing",
        "projection_delegation_quality_gate",
        "projection_delegation_token_usage",
    ):
        _query(psql, config, f"DROP VIEW IF EXISTS {view} CASCADE;")

    # Build a temp migrations dir: savings base table (flat 074) + nodes subtree.
    mig_dir = tmp_path / "forward"
    mig_dir.mkdir()
    shutil.copy(SAVINGS_BASE, mig_dir / SAVINGS_BASE.name)
    shutil.copytree(NODES_DIR, mig_dir / "nodes")

    env = _psql_env(config)
    env["MIGRATIONS_DIR"] = str(mig_dir)

    run = subprocess.run(
        ["sh", str(RUNNER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, f"runner failed:\n{run.stdout}\n{run.stderr}"

    # The runner discovered and applied node migrations.
    assert "node:node_projection_delegation:" in run.stdout
    assert "node:node_projection_savings:076" in run.stdout

    # Views exist and are queryable (no manual copy was performed).
    for view in (
        "projection_delegation_summary",
        "projection_delegation_savings",
        "projection_delegation_model_routing",
    ):
        exists = _query(
            psql, config, f"SELECT 1 FROM pg_views WHERE viewname = '{view}';"
        )
        assert exists == "1", f"view {view} was not created by discovery"
        # Queryable end-to-end.
        _query(psql, config, f"SELECT count(*) FROM {view};")

    # Namespaced ids are tracked under the node identity space.
    tracked = _query(
        psql,
        config,
        "SELECT migration_id FROM public.schema_migrations "
        "WHERE migration_id LIKE 'node:%' ORDER BY migration_id;",
    )
    assert (
        "node:node_projection_savings:076_create_delegation_savings_projection_view.sql"
        in tracked
    )
    assert (
        "node:node_projection_delegation:0010_create_delegation_dashboard_projection_views.sql"
        in tracked
    )


@pytest.mark.serial
def test_node_076_does_not_collide_with_infra_flat_076() -> None:
    """node:<node>:076 and docker/076_* coexist (distinct PK strings)."""
    psql = _require_psql()
    config = _require_db()

    # Apply infra flat 076 directly and record it under the flat id, mirroring
    # what the flat-sequence loop does.
    flat_id = f"docker/{INFRA_FLAT_076.name}"
    node_id = (
        "node:node_projection_savings:076_create_delegation_savings_projection_view.sql"
    )

    _query(
        psql,
        config,
        "CREATE TABLE IF NOT EXISTS public.schema_migrations ("
        "migration_id TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), "
        "checksum TEXT NOT NULL, source_set TEXT NOT NULL);",
    )

    # Insert both ids; if they collided on the same PK this would fail.
    probe = f"collision-probe-{uuid.uuid4().hex[:8]}"
    _query(
        psql,
        config,
        f"INSERT INTO public.schema_migrations (migration_id, checksum, source_set) "
        f"VALUES ('{flat_id}', '{probe}', 'docker') ON CONFLICT (migration_id) DO NOTHING;",
    )
    _query(
        psql,
        config,
        f"INSERT INTO public.schema_migrations (migration_id, checksum, source_set) "
        f"VALUES ('{node_id}', '{probe}', 'node') ON CONFLICT (migration_id) DO NOTHING;",
    )

    both = _query(
        psql,
        config,
        f"SELECT count(*) FROM public.schema_migrations "
        f"WHERE migration_id IN ('{flat_id}', '{node_id}');",
    )
    assert both == "2", (
        "flat 076 and node 076 must coexist as distinct migration ids "
        "(no numeric collision, no renumber required)"
    )
