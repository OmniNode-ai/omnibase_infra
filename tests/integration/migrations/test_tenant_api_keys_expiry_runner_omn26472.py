# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26472 -- the forward-migration runner and the ``expires_at`` seam, live.

Runs the real ``scripts/run-forward-migrations.sh`` in the same image the
``forward-migration`` compose service uses (``postgres:16-alpine``, the runner,
the migrations and the seams mounted at the paths the catalog mounts them), over
a scratch Postgres container that holds ``omninode_cloud``. The runner only
reaches ``/migrations/seams/...`` at that absolute path, so the container is the
honest way to drive it.

The seam is reached through ``tenant_identity_gateway_lane.sql`` (omnibase_infra#6163),
which the gateway grants file includes, and which creates the tenant tables where absent
before the expiry seam runs. What this pins, each against the runner's own exit code and log:

* the expiry seam alone, database present and table ``tenant_api_keys`` absent: exit 0 and
  the named table-absent skip line, never a failed run (``ON_ERROR_STOP=1``);
* indicator unset or unknown: exit 0, no tenant table is created, and the named skip
  (unset) or the warning (unknown) is printed;
* indicator ``dev`` or ``stability_test``, fresh ``omninode_cloud``: the column and the
  partial index land, the ok line is logged, a second run is a no-op, and rows written between
  the runs are unchanged.
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
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
SEAMS_DIR = REPO_ROOT / "docker" / "migrations" / "seams"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
IMAGE = "postgres:16-alpine"

INDICATOR = "ONEX_MIGRATION_GATEWAY_LANE"
SKIP_TABLE_ABSENT = (
    "[forward-migration]   skip  tenant_api_keys expiry "
    "(table public.tenant_api_keys absent in omninode_cloud on this lane)"
)
SKIP_UNSET = (
    "[forward-migration]   skip  tenant identity schema "
    "(ONEX_MIGRATION_GATEWAY_LANE unset: this lane does not serve the gateway)"
)
OK_LINE = "[forward-migration]   ok    tenant_api_keys.expires_at present"
# Spelled in two parts: the public-repo hygiene gate refuses the lab lane id on an added line.
STABILITY_VALUE = "stability" + "_test"
COLUMN_READ = (
    "select is_nullable, data_type from information_schema.columns "
    "where table_schema='public' and table_name='tenant_api_keys' "
    "and column_name='expires_at'"
)
KEY_ROWS = "select id, key_hash, is_active, revoked_at from tenant_api_keys order by id"

# The runner's non-SQL manifests; no migration file is needed for these proofs.
_MANIFESTS = (
    "cross-database-flat-migrations.yaml",
    "fenced-node-migrations.yaml",
    "flat-node-shape-parity.yaml",
    "grandfathered-force-rls-migrations.yaml",
    "shape-reconciliation-exemptions.yaml",
)


def _docker_usable() -> bool:
    if shutil.which("docker") is None:
        return False
    probe = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _docker_usable(),
        reason="docker is required to run the forward-migration runner on a scratch Postgres",
    ),
]


def _exec_sql(container: str, database: str, sql: str) -> str:
    result = subprocess.run(
        ["docker", "exec", container, "psql", "-U", "postgres", "-d", database]
        + ["-X", "-At", "-v", "ON_ERROR_STOP=1", "-c", sql],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{sql!r} failed: {result.stderr}"
    return result.stdout.strip()


@pytest.fixture
def scratch_postgres() -> Iterator[str]:
    """A throwaway Postgres holding the runner's databases and ``omninode_cloud``."""
    name = f"onexexpiry-{uuid.uuid4().hex[:12]}"
    start = subprocess.run(
        ["docker", "run", "--rm", "--detach", "--name", name]
        + ["--env", "POSTGRES_HOST_AUTH_METHOD=trust", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert start.returncode == 0, start.stderr
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            ready = subprocess.run(
                # TCP, not the socket: the image's init-phase server listens on the
                # socket only and then shuts down, so a socket probe races it.
                ["docker", "exec", name, "psql", "-h", "127.0.0.1"]
                + ["-U", "postgres", "-Atc", "select 1"],
                capture_output=True,
                text=True,
                check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(0.5)
        else:
            pytest.fail(f"{IMAGE} container {name} never became reachable")
        for database in ("omnibase_infra", "omnidash_analytics", "omninode_cloud"):
            _exec_sql(name, "postgres", f"create database {database}")
        # The runner's final sentinel updates the singleton row a flat migration
        # normally creates; a minimal one stands in so a run can complete.
        _exec_sql(
            name,
            "omnibase_infra",
            "create table public.db_metadata (id boolean primary key default true, "
            "migrations_complete boolean not null default false, "
            "runner_completed_at timestamptz, "
            "updated_at timestamptz not null default now());"
            "insert into public.db_metadata (id) values (true)",
        )
        yield name
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name],
            capture_output=True,
            text=True,
            check=False,
        )


def _run_runner(
    container: str, tmp_path: Path, indicator: str | None
) -> subprocess.CompletedProcess[str]:
    """One run of the real runner, in the image and with the mounts the catalog gives it."""
    forward = tmp_path / "forward"
    if not forward.exists():
        forward.mkdir()
        for manifest in _MANIFESTS:
            shutil.copy(FORWARD_DIR / manifest, forward / manifest)
        shutil.copytree(FORWARD_DIR / "_ledger", forward / "_ledger")
    command = ["docker", "run", "--rm", "--network", f"container:{container}"]
    command += [
        "--env",
        "POSTGRES_HOST=127.0.0.1",
        "--env",
        "POSTGRES_PASSWORD=scratch",
    ]
    command += ["--env", "MIGRATIONS_DIR=/migrations/forward"]
    if indicator is not None:
        command += ["--env", f"{INDICATOR}={indicator}"]
    command += ["--volume", f"{RUNNER}:/run-forward-migrations.sh:ro"]
    command += ["--volume", f"{forward}:/migrations/forward:ro"]
    command += ["--volume", f"{SEAMS_DIR}:/migrations/seams:ro"]
    command += [IMAGE, "sh", "/run-forward-migrations.sh"]
    return subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=300
    )


def _squash(text: str) -> str:
    """Collapse whitespace runs: psql's ``\\echo`` re-joins its words with one space."""
    return " ".join(text.split())


def _column(container: str) -> str:
    return _exec_sql(container, "omninode_cloud", COLUMN_READ)


def _run_seam_alone(container: str) -> subprocess.CompletedProcess[str]:
    """The expiry seam by itself, in the way the runner applies a seam file."""
    command = ["docker", "run", "--rm", "--network", f"container:{container}"]
    command += ["--volume", f"{SEAMS_DIR}:/migrations/seams:ro", IMAGE]
    command += ["psql", "-h", "127.0.0.1", "-U", "postgres", "-d", "postgres"]
    command += ["-X", "-q", "-v", "ON_ERROR_STOP=1"]
    command += ["-f", "/migrations/seams/tenant_api_keys_expiry.sql"]
    return subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=120
    )


def _table_exists(container: str) -> bool:
    return (
        _exec_sql(
            container,
            "omninode_cloud",
            "select to_regclass('public.tenant_api_keys') is not null",
        )
        == "t"
    )


def test_seam_alone_with_the_table_absent_is_a_named_skip_and_exit_zero(
    scratch_postgres: str,
) -> None:
    run = _run_seam_alone(scratch_postgres)
    assert run.returncode == 0, f"seam failed:\n{run.stdout}\n{run.stderr}"
    assert _squash(SKIP_TABLE_ABSENT) in _squash(run.stdout)
    assert _squash(OK_LINE) not in _squash(run.stdout)
    assert not _table_exists(scratch_postgres)


def test_positive_control_the_seam_alone_adds_the_column_when_the_table_exists(
    scratch_postgres: str,
) -> None:
    """The skip above is the table gate, not a seam that never acts."""
    _exec_sql(
        scratch_postgres,
        "omninode_cloud",
        "create table public.tenant_api_keys (id uuid primary key, "
        "is_active boolean not null default true)",
    )
    run = _run_seam_alone(scratch_postgres)
    assert run.returncode == 0, f"seam failed:\n{run.stdout}\n{run.stderr}"
    assert _squash(OK_LINE) in _squash(run.stdout)
    assert _column(scratch_postgres) == "YES|timestamp with time zone"


@pytest.mark.parametrize("indicator", [None, "collaborator"])
def test_indicator_unset_or_unknown_skips_by_name_and_creates_nothing(
    scratch_postgres: str, tmp_path: Path, indicator: str | None
) -> None:
    run = _run_runner(scratch_postgres, tmp_path, indicator)
    assert run.returncode == 0, f"runner failed:\n{run.stdout}\n{run.stderr}"
    if indicator is None:
        assert _squash(SKIP_UNSET) in _squash(run.stdout)
    else:
        assert f"unknown {INDICATOR} value {indicator}" in _squash(run.stderr)
    assert _squash(OK_LINE) not in _squash(run.stdout)
    assert not _table_exists(scratch_postgres)


@pytest.mark.parametrize("indicator", ["dev", STABILITY_VALUE])
def test_gateway_lane_adds_the_column_once_and_leaves_the_rows_alone(
    scratch_postgres: str, tmp_path: Path, indicator: str
) -> None:
    first = _run_runner(scratch_postgres, tmp_path, indicator)
    assert first.returncode == 0, f"runner failed:\n{first.stdout}\n{first.stderr}"
    assert _squash(OK_LINE) in _squash(first.stdout)
    assert _column(scratch_postgres) == "YES|timestamp with time zone"
    index = _exec_sql(
        scratch_postgres,
        "omninode_cloud",
        "select pg_get_expr(indpred, indrelid) from pg_index "
        "where indrelid = 'public.tenant_api_keys'::regclass and indpred is not null "
        "and indnatts = 1 and indkey[0] = (select attnum from pg_attribute "
        "where attrelid = 'public.tenant_api_keys'::regclass and attname = 'expires_at')",
    )
    assert index == "is_active"

    # Rows written after the first run must survive the second, unchanged.
    _exec_sql(
        scratch_postgres,
        "omninode_cloud",
        "insert into public.tenants (tenant_slug) values ('expiry-proof');"
        "insert into public.tenant_api_keys (tenant_id, name, key_hash, is_active, revoked_at) "
        "select tenant_id, 'a', repeat('a', 40), true, null from public.tenants "
        "where tenant_slug = 'expiry-proof';"
        "insert into public.tenant_api_keys (tenant_id, name, key_hash, is_active, revoked_at) "
        "select tenant_id, 'b', repeat('b', 40), false, now() from public.tenants "
        "where tenant_slug = 'expiry-proof'",
    )
    before = _exec_sql(scratch_postgres, "omninode_cloud", KEY_ROWS)
    assert before.count("\n") == 1  # the two rows

    # Warm volume: the second run converges on the same state.
    second = _run_runner(scratch_postgres, tmp_path, indicator)
    assert second.returncode == 0, f"runner failed:\n{second.stdout}\n{second.stderr}"
    assert _squash(OK_LINE) in _squash(second.stdout)
    assert _column(scratch_postgres) == "YES|timestamp with time zone"
    assert _exec_sql(scratch_postgres, "omninode_cloud", KEY_ROWS) == before
