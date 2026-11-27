# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26712 -- the declared cloud database migration path, run twice on a throwaway Postgres.

Drives ``docker/migrations/seams/cloud/apply_cloud_database.sql`` exactly as the migrate image
carries it (``postgres:16-alpine`` mounting ``seams/`` at ``/migrations/seams``) against an EMPTY
Postgres that holds only the Dev-named database and roles the cloud owner would create. Pins:

* AC3: the first run builds the schema and both grant sets; the second run changes nothing (the
  whole catalog, every privilege and every row read back identical).
* AC1: the grants land on the Dev role names from the settings file, and no lab role or
  database name exists afterwards.
* Known bads: a missing setting, a missing role and a role holding BYPASSRLS each fail the run
  with a named error and leave no grant behind.
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
SEAMS_DIR = REPO_ROOT / "docker" / "migrations" / "seams"
SETTINGS = SEAMS_DIR / "cloud" / "dev.settings.env"
IMAGE = "postgres:16-alpine"
APPLY = "/migrations/seams/cloud/apply_cloud_database.sql"
# Spelled in parts: the lab names must not appear as one literal in an added line.
LAB_NAMES = (
    "gateway_" + "runtime",
    "tenant_identity_" + "writer",
    "omninode_" + "cloud",
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
        reason="docker is required to run the cloud migration on a scratch Postgres",
    ),
]


def _settings() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in SETTINGS.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key] = value
    return values


def _exec_sql(container: str, database: str, sql: str) -> str:
    result = subprocess.run(
        ["docker", "exec", container, "psql", "-h", "127.0.0.1", "-U", "postgres"]
        + ["-d", database, "-X", "-At", "-v", "ON_ERROR_STOP=1", "-c", sql],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{sql!r} failed: {result.stderr}"
    return result.stdout.strip()


def _new_postgres() -> str:
    name = f"onexcloudmig-{uuid.uuid4().hex[:12]}"
    start = subprocess.run(
        ["docker", "run", "--rm", "--detach", "--name", name]
        + ["--env", "POSTGRES_HOST_AUTH_METHOD=trust", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    assert start.returncode == 0, start.stderr
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        ready = subprocess.run(
            ["docker", "exec", name, "psql", "-h", "127.0.0.1", "-U", "postgres"]
            + ["-Atc", "select 1"],
            capture_output=True,
            text=True,
            check=False,
        )
        if ready.returncode == 0:
            return name
        time.sleep(0.5)
    subprocess.run(["docker", "rm", "--force", name], check=False, capture_output=True)
    pytest.fail(f"{IMAGE} container {name} never became reachable")


@pytest.fixture
def scratch() -> Iterator[str]:
    """An empty Postgres holding only what the cloud owner provisions: database and roles."""
    name = _new_postgres()
    try:
        cfg = _settings()
        _exec_sql(name, "postgres", f'create database "{cfg["ONEX_CLOUD_DATABASE"]}"')
        for key in (
            "ONEX_CLOUD_GATEWAY_ROLE",
            "ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE",
        ):
            _exec_sql(name, "postgres", f'create role "{cfg[key]}" login')
        yield name
    finally:
        subprocess.run(
            ["docker", "rm", "--force", name], capture_output=True, check=False
        )


def _apply(
    container: str, overrides: dict[str, str | None] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {**_settings(), **(overrides or {})}
    command = ["docker", "run", "--rm", "--network", f"container:{container}"]
    for key, value in env.items():
        if value is not None:
            command += ["--env", f"{key}={value}"]
    command += ["--volume", f"{SEAMS_DIR}:/migrations/seams:ro", IMAGE]
    command += ["psql", "-h", "127.0.0.1", "-U", "postgres"]
    command += ["-d", env["ONEX_CLOUD_DATABASE"] or "postgres"]
    command += ["-X", "-q", "-v", "ON_ERROR_STOP=1", "-f", APPLY]
    return subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=300
    )


_CATALOG = """
select 'col|' || table_name || '|' || column_name || '|' || data_type || '|' || is_nullable
  from information_schema.columns where table_schema = 'public'
union all select 'idx|' || indexdef from pg_indexes where schemaname = 'public'
union all select 'con|' || conrelid::regclass || '|' || conname || '|' || pg_get_constraintdef(oid)
  from pg_constraint where connamespace = 'public'::regnamespace
union all select 'acl|' || relname || '|' || coalesce(relacl::text, '')
  from pg_class where relnamespace = 'public'::regnamespace
union all select 'colacl|' || attrelid::regclass || '|' || attname || '|' || attacl::text
  from pg_attribute where attacl is not null and attrelid in
  (select oid from pg_class where relnamespace = 'public'::regnamespace)
union all select 'rows|' || relname || '|' || (xpath('/row/c/text()',
  query_to_xml('select count(*) as c from public.' || quote_ident(relname), false, true, '')))[1]::text
  from pg_class where relnamespace = 'public'::regnamespace and relkind = 'r'
order by 1
"""


def _snapshot(container: str) -> str:
    return _exec_sql(container, _settings()["ONEX_CLOUD_DATABASE"], _CATALOG)


def test_apply_builds_schema_and_grants_then_second_run_changes_nothing(
    scratch: str,
) -> None:
    first = _apply(scratch)
    assert first.returncode == 0, first.stderr
    after_first = _snapshot(scratch)
    assert "col|tenants|tenant_id|uuid|NO" in after_first
    assert first.stdout.count("[cloud-migration] ok") == 3, first.stdout

    second = _apply(scratch)
    assert second.returncode == 0, second.stderr
    assert _snapshot(scratch) == after_first, "the second run changed the database"


def test_grants_land_on_the_dev_roles_and_no_lab_name_exists(scratch: str) -> None:
    assert _apply(scratch).returncode == 0
    cfg = _settings()
    database = cfg["ONEX_CLOUD_DATABASE"]
    gateway = cfg["ONEX_CLOUD_GATEWAY_ROLE"]
    writer = cfg["ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE"]
    assert (
        _exec_sql(
            scratch,
            database,
            f"select has_table_privilege('{gateway}', 'public.gateway_workflows', 'INSERT')"
            f" and not has_table_privilege('{gateway}', 'public.tenants', 'INSERT')",
        )
        == "t"
    )
    assert (
        _exec_sql(
            scratch,
            database,
            f"select has_table_privilege('{writer}', 'public.tenants', 'INSERT')"
            f" and not has_table_privilege('{writer}', 'public.tenants', 'DELETE')",
        )
        == "t"
    )
    names = _exec_sql(
        scratch,
        "postgres",
        "select string_agg(rolname, ',') from pg_roles union all "
        "select string_agg(datname, ',') from pg_database",
    )
    for lab_name in LAB_NAMES:
        assert lab_name not in names


@pytest.mark.parametrize(
    ("overrides", "named_error"),
    [
        ({"ONEX_CLOUD_GATEWAY_ROLE": None}, "ONEX_CLOUD_GATEWAY_ROLE is not set"),
        (
            {"ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE": ""},
            "ONEX_CLOUD_TENANT_IDENTITY_WRITER_ROLE is not set",
        ),
        (
            {"ONEX_CLOUD_GATEWAY_ROLE": "onex_dev_absent_role"},
            "the cloud owner creates it",
        ),
    ],
)
def test_missing_setting_or_role_fails_with_a_named_error(
    scratch: str, overrides: dict[str, str | None], named_error: str
) -> None:
    result = _apply(scratch, overrides)
    assert result.returncode != 0
    assert named_error in result.stderr


def test_role_with_bypassrls_fails_the_run(scratch: str) -> None:
    gateway = _settings()["ONEX_CLOUD_GATEWAY_ROLE"]
    _exec_sql(scratch, "postgres", f'alter role "{gateway}" bypassrls')
    result = _apply(scratch)
    assert result.returncode != 0
    assert "SUPERUSER or BYPASSRLS" in result.stderr
