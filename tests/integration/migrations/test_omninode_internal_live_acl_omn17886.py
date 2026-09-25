# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17886: the live-ACL validator against a REAL Postgres catalog.

The unit tests pin the diff over hand-built rows. This proves the three catalog
queries read a real ``pg_class.relacl`` / ``nspacl`` / ``pg_default_acl`` the way
the diff expects, end to end through the CLI, on a throwaway cluster:

* GREEN: a schema granted exactly as the onex-dev topology declares -> exit 0.
* RED: one undeclared grant plus one missing declared grant -> exit 1, both named.
* A default-privilege rule in the schema -> exit 1, named.
* Positive control: the GREEN run reports every declared relation as seen.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.topology import load_topology_profile
from omnibase_infra.validation.omninode_internal_live_acl import declared_acl
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]
CLI = REPO_ROOT / "scripts" / "validation" / "check_omninode_internal_live_acl.py"
DB = "omn17886_acl"
ROLES = ("omninode_runtime", "validator_ro", "jake_ro")


def _setup(pg: EphemeralPostgres) -> None:
    declared = declared_acl(load_topology_profile("onex-dev"))
    statements = [f"CREATE DATABASE {DB}"]
    assert pg.psql("-v", "ON_ERROR_STOP=1", "-c", statements[0]).returncode == 0
    body = [f"CREATE ROLE {role} NOLOGIN;" for role in ROLES]
    body.append("CREATE SCHEMA omninode_internal;")
    for principal, privilege in sorted(declared.schema_grants):
        body.append(f"GRANT {privilege} ON SCHEMA omninode_internal TO {principal};")
    for relation in sorted(declared.relations):
        body.append(f"CREATE TABLE omninode_internal.{relation} (id integer);")
    for principal, relation, privilege in sorted(declared.table_grants):
        body.append(
            f"GRANT {privilege} ON omninode_internal.{relation} TO {principal};"
        )
    result = pg.psql("-v", "ON_ERROR_STOP=1", "-c", "\n".join(body), dbname=DB)
    assert result.returncode == 0, result.stderr


def _run(pg: EphemeralPostgres) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["OMN17886_IT_DSN"] = (
        f"host={pg.socket_dir} port={pg.port} user=postgres dbname={DB}"
    )
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--profile",
            "onex-dev",
            "--dsn-env",
            "OMN17886_IT_DSN",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=REPO_ROOT,
    )


def test_live_catalog_green_red_and_default_rule(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    _setup(pg)
    declared = declared_acl(load_topology_profile("onex-dev"))

    green = _run(pg)
    assert green.returncode == 0, green.stdout + green.stderr
    assert green.stdout.strip().endswith("clean")
    # Positive control: every declared relation was seen in the live catalog.
    assert f"{len(declared.relations)} live relation(s)" in green.stdout

    red_sql = (
        "GRANT SELECT ON omninode_internal.live_events TO jake_ro;\n"
        "REVOKE UPDATE ON omninode_internal.work_events FROM omninode_runtime;"
    )
    assert pg.psql("-v", "ON_ERROR_STOP=1", "-c", red_sql, dbname=DB).returncode == 0
    red = _run(pg)
    assert red.returncode == 1, red.stdout + red.stderr
    assert (
        "UNDECLARED_GRANT jake_ro SELECT ON omninode_internal.live_events" in red.stdout
    )
    assert (
        "MISSING_DECLARED_GRANT omninode_runtime UPDATE ON omninode_internal.work_events"
        in red.stdout
    )

    restore_and_default = (
        "REVOKE SELECT ON omninode_internal.live_events FROM jake_ro;\n"
        "GRANT UPDATE ON omninode_internal.work_events TO omninode_runtime;\n"
        "ALTER DEFAULT PRIVILEGES IN SCHEMA omninode_internal "
        "GRANT SELECT ON TABLES TO jake_ro;"
    )
    assert (
        pg.psql(
            "-v", "ON_ERROR_STOP=1", "-c", restore_and_default, dbname=DB
        ).returncode
        == 0
    )
    default_rule = _run(pg)
    assert default_rule.returncode == 1, default_rule.stdout + default_rule.stderr
    assert (
        "UNDECLARED_DEFAULT_PRIVILEGE owner=postgres objtype=r jake_ro SELECT "
        "IN SCHEMA omninode_internal" in default_rule.stdout
    )
