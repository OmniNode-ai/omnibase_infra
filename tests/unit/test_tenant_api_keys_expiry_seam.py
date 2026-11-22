# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26472 -- the ``tenant_api_keys.expires_at`` seam, read as data.

``docker/migrations/seams/tenant_api_keys_expiry.sql`` adds a nullable
``expires_at timestamptz`` and a partial index to the cloud identity table on
the lanes that serve the gateway. It runs from the forward-migration runner
behind its own lane indicator, so these pins read the committed seam and the
runner's wiring: the ``ALTER`` sits inside the table-present gate with a named
skip in its ``\\else``, the readback raises on any drift, and the runner calls
the seam only for the two gateway lanes and skips by name otherwise.

The live behaviour (a scratch Postgres with ``omninode_cloud`` and no
``tenant_api_keys``) is proved by
``tests/integration/migrations/test_tenant_api_keys_expiry_runner_omn26472.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SEAM = ROOT / "docker" / "migrations" / "seams" / "tenant_api_keys_expiry.sql"
RUNNER = ROOT / "scripts" / "run-forward-migrations.sh"
GRANTS_CALL = "gateway_grants_file=/migrations/seams/gateway_runtime_grants.sql"
INDICATOR = "ONEX_MIGRATION_GATEWAY_LANE"

SKIP_DB_ABSENT = (
    r"\echo [forward-migration]   skip  tenant_api_keys expiry "
    "(database omninode_cloud absent on this lane)"
)
SKIP_TABLE_ABSENT = (
    r"\echo [forward-migration]   skip  tenant_api_keys expiry "
    "(table public.tenant_api_keys absent in omninode_cloud on this lane)"
)
OK_LINE = r"\echo [forward-migration]   ok tenant_api_keys.expires_at present"
SKIP_UNSET = (
    "[forward-migration]   skip  tenant_api_keys expiry "
    "(ONEX_MIGRATION_GATEWAY_LANE unset: this lane does not serve the gateway)"
)


def _seam() -> str:
    assert SEAM.is_file(), f"{SEAM} does not exist"
    return SEAM.read_text(encoding="utf-8")


def _sql_lines(text: str) -> list[str]:
    """The seam's lines with ``--`` comment lines dropped."""
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("--")]


def test_seam_adds_a_nullable_timestamptz_column_idempotently() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    assert re.search(
        r"ALTER\s+TABLE\s+public\.tenant_api_keys\s+"
        r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+expires_at\s+timestamptz\s+NULL\s*;",
        sql,
    ), "the ALTER must be ADD COLUMN IF NOT EXISTS expires_at timestamptz NULL"
    alter = re.search(r"ALTER\s+TABLE[^;]*;", sql)
    assert alter is not None
    assert not re.search(r"NOT\s+NULL", alter.group(0), re.IGNORECASE), (
        "expires_at must stay nullable: existing key rows carry no expiry"
    )
    assert not re.search(r"\bDEFAULT\b", alter.group(0), re.IGNORECASE), (
        "no default: existing rows must be unchanged by the ALTER"
    )


def test_seam_adds_a_partial_index_on_expires_at_where_is_active() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    assert re.search(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+\w+\s+ON\s+public\.tenant_api_keys"
        r"\s*\(\s*expires_at\s*\)\s+WHERE\s+is_active\s*;",
        sql,
    )


def test_alter_sits_inside_the_to_regclass_gate_with_the_named_skip_in_else() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    assert "to_regclass('public.tenant_api_keys') IS NOT NULL" in sql
    assert re.search(r"\\gset", sql)
    gate = re.search(r"\\if :table_present\b(?P<body>.*?)\\endif", sql, re.DOTALL)
    assert gate is not None, "no `\\if :table_present` gate around the ALTER"
    body = gate.group("body")
    then_branch, sep, else_branch = body.partition(r"\else")
    assert sep, "the table gate has no \\else branch"
    assert "ALTER TABLE public.tenant_api_keys" in then_branch
    assert "ALTER TABLE" not in else_branch
    assert SKIP_TABLE_ABSENT in else_branch
    assert OK_LINE in then_branch


def test_database_gate_precedes_the_connect_and_names_its_skip() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    db_gate = sql.index("datname = 'omninode_cloud'")
    connect = sql.index(r"\connect omninode_cloud")
    table_gate = sql.index("to_regclass('public.tenant_api_keys')")
    alter = sql.index("ALTER TABLE")
    assert db_gate < connect < table_gate < alter
    assert SKIP_DB_ABSENT in sql
    skip_at = sql.index(SKIP_DB_ABSENT)
    assert sql.rindex(r"\else", 0, skip_at) > alter, (
        "the database-absent skip must sit in the \\else of the database gate"
    )


def test_readback_raises_on_every_drift_the_plan_names() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    readback = re.search(
        r"DO\s+\$(?P<tag>\w+)\$(?P<body>.*?)\$(?P=tag)\$", sql, re.DOTALL
    )
    assert readback is not None, "no DO block reads the column back"
    body = readback.group("body")
    assert "RAISE EXCEPTION" in body
    # missing column, nullability, type, and the index's key and predicate
    assert "information_schema.columns" in body or "pg_attribute" in body
    assert "is_nullable" in body or "attnotnull" in body
    assert "timestamp with time zone" in body or "timestamptz" in body
    assert "pg_index" in body
    assert "indpred" in body
    assert "indkey" in body or "pg_get_indexdef" in body
    # the readback runs after the DDL, inside the gate
    assert sql.index("ALTER TABLE") < sql.index("RAISE EXCEPTION")
    assert sql.index("RAISE EXCEPTION") < sql.index(OK_LINE)


def test_seam_carries_no_grant_and_no_data_write() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    assert not re.search(r"\b(GRANT|REVOKE|INSERT|UPDATE|DELETE|DROP|TRUNCATE)\b", sql)


def test_runner_documents_the_indicator_beside_the_fence_indicator() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    header = text[: text.index("\nset -e\n")]
    fence = header.index("#   ONEX_MIGRATION_LANE (default")
    gateway = header.index(f"#   {INDICATOR} (default")
    assert gateway > fence
    assert "`dev`" in header[gateway:]
    assert "`stability`" in header[gateway:]


def test_runner_runs_the_seam_only_for_the_two_gateway_lanes() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    case = re.search(
        r'case "\$\{ONEX_MIGRATION_GATEWAY_LANE:-\}" in(?P<body>.*?)\nesac',
        text,
        re.DOTALL,
    )
    assert case is not None, "no case on ONEX_MIGRATION_GATEWAY_LANE in the runner"
    body = case.group("body")
    arms = re.split(r"\n\s*;;\s*", body)
    run_arm = next(a for a in arms if "dev|stability)" in a)
    assert "/migrations/seams/tenant_api_keys_expiry.sql" in run_arm
    assert "-v ON_ERROR_STOP=1" in run_arm
    assert "psql" in run_arm
    # an unset value and an unknown value both skip by name, and neither runs psql
    other_arms = [a for a in arms if a is not run_arm]
    assert any(SKIP_UNSET in a for a in other_arms)
    assert not any("psql" in a for a in other_arms)
    assert any(
        "*)" in a and "WARNING" in a and "unknown" in a.lower() for a in other_arms
    ), "an unknown value must warn (fail closed)"


def test_runner_calls_the_expiry_seam_before_the_grant_seams() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    call = text.index('case "${ONEX_MIGRATION_GATEWAY_LANE:-}" in')
    assert call < text.index("# ---- BEGIN login-only role grant seam (OMN-18060) ----")
    assert call < text.index(GRANTS_CALL)
