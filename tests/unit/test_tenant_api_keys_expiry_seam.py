# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26472 -- the ``tenant_api_keys.expires_at`` seam, read as data.

``docker/migrations/seams/tenant_api_keys_expiry.sql`` (carried by omnibase_infra#6163) adds a
nullable ``expires_at timestamptz`` and a partial index to the cloud identity table on the lanes
that serve the gateway. These pins read the committed seam: the ``ALTER`` sits inside the
table-present gate with a named skip in its ``\\else``, the column is nullable with no default, the
index is the partial one on ``(expires_at) WHERE is_active``, the readback raises on any drift,
and the seam issues no grant and writes no row. The seam is reached through
``tenant_identity_gateway_lane.sql`` only for the two gateway lanes; that wiring and the lane
indicator in the rendered compose are pinned by ``tests/unit/test_tenant_identity_schema_seam.py``
and the render tests.

The live behaviour (a scratch Postgres, the real runner) is proved by
``tests/integration/migrations/test_tenant_api_keys_expiry_runner_omn26472.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
SEAM = ROOT / "docker" / "migrations" / "seams" / "tenant_api_keys_expiry.sql"

SKIP_DB_ABSENT = (
    r"\echo [forward-migration]   skip  tenant_api_keys expiry "
    "(database omninode_cloud absent on this lane)"
)
SKIP_TABLE_ABSENT = (
    r"\echo [forward-migration]   skip  tenant_api_keys expiry "
    "(table public.tenant_api_keys absent in omninode_cloud on this lane)"
)
OK_LINE = r"\echo [forward-migration]   ok    tenant_api_keys.expires_at present"


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
    db_gate = sql.index("datname = :'cloud_database'")
    connect = sql.index(r'\connect :"cloud_database"')
    table_gate = sql.index("to_regclass('public.tenant_api_keys')")
    alter = sql.index("ALTER TABLE")
    assert db_gate < connect < table_gate < alter
    assert SKIP_DB_ABSENT in sql
    skip_at = sql.index(SKIP_DB_ABSENT)
    assert sql.rindex(r"\else", 0, skip_at) > alter, (
        "the database-absent skip must sit in the \\else of the database gate"
    )


def _readback_complete(sql: str) -> bool:
    """True when the seam's DO block raises on every drift the plan names."""
    block = re.search(r"DO\s+\$(?P<tag>\w+)\$(?P<body>.*?)\$(?P=tag)\$", sql, re.DOTALL)
    if block is None:
        return False
    body = block.group("body")
    return (
        all(
            needle in body
            for needle in (
                "col_type IS NULL",  # the column is missing
                "col_nullable <> 'YES'",  # not nullable
                "col_type <> 'timestamp with time zone'",  # wrong type
                "pg_catalog.pg_indexes",
                "'CREATE INDEX idx_tenant_api_keys_expires_at ON public.tenant_api_keys "
                "USING btree (expires_at) WHERE is_active'",  # index key and predicate, in full
            )
        )
        and body.count("RAISE EXCEPTION") >= 3
    )


def test_readback_raises_on_every_drift_the_plan_names() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    assert _readback_complete(sql)
    # the readback runs after the DDL, inside the gate
    assert sql.index("ALTER TABLE") < sql.index("RAISE EXCEPTION")
    assert sql.index("RAISE EXCEPTION") < sql.index(OK_LINE)


def test_positive_control_a_readback_missing_one_check_is_seen() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    for dropped in (
        "col_type IS NULL",
        "col_nullable <> 'YES'",
        "col_type <> 'timestamp with time zone'",
        "WHERE is_active'",
    ):
        mutated = sql.replace(dropped, "TRUE", 1)
        assert mutated != sql, dropped
        assert not _readback_complete(mutated), dropped


def test_seam_carries_no_grant_and_no_data_write() -> None:
    sql = "\n".join(_sql_lines(_seam()))
    assert not re.search(r"\b(GRANT|REVOKE|INSERT|UPDATE|DELETE|DROP|TRUNCATE)\b", sql)
