# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev access prerequisite only; these checks do not accept dashboard figures."""

from pathlib import Path

import pytest
from psycopg2 import sql

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]
ROOT = Path(__file__).resolve().parents[3]
GRANTS = ROOT / "deploy/lab/dev-contractor-usage-grants.sql"
TABLES = ("usage_by_model_day", "usage_by_model_day_calls", "llm_call_metrics")
DB = "omnidash_analytics"


def statement(pg: EphemeralPostgres, template: str, table: str) -> str:
    with pg.connect(dbname=DB) as conn:
        return sql.SQL(template).format(sql.Identifier("public", table)).as_string(conn)


def test_grants_enable_only_select(ephemeral_postgres: EphemeralPostgres) -> None:
    pg = ephemeral_postgres
    assert pg.psql("-c", f"CREATE DATABASE {DB}").returncode == 0
    setup = ["CREATE ROLE lakshman_ro LOGIN;"]
    for count, table in enumerate(TABLES, start=1):
        setup += [
            f"CREATE TABLE public.{table} (id integer);",
            f"INSERT INTO public.{table} SELECT generate_series(1,{count});",
        ]
    assert (
        pg.psql("-v", "ON_ERROR_STOP=1", "-c", "\n".join(setup), dbname=DB).returncode
        == 0
    )
    # Falsifier: every read is refused before the tracked grant file is applied.
    for table in TABLES:
        before = pg.psql(
            "-c",
            statement(pg, "SELECT count(*) FROM {}", table),
            user="lakshman_ro",
            dbname=DB,
        )
        assert before.returncode != 0 and "permission denied" in before.stderr

    refused = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(GRANTS), dbname=DB)
    assert refused.returncode != 0 and "dev contractor grant runner" in refused.stderr
    wrong_db = pg.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "SET onex.dev_contractor_grants='dev';",
        "-f",
        str(GRANTS),
    )
    assert wrong_db.returncode != 0 and "analytics database required" in wrong_db.stderr

    # Pre-existing write access is refused and the whole file rolls back.
    first = statement(pg, "{}", TABLES[0])
    for pre_grant in (
        f"GRANT INSERT ON {first} TO lakshman_ro",
        f"GRANT SELECT ON {first} TO lakshman_ro WITH GRANT OPTION",
    ):
        assert pg.psql("-c", pre_grant, dbname=DB).returncode == 0
        broader = pg.psql(
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "SET onex.dev_contractor_grants='dev';",
            "-f",
            str(GRANTS),
            dbname=DB,
        )
        assert broader.returncode != 0 and "must end SELECT-only" in broader.stderr
        assert (
            pg.psql(
                "-c", f"REVOKE ALL ON {first} FROM lakshman_ro", dbname=DB
            ).returncode
            == 0
        )
        rolled_back = pg.psql(
            "-c",
            statement(pg, "SELECT count(*) FROM {}", TABLES[1]),
            user="lakshman_ro",
            dbname=DB,
        )
        assert rolled_back.returncode != 0 and "permission denied" in rolled_back.stderr

    # Reapplying is safe. The real SQL is the proof surface, not a test copy.
    for _ in range(2):
        applied = pg.psql(
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            "SET onex.dev_contractor_grants='dev';",
            "-f",
            str(GRANTS),
            dbname=DB,
        )
        assert applied.returncode == 0, applied.stderr
    for count, table in enumerate(TABLES, start=1):
        after = pg.psql(
            "-Atc",
            statement(pg, "SELECT count(*) FROM {}", table),
            user="lakshman_ro",
            dbname=DB,
        )
        assert after.returncode == 0 and after.stdout.strip() == str(count)
        for write_sql in (
            statement(pg, "INSERT INTO {} VALUES (99)", table),
            statement(pg, "UPDATE {} SET id=99 WHERE false", table),
            statement(pg, "DELETE FROM {} WHERE false", table),
            statement(pg, "TRUNCATE {}", table),
        ):
            denied = pg.psql("-c", write_sql, user="lakshman_ro", dbname=DB)
            assert denied.returncode != 0 and "permission denied" in denied.stderr
    acl = pg.psql(
        "-Atc",
        "SELECT DISTINCT privilege_type, is_grantable FROM information_schema.role_table_grants WHERE grantee='lakshman_ro'",
        dbname=DB,
    )
    assert acl.stdout.strip() == "SELECT|NO"
