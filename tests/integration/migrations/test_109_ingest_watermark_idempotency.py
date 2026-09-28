# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real PostgreSQL proof that migration 109 retries safely and fails closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "docker/migrations/forward/044_create_event_ledger.sql"
MIGRATION = ROOT / "docker/migrations/forward/109_add_event_ledger_ingest_watermark.sql"


def _apply(database: EphemeralPostgres, path: Path) -> None:
    result = database.psql("-v", "ON_ERROR_STOP=1", "-f", str(path))
    assert result.returncode == 0, result.stderr


@pytest.mark.integration
def test_109_reapply_accepts_exact_existing_constraint(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, BASE)
    _apply(ephemeral_postgres, MIGRATION)
    _apply(ephemeral_postgres, MIGRATION)

    with ephemeral_postgres.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT regexp_replace(lower(pg_get_constraintdef(oid)), "
            "'[[:space:]()]', '', 'g'), convalidated "
            "FROM pg_constraint WHERE conrelid = 'public.event_ledger'::regclass "
            "AND conname = 'event_ledger_ingest_watermark_positive'"
        )
        assert cursor.fetchone() == (
            "checkingest_watermarkisnulloringest_watermark>0",
            True,
        )


@pytest.mark.integration
def test_109_refuses_wrong_existing_constraint_definition(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    _apply(ephemeral_postgres, BASE)
    conflict = ephemeral_postgres.psql(
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        "ALTER TABLE public.event_ledger ADD COLUMN ingest_watermark BIGINT; "
        "ALTER TABLE public.event_ledger ADD CONSTRAINT "
        "event_ledger_ingest_watermark_positive "
        "CHECK (ingest_watermark IS NULL OR ingest_watermark >= 0)",
    )
    assert conflict.returncode == 0, conflict.stderr

    refused = ephemeral_postgres.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION))
    assert refused.returncode != 0
    assert "noncanonical definition" in refused.stderr
