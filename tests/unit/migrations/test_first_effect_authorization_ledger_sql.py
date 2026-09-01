# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline contract tests for migration 104's durable state-machine guards."""

from __future__ import annotations

from pathlib import Path

import pytest

_MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "104_create_first_effect_authorization_ledger.sql"
)
_ROLLBACK = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_104_create_first_effect_authorization_ledger.sql"
)


@pytest.mark.unit
def test_first_effect_ledger_has_only_redacted_identity_and_required_uniqueness() -> (
    None
):
    sql = _MIGRATION.read_text(encoding="utf-8")
    table_definition = "\n".join(
        line
        for line in sql.split(");", maxsplit=1)[0].splitlines()
        if not line.lstrip().startswith("--")
    )

    assert "CREATE TABLE public.first_effect_authorization_ledger" in sql
    assert "CREATE TABLE IF NOT EXISTS" not in sql
    assert "authorization_digest      TEXT PRIMARY KEY" in sql
    assert "nonce_digest              TEXT NOT NULL UNIQUE" in sql
    assert "correlation_id            UUID NOT NULL UNIQUE" in sql
    assert "request_digest            TEXT NOT NULL UNIQUE" in sql
    assert "manifest_hash             TEXT NOT NULL" in sql
    assert "payload" not in table_definition.lower()
    assert "prompt" not in table_definition.lower()
    assert "secret" not in table_definition.lower()


@pytest.mark.unit
def test_first_effect_ledger_uses_explicit_public_objects_and_has_a_rollback() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")
    rollback = _ROLLBACK.read_text(encoding="utf-8")

    assert "ON public.first_effect_authorization_ledger" in sql
    assert "CREATE FUNCTION public.enforce_first_effect_authorization_transition" in sql
    assert "SET search_path = pg_catalog, public" in sql
    assert "CREATE TRIGGER trg_first_effect_authorization_transition" in sql
    assert "ON public.first_effect_authorization_ledger" in sql
    assert (
        "DROP TABLE IF EXISTS public.first_effect_authorization_ledger RESTRICT"
        in rollback
    )
    assert (
        "DROP FUNCTION IF EXISTS public.enforce_first_effect_authorization_transition"
        in rollback
    )


@pytest.mark.unit
def test_first_effect_ledger_is_a_non_authorizing_recording_scaffold() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")

    assert "observation/recording scaffold" in sql
    assert "confers no effect permission" in sql
    assert "not grants" in sql
    assert "No GRANT or ALTER OWNER appears here" in sql


@pytest.mark.unit
def test_first_effect_ledger_pins_all_legal_transitions_and_unknown_recovery() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")

    assert "OLD.state = 'ISSUED' AND NEW.state = 'PREFLIGHT_CONSUMED'" in sql
    assert "OLD.state = 'PREFLIGHT_CONSUMED' AND NEW.state = 'PUBLISHING'" in sql
    assert (
        "OLD.state = 'PUBLISHING' AND NEW.state IN ('PUBLISHED_UNKNOWN', 'TERMINAL_OBSERVED', 'BLOCKED')"
        in sql
    )
    assert (
        "OLD.state = 'PUBLISHED_UNKNOWN' AND NEW.state IN ('TERMINAL_OBSERVED', 'BLOCKED')"
        in sql
    )
    assert "illegal first-effect authorization transition" in sql
    assert "NEW.version <> OLD.version + 1" in sql


@pytest.mark.unit
def test_first_effect_ledger_requires_evidence_for_ambiguous_or_terminal_outcome() -> (
    None
):
    sql = _MIGRATION.read_text(encoding="utf-8")

    assert "state = 'PUBLISHED_UNKNOWN'" in sql
    assert "published_unknown_at IS NOT NULL" in sql
    assert "publish_evidence_hash IS NOT NULL" in sql
    assert "state = 'TERMINAL_OBSERVED'" in sql
    assert "terminal_receipt_hash IS NOT NULL" in sql
    assert "identity is immutable" in sql
    assert "evidence and transition timestamps are immutable" in sql
