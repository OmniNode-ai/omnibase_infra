# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Offline contract checks for the payload-free verified-grant migration."""

from __future__ import annotations

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MIGRATION = (
    _ROOT
    / "docker/migrations/forward/105_create_verified_first_effect_grant_ledger.sql"
)
_ROLLBACK = (
    _ROOT
    / "docker/migrations/rollback/rollback_105_create_verified_first_effect_grant_ledger.sql"
)
_RSD_V2_MIGRATION = (
    _ROOT
    / "docker/migrations/forward/106_align_verified_first_effect_grant_with_rsd_v2.sql"
)
_RSD_V2_ROLLBACK = (
    _ROOT
    / "docker/migrations/rollback/rollback_106_align_verified_first_effect_grant_with_rsd_v2.sql"
)


@pytest.mark.unit
def test_105_has_immutable_causal_identities_and_no_payload_copy() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")
    definition = "\n".join(
        line
        for line in sql.split("CREATE FUNCTION", maxsplit=1)[0].splitlines()
        if not line.lstrip().startswith("--")
    )

    assert "CREATE TABLE public.first_effect_verified_grant_ledger" in sql
    assert "authorization_digest              TEXT PRIMARY KEY" in sql
    for identity in (
        "grant_id                          UUID NOT NULL UNIQUE",
        "grant_envelope_id                 UUID NOT NULL UNIQUE",
        "nonce_digest                      TEXT NOT NULL UNIQUE",
        "request_digest                    TEXT NOT NULL UNIQUE",
        "correlation_id                    TEXT NOT NULL UNIQUE",
        "outbox_envelope_id                UUID UNIQUE",
    ):
        assert identity in sql
    assert "REFERENCES public.delegation_workflow_state(correlation_id)" in sql
    assert "pending_emissions" not in definition
    assert "payload" not in definition.lower()
    assert "prompt" not in definition.lower()
    assert "signature" not in definition.lower()


@pytest.mark.unit
def test_105_pins_lifecycle_and_output_shape_and_has_rollback() -> None:
    sql = _MIGRATION.read_text(encoding="utf-8")
    rollback = _ROLLBACK.read_text(encoding="utf-8")

    assert (
        "'VERIFIED', 'STAGED', 'PUBLISHING', 'CLAIMED', 'PUBLISHED_UNKNOWN', 'TERMINAL'"
        in sql
    )
    assert "outbox_topic = expected_output_topic" in sql
    assert "outbox_event_class = expected_output_event_class" in sql
    assert "outbox_event_index = expected_output_event_index" in sql
    assert "OLD.state = 'VERIFIED' AND NEW.state = 'STAGED'" in sql
    assert "OLD.state = 'STAGED' AND NEW.state = 'PUBLISHING'" in sql
    assert (
        "OLD.state = 'PUBLISHING' AND NEW.state IN ('PUBLISHED_UNKNOWN', 'CLAIMED')"
        in sql
    )
    assert "OLD.state = 'PUBLISHED_UNKNOWN' AND NEW.state = 'CLAIMED'" in sql
    assert "OLD.state = 'CLAIMED' AND NEW.state = 'TERMINAL'" in sql
    assert "causal projection is immutable" in sql
    assert "staged binding is immutable" in sql
    assert (
        "DROP TABLE IF EXISTS public.first_effect_verified_grant_ledger RESTRICT"
        in rollback
    )


@pytest.mark.unit
def test_106_aligns_only_the_signed_rsd_v2_constraints_with_safe_rollback() -> None:
    sql = _RSD_V2_MIGRATION.read_text(encoding="utf-8")
    rollback = _RSD_V2_ROLLBACK.read_text(encoding="utf-8")

    assert "DROP CONSTRAINT ck_verified_first_effect_retry_disposition" in sql
    assert "'never-republish-after-ambiguous.v1'" in sql
    assert "'forbidden'" in sql
    assert "^[a-z][a-z0-9-]*(\\.[a-z][a-z0-9-]*)+$" in sql
    assert "cannot roll back migration 106" in rollback
    assert "retry_disposition = 'forbidden'" in rollback
