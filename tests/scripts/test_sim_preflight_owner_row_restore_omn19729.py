# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static safety contract for the private sim analytics owner-row path."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / "scripts" / "runtime_build" / "capture_sim_preflight_owner_row.sh"
RESTORE = ROOT / "scripts" / "runtime_build" / "restore_sim_preflight_owner_row.sh"


def test_owner_capture_is_one_row_read_only_binary_copy() -> None:
    raw = CAPTURE.read_text(encoding="utf-8")
    assert 'readonly ANALYTICS_DB="omnidash_analytics"' in raw
    assert 'readonly OWNER_TABLE="public.delegation_events"' in raw
    assert "source owner predicate must select exactly one row" in raw
    assert "COPY (SELECT * FROM ${OWNER_TABLE}" in raw
    assert "TO STDOUT WITH (FORMAT binary)" in raw
    assert "row_sha256" in raw
    assert "printf '%s\\n' \"owner-row-captured" in raw
    assert "SELECT * FROM public.delegation_events;" not in raw


def test_owner_restore_accepts_only_the_named_disposable_empty_relation() -> None:
    raw = RESTORE.read_text(encoding="utf-8")
    assert 'readonly TARGET_CONTAINER="omnibase-infra-sim-preflight-postgres"' in raw
    assert 'readonly TARGET_PROJECT="omnibase-infra-sim-preflight"' in raw
    assert 'readonly ANALYTICS_DB="omnidash_analytics"' in raw
    assert "com.docker.compose.project" in raw
    assert (
        "docker inspect -f '{{ index .Config.Labels \"com.docker.compose.project\" }}'"
        in raw
    )
    assert r"\"com.docker.compose.project\"" not in raw
    assert "target analytics schema does not match captured owner row" in raw
    assert "owner metadata coordinates must be UUIDs" in raw
    assert "owner metadata checksums must be lowercase SHA-256" in raw
    assert "target owner relation must be empty before narrow restore" in raw
    assert "COPY ${OWNER_TABLE} FROM STDIN WITH (FORMAT binary);" in raw
    assert "exact_count" in raw
    assert "total_count" in raw
    assert "target owner readback is not exactly the captured owner row" in raw
