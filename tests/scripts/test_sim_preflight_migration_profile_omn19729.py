# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the non-mutating disposable source-shape profile verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.runtime_build.verify_sim_preflight_migration_profile import verify

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "docker" / "sim-preflight-migration-profile.json"
OVERLAY = ROOT / "docker" / "docker-compose.sim-preflight.yml"
RUNNER = ROOT / "scripts" / "run-forward-migrations.sh"


def _write_profile(tmp_path: Path, required: list[dict[str, str]]) -> Path:
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "profile": "sim-preflight-source-compatible-v1",
                "lane": "sim-preflight",
                "required_migrations": required,
            }
        ),
        encoding="utf-8",
    )
    return profile


def test_profile_declares_source_shape_outbox_and_watermark_migrations() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["lane"] == "sim-preflight"
    assert [item["id"] for item in profile["required_migrations"]] == [
        "node:node_projection_delegation:0037_delegation_events_uuid_mixed_representation_guard_before_set_role.sql",
        "node:node_projection_delegation:0048_delegation_events_caller_lane.sql",
        "docker/108_create_sim_archive_rehydration_outbox.sql",
        "docker/109_add_event_ledger_ingest_watermark.sql",
    ]
    assert "ONEX_MIGRATION_LANE: sim-preflight" in OVERLAY.read_text(encoding="utf-8")
    assert "  sim-preflight)" in RUNNER.read_text(encoding="utf-8")


def test_verifier_refuses_absent_or_modified_declared_migrations(
    tmp_path: Path,
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    required = [
        {
            "id": "docker/109_add_event_ledger_ingest_watermark.sql",
            "path": "109_add_event_ledger_ingest_watermark.sql",
            "sha256": hashlib.sha256(b"known bytes").hexdigest(),
        },
        {
            "id": "node:one",
            "path": "one.sql",
            "sha256": hashlib.sha256(b"known bytes").hexdigest(),
        },
        {
            "id": "node:two",
            "path": "two.sql",
            "sha256": hashlib.sha256(b"known bytes").hexdigest(),
        },
        {
            "id": "docker/108_create_sim_archive_rehydration_outbox.sql",
            "path": "108_create_sim_archive_rehydration_outbox.sql",
            "sha256": hashlib.sha256(b"known bytes").hexdigest(),
        },
    ]
    profile = _write_profile(tmp_path, required)
    with pytest.raises(ValueError, match="absent"):
        verify(profile, migrations)


def test_verifier_accepts_complete_exact_corpus(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    nested = migrations / "nodes" / "node_projection_delegation"
    nested.mkdir(parents=True)
    contents = {
        "nodes/node_projection_delegation/0037.sql": b"uuid tenant conversion",
        "nodes/node_projection_delegation/0048.sql": b"caller lane",
        "108.sql": b"rehydration outbox",
        "109.sql": b"ingest watermark",
    }
    required: list[dict[str, str]] = []
    for index, (relative_path, content) in enumerate(contents.items()):
        (migrations / relative_path).write_bytes(content)
        required.append(
            {
                "id": f"migration:{index}",
                "path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    assert verify(_write_profile(tmp_path, required), migrations) == tuple(
        item["id"] for item in required
    )


def test_verifier_checks_real_corpus_and_rejects_modified_outbox(
    tmp_path: Path,
) -> None:
    """The replay table must be present with the audited schema bytes."""
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert verify(PROFILE, ROOT / "docker" / "migrations" / "forward") == tuple(
        item["id"] for item in profile["required_migrations"]
    )
    migrations = tmp_path / "migrations"
    for item in profile["required_migrations"]:
        target = migrations / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(
            (ROOT / "docker" / "migrations" / "forward" / item["path"]).read_bytes()
        )
    (migrations / "108_create_sim_archive_rehydration_outbox.sql").write_text(
        "-- missing replay table\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"checksum differs.*108_"):
        verify(PROFILE, migrations)
