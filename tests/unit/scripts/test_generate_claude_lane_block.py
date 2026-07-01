# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/generate_claude_lane_block.py (OMN-13034).

Pins that the generated CLAUDE.md lane-table block:
  - contains the GENERATED_LANE_TABLE BEGIN/END delimiters
  - includes a verified: line derived from the snapshot emitted_at
  - lists every lane from the manifest
  - degrades gracefully when no snapshot is present
  - update_claude_md replaces an existing block in-place
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "generate_claude_lane_block.py"
_MANIFEST_PATH = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"
_SNAPSHOT_PATH = _REPO / "deploy" / "lane-census" / "census-snapshot.json"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("generate_claude_lane_block", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _minimal_manifest(lanes: list[str]) -> dict:
    return {
        "schema_version": "1.0.0",
        "lanes": {
            lane: {
                "compose_project": f"omnibase-infra-{lane}",
                "network": f"omnibase-infra-{lane}-network",
                "services": [
                    {"name": f"{lane}-redpanda", "kind": "service", "replicas": 1},
                ],
            }
            for lane in lanes
        },
    }


def _minimal_snapshot(emitted_at: str = "2026-06-17T13:00:00+00:00") -> dict:
    return {
        "schema_version": "1.0.0",
        "emitted_at": emitted_at,
        "lanes_checked": ["prod", "stability-test"],
        "drift_count": 0,
        "findings": [],
    }


# ---------------------------------------------------------------------------
# Block structure
# ---------------------------------------------------------------------------


def test_block_has_begin_end_delimiters() -> None:
    manifest = _minimal_manifest(["prod", "stability-test"])
    block = MOD.generate_block(
        manifest, None, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )
    assert "<!-- GENERATED_LANE_TABLE BEGIN" in block
    assert "<!-- GENERATED_LANE_TABLE END -->" in block


def test_block_contains_all_lanes() -> None:
    manifest = _minimal_manifest(["prod", "stability-test", "judge"])
    block = MOD.generate_block(
        manifest, None, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )
    assert "| prod" in block
    assert "| stability-test" in block
    assert "| judge" in block


def test_block_with_snapshot_contains_verified_line() -> None:
    manifest = _minimal_manifest(["prod"])
    snapshot = _minimal_snapshot("2026-06-17T09:00:00+00:00")
    block = MOD.generate_block(
        manifest, snapshot, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )
    assert (
        "verified: 2026-06-17T09:00:00+00:00 via lane-census-check.sh on 192.168.86.201"
        in block
    )


def test_block_without_snapshot_shows_missing_warning() -> None:
    manifest = _minimal_manifest(["prod"])
    block = MOD.generate_block(
        manifest, None, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )
    assert "MISSING" in block


def test_block_includes_generated_timestamp() -> None:
    manifest = _minimal_manifest(["prod"])
    now = datetime(2026, 6, 17, 12, 0, 0, tzinfo=UTC)
    block = MOD.generate_block(manifest, None, now=now)
    assert "generated: 2026-06-17T12:00:00Z" in block


# ---------------------------------------------------------------------------
# Lane port map
# ---------------------------------------------------------------------------


def test_block_includes_port_for_dev() -> None:
    manifest = _minimal_manifest(["dev"])
    # dev is special — uses optional+generated compose
    manifest["lanes"]["dev"]["optional"] = True
    block = MOD.generate_block(
        manifest, None, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )
    assert "`8085`" in block


def test_block_includes_port_for_prod() -> None:
    manifest = _minimal_manifest(["prod"])
    block = MOD.generate_block(
        manifest, None, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )
    assert "`28085`" in block


def test_block_unknown_lane_gets_dash_ports() -> None:
    manifest = _minimal_manifest(["custom-lane"])
    block = MOD.generate_block(
        manifest, None, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )
    assert "| custom-lane |" in block


# ---------------------------------------------------------------------------
# update_claude_md in-place replacement
# ---------------------------------------------------------------------------


def test_update_claude_md_replaces_existing_block() -> None:
    old_content = (
        "# CLAUDE.md\n\n"
        "Some preamble.\n\n"
        "<!-- GENERATED_LANE_TABLE BEGIN\n"
        "     generated: 2026-01-01T00:00:00Z\n"
        "-->\n"
        "| Lane | old data |\n"
        "<!-- GENERATED_LANE_TABLE END -->\n\n"
        "Some postamble.\n"
    )
    manifest = _minimal_manifest(["prod"])
    snapshot = _minimal_snapshot("2026-06-17T13:00:00+00:00")
    new_block = MOD.generate_block(
        manifest, snapshot, now=datetime(2026, 6, 17, 13, 0, tzinfo=UTC)
    )

    with tempfile.NamedTemporaryFile(
        suffix=".md", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(old_content)
        path = Path(f.name)

    result = MOD.update_claude_md(path, new_block)
    assert result is True

    updated = path.read_text(encoding="utf-8")
    assert "old data" not in updated
    assert "| prod" in updated
    assert "2026-06-17T13:00:00+00:00" in updated
    assert "Some preamble." in updated
    assert "Some postamble." in updated
    path.unlink(missing_ok=True)


def test_update_claude_md_no_block_returns_false(capsys: Any) -> None:
    content = "# CLAUDE.md\n\nNo generated block here.\n"
    manifest = _minimal_manifest(["prod"])
    new_block = MOD.generate_block(
        manifest, None, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )

    with tempfile.NamedTemporaryFile(
        suffix=".md", mode="w", encoding="utf-8", delete=False
    ) as f:
        f.write(content)
        path = Path(f.name)

    result = MOD.update_claude_md(path, new_block)
    assert result is False
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Real manifest + snapshot smoke test
# ---------------------------------------------------------------------------


def test_real_manifest_generates_valid_block() -> None:
    """The committed lane-manifest.yaml must produce a valid block."""
    assert _MANIFEST_PATH.exists(), "deploy/lane-census/lane-manifest.yaml missing"
    with open(_MANIFEST_PATH, encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    snapshot: dict | None = None
    if _SNAPSHOT_PATH.exists():
        with open(_SNAPSHOT_PATH, encoding="utf-8") as fh:
            snapshot = json.load(fh)

    block = MOD.generate_block(
        manifest, snapshot, now=datetime(2026, 6, 17, 12, 0, tzinfo=UTC)
    )

    assert "<!-- GENERATED_LANE_TABLE BEGIN" in block
    assert "<!-- GENERATED_LANE_TABLE END -->" in block
    # Every declared lane must appear
    for lane in manifest.get("lanes", {}):
        assert f"| {lane}" in block
