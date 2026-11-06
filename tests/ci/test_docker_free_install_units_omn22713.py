# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22713: no declared install puts a Docker-only unit on a Docker-free Mac."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "deploy" / "unit-drift-manifest.yaml"
WATCHER_PLIST = REPO_ROOT / "scripts" / "launchd" / "com.omninode.monitor-logs.plist"


@pytest.mark.unit
def test_log_watcher_launchd_plist_is_not_shipped() -> None:
    assert not WATCHER_PLIST.exists()


@pytest.mark.unit
def test_drift_manifest_declares_no_launchd_plist_unit() -> None:
    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    plists = [u["name"] for u in manifest["units"] if u["name"].endswith(".plist")]
    assert plists == []
