# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A readiness probe or an http_mount needs an inventory entry (OMN-22377, A2)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "decisions" / "shared_listener_inventory.yaml"
SRC = ROOT / "src"
PROBE_CALL = re.compile(r"^\s*\w[\w.]*\.register_readiness_probe\(", re.MULTILINE)
MOUNT_BLOCK = re.compile(r"^http_mount:", re.MULTILINE)


def _documented(section: str) -> dict[str, int]:
    document = yaml.safe_load(DOC.read_text(encoding="utf-8"))
    entries = document[section]
    if section == "probe_sites":
        return {path: int(entry["count"]) for path, entry in entries.items()}
    return dict.fromkeys(entries, 1)


def test_shared_listener_inventory_lists_every_readiness_probe() -> None:
    found = {
        str(path.relative_to(ROOT)): len(PROBE_CALL.findall(text))
        for path in SRC.rglob("*.py")
        if (text := path.read_text(encoding="utf-8")) and PROBE_CALL.search(text)
    }
    assert found, "positive control: the scan must see the known probe sites"
    assert _documented("probe_sites") == found


def test_shared_listener_inventory_lists_every_http_mount() -> None:
    found = {
        str(path.relative_to(ROOT)): 1
        for path in SRC.rglob("contract.yaml")
        if MOUNT_BLOCK.search(path.read_text(encoding="utf-8"))
    }
    assert found, "positive control: the scan must see the known mount"
    assert _documented("http_mounts") == found
