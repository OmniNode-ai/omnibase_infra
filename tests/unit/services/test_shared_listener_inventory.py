# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A readiness probe or an http_mount needs an inventory entry (OMN-22377, A2)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / "docs" / "decisions" / "shared_listener_inventory.md"
SRC = ROOT / "src"
PROBE_CALL = re.compile(r"^\s*\w[\w.]*\.register_readiness_probe\(", re.MULTILINE)
MOUNT_BLOCK = re.compile(r"^http_mount:", re.MULTILINE)


def _documented(kind: str) -> dict[str, int]:
    entries: dict[str, int] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(rf"- {kind}: (\S+)(?: x(\d+))?", line)
        if match:
            entries[match.group(1)] = int(match.group(2) or 1)
    return entries


def test_shared_listener_inventory_lists_every_readiness_probe() -> None:
    found = {
        str(path.relative_to(ROOT)): len(PROBE_CALL.findall(text))
        for path in SRC.rglob("*.py")
        if (text := path.read_text(encoding="utf-8")) and PROBE_CALL.search(text)
    }
    assert found, "positive control: the scan must see the known probe sites"
    assert _documented("probe-site") == found


def test_shared_listener_inventory_lists_every_http_mount() -> None:
    found = {
        str(path.relative_to(ROOT)): 1
        for path in SRC.rglob("contract.yaml")
        if MOUNT_BLOCK.search(path.read_text(encoding="utf-8"))
    }
    assert found, "positive control: the scan must see the known mount"
    assert _documented("http-mount") == found
