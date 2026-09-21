# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19013 vendor identity for the terminal-construction metrics migration."""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_VENDOR = _FORWARD / "nodes" / "node_projection_delegation"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_FILENAME = "0045_terminal_construction_outcome_metrics.sql"
_SHA256 = "389f85226ac6192e00949e5b6b052a8f78b17a1925d1c1ff471b9ee0e77bf897"


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


def test_vendor_bytes_and_manifest_binding_are_exact() -> None:
    artifact_path = f"nodes/node_projection_delegation/{_FILENAME}"
    assert hashlib.sha256((_VENDOR / _FILENAME).read_bytes()).hexdigest() == _SHA256
    assert _manifest_rows()[artifact_path] == [
        artifact_path,
        "node:node_projection_delegation",
        "node:node_projection_delegation",
        "tenant",
        f"node:node_projection_delegation:{_FILENAME}",
        _SHA256,
    ]
