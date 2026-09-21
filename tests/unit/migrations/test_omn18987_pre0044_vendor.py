# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18987 vendor identity and immutable migration checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_VENDOR = _FORWARD / "nodes/node_projection_delegation"
_MANIFEST = _FORWARD / "_ledger/application-migrations.tsv"
_PRECHECK = "0043z_preflight_delegation_shadow_comparisons.sql"
_FROZEN = "0044_restore_delegation_shadow_comparisons.sql"
_PRECHECK_SHA256 = "3652f86ca33af7999d1f1b2f5c1d4d54644ff0bf342c2c5a4664f0df0d5a0700"
_FROZEN_SHA256 = "3a1089294056fafeebbe5fdbe1c0910d3dc178b37d9402e2f37c19a32161c298"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows() -> dict[str, list[str]]:
    return {
        row.split("\t", 1)[0]: row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    }


def test_vendor_bytes_and_manifest_bindings_are_exact() -> None:
    rows = _manifest_rows()
    precheck_path = f"nodes/node_projection_delegation/{_PRECHECK}"
    frozen_path = f"nodes/node_projection_delegation/{_FROZEN}"
    assert _sha256(_VENDOR / _PRECHECK) == _PRECHECK_SHA256
    assert _sha256(_VENDOR / _FROZEN) == _FROZEN_SHA256
    assert rows[precheck_path] == [
        precheck_path,
        "node:node_projection_delegation",
        "node:node_projection_delegation",
        "tenant",
        "node:node_projection_delegation:0043z_preflight_delegation_shadow_comparisons.sql",
        _PRECHECK_SHA256,
    ]
    assert rows[frozen_path] == [
        frozen_path,
        "node:node_projection_delegation",
        "node:node_projection_delegation",
        "tenant",
        "node:node_projection_delegation:0044_restore_delegation_shadow_comparisons.sql",
        _FROZEN_SHA256,
    ]


def test_predecessor_orders_before_immutable_successor() -> None:
    names = sorted(path.name for path in _VENDOR.glob("004*.sql"))
    assert names.index(_PRECHECK) < names.index(_FROZEN)
