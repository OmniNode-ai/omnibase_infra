# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18693 vendor identity and immutable migration checks.

The 0043z preflight predecessor is NOT vendored here. It reads
``platform_catalog.schema_migrations``, for which no ownership manifest the
OMN-15361 application-database SQL gate reads carries a declaration, so the
gate refuses it and no rerun can change that. Vendoring the immutable 0044
restore on its own is what omnimarket#2699 -- which changes 0044 and nothing
else -- needs its vendor-parity gate to resolve against.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_VENDOR = _FORWARD / "nodes/node_projection_delegation"
_MANIFEST = _FORWARD / "_ledger/application-migrations.tsv"
_PRECHECK = "0043z_preflight_delegation_shadow_comparisons.sql"
_FROZEN = "0044_restore_delegation_shadow_comparisons.sql"
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
    frozen_path = f"nodes/node_projection_delegation/{_FROZEN}"
    assert _sha256(_VENDOR / _FROZEN) == _FROZEN_SHA256
    assert rows[frozen_path] == [
        frozen_path,
        "node:node_projection_delegation",
        "node:node_projection_delegation",
        "tenant",
        "node:node_projection_delegation:0044_restore_delegation_shadow_comparisons.sql",
        _FROZEN_SHA256,
    ]


def test_the_refused_preflight_predecessor_is_not_vendored() -> None:
    """A red test the day 0043z is vendored without its ownership declaration.

    Deleting this assertion instead of declaring an owner for
    ``platform_catalog.schema_migrations`` reintroduces the exact gate failure
    this PR removed, so the assertion is the reminder, not a preference.
    """
    assert not (_VENDOR / _PRECHECK).exists()
    assert f"nodes/node_projection_delegation/{_PRECHECK}" not in _manifest_rows()
