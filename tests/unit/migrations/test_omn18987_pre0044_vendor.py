# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18987 vendor identity and immutable migration checks.

The 0043z preflight predecessor is vendored again here. It was removed by
OMN-18693 because it reads ``platform_catalog.schema_migrations``, which no
ownership manifest the OMN-15361 application-database SQL gate reads carried a
declaration for -- so the gate refused it and no rerun could change that. The
owner is now declared, exactly once, in omninode_infra
``k8s/migrations/application-relation-ownership.yaml`` under authority
``service:omnibase_infra_node_migration_runner``, the omnibase_infra migration
runner whose own ``_ledger/bootstrap.sql`` creates and writes that relation.

The bytes are omnimarket's, copied rather than retyped: this file's vendored
sha256 and the manifest row below both pin
``_PRECHECK_SHA256``, which equals the sha256 of the source migration at
omnimarket#2770. The dynamic-SQL probes the gate also refused were rewritten
static in omnimarket first, so the vendored copy carries that rewrite.
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
_PRECHECK_SHA256 = "9ce5a0d5f17d8082023343e0a57abed8815913ad17788c39e5ade741dac3d278"
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
