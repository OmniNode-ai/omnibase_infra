# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22478: pin the overlay store migration bytes, ledger binding and table set."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_overlay_store"
_VENDOR = _FORWARD / "nodes" / _NODE
_CREATE = "0000_create_overlay_store.sql"
_SHA256 = "b7da031f1b11da3e5f7f1d60d953d610e7498422f8458e8d033607f32c867956"
_GRANT = "0001_grant_omninode_runtime_overlay_store.sql"
_GRANT_SHA256 = "796344234e9bb22f007a517ce63e7a48feb62c68d52a6aa47dd851f4039ded9d"


def test_only_the_create_and_grant_files_are_vendored() -> None:
    assert sorted(path.name for path in _VENDOR.glob("*")) == [_CREATE, _GRANT]


@pytest.mark.parametrize(
    ("name", "sha256"), [(_CREATE, _SHA256), (_GRANT, _GRANT_SHA256)]
)
def test_vendor_bytes_and_manifest_binding_are_exact(name: str, sha256: str) -> None:
    path = _VENDOR / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256
    artifact = f"nodes/{_NODE}/{name}"
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.split("\t", 1)[0] == artifact
    ]
    assert rows == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{name}",
            sha256,
        ]
    ]


def test_the_document_column_keeps_key_order() -> None:
    sql = (_VENDOR / _CREATE).read_text(encoding="utf-8")
    assert "document        JSON NOT NULL" in sql
    assert "JSONB NOT NULL" not in sql


def test_migration_classes_name_both_files() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_CREATE}"] == "forward-only"
    assert classes[f"forward/nodes/{_NODE}/{_GRANT}"] == "expand-only"
