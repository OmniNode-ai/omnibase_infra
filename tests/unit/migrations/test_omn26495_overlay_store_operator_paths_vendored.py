# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26495: the overlay store operator_paths migration, vendored and bound.

omnimarket's node_projection_overlay_store adds the operator_paths column to
omninode_internal.overlay_store in 0002. This is its vendored copy. Failure
modes this is written against:

- the vendored file is absent, so omnimarket's Sibling Release Compatibility
  Gate finds no byte-identical copy in the pinned omnibase_infra release;
- the vendored file drifts from the omnimarket source (its checksum changes);
- the file has no ledger row, or a row that names another checksum or domain;
- the file is not declared in config/migration_classes.yaml;
- the column is added NOT NULL without a default, which would fail on the rows
  already stored.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_NODE = "node_projection_overlay_store"
_NAME = "0002_add_operator_paths.sql"
_RELATIVE = f"nodes/{_NODE}/{_NAME}"
# The checksum of the omnimarket source this file is copied from.
_SOURCE_SHA256 = "64420fdaf995350ef8ec7fb423a35ff58319a9e80546364fbb778857b00a68b9"


def test_0002_is_vendored() -> None:
    assert (_FORWARD / _RELATIVE).is_file()


def test_0002_bytes_and_manifest_binding_are_exact() -> None:
    digest = hashlib.sha256((_FORWARD / _RELATIVE).read_bytes()).hexdigest()
    assert digest == _SOURCE_SHA256
    rows = [
        row.split("\t")
        for row in (_FORWARD / "_ledger" / "application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
        if row.split("\t", 1)[0] == _RELATIVE
    ]
    assert rows == [
        [
            _RELATIVE,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "omninode_internal",
            f"node:{_NODE}:{_NAME}",
            _SOURCE_SHA256,
        ]
    ]


def test_0002_is_a_declared_migration_class() -> None:
    classes = yaml.safe_load(
        (_ROOT / "config" / "migration_classes.yaml").read_text(encoding="utf-8")
    )["migrations"]
    assert classes[f"forward/{_RELATIVE}"] in {"expand-only", "forward-only"}


def test_0002_adds_a_defaulted_json_column_to_the_overlay_store() -> None:
    sql = re.sub(r"--[^\n]*", "", (_FORWARD / _RELATIVE).read_text(encoding="utf-8"))
    assert "ALTER TABLE omninode_internal.overlay_store" in sql
    assert "ADD COLUMN IF NOT EXISTS operator_paths JSON NOT NULL DEFAULT '[]'" in sql
