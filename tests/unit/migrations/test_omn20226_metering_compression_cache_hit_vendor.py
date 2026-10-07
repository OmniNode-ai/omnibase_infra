# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20226 vendor identity for the metering-summary compression and cache-hit migration.

The file is a byte-identical copy of omnimarket's node migration. The pinned
digest below was read from omnimarket at
``b86cfaed2272065a5d5abe2de6143777d60f1d9e``, at
``src/omnimarket/nodes/node_projection_metering_summary/migrations/<file>``. A
vendored file that drifts from that source, or a ledger or class row that no
longer describes it, fails here instead of on a lane runner.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker" / "migrations" / "forward"
_MANIFEST = _FORWARD / "_ledger" / "application-migrations.tsv"
_CLASSES = _ROOT / "config" / "migration_classes.yaml"

_NODE = "node_projection_metering_summary"
_NAME = "0003_metering_summary_compression_and_cache_hit.sql"
_SHA256 = "a5729fd46157f09b9e890a334a4211c1070c176185405a252fbc65c9d6b82af5"
_SCOPE = "omninode_internal"
_CLASS = "expand-only"


def _vendored() -> Path:
    return _FORWARD / "nodes" / _NODE / _NAME


def test_vendored_bytes_match_the_omnimarket_source() -> None:
    assert hashlib.sha256(_vendored().read_bytes()).hexdigest() == _SHA256


def test_the_file_has_exactly_one_exact_ledger_row() -> None:
    artifact_path = f"nodes/{_NODE}/{_NAME}"
    rows = [
        row.split("\t")
        for row in _MANIFEST.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    assert [row for row in rows if row[0] == artifact_path] == [
        [
            artifact_path,
            f"node:{_NODE}",
            f"node:{_NODE}",
            _SCOPE,
            f"node:{_NODE}:{_NAME}",
            _SHA256,
        ]
    ]


def test_the_file_has_its_migration_class() -> None:
    classes = yaml.safe_load(_CLASSES.read_text(encoding="utf-8"))["migrations"]
    assert classes[f"forward/nodes/{_NODE}/{_NAME}"] == _CLASS


def test_0003_adds_three_nullable_columns_with_no_default() -> None:
    sql = _vendored().read_text(encoding="utf-8")
    for column in (
        "compression_ratio TEXT;",
        "cache_hit_rate TEXT;",
        "runs_cache_answered INTEGER;",
    ):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in sql
    assert "NOT NULL" not in sql
    assert "DEFAULT" not in sql
