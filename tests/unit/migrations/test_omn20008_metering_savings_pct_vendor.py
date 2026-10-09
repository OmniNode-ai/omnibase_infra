# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20008 vendor identity for the metering-summary savings-share migration.

The file is a byte-identical copy of omnimarket's node migration. The pinned
digest below was read from omnimarket at
``7482758914cf21c6f4d79029a4ce7bc693f54042``, at
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
_NAME = "0004_metering_summary_savings_pct.sql"
_SHA256 = "38b9be4c32b267ba09d970616047014eaa70b52e56f1fc9756e178c5b27f6152"
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


def test_0004_adds_one_nullable_column_with_no_default() -> None:
    sql = _vendored().read_text(encoding="utf-8")
    assert "ADD COLUMN IF NOT EXISTS savings_pct_of_counterfactual TEXT;" in sql
    assert sql.count("ADD COLUMN") == 1
    assert "NOT NULL" not in sql
    assert "DEFAULT" not in sql
