# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the AC4 usage node migration 0003, not page acceptance.

The digest was measured from the omnimarket source the vendor was copied from,
node_projection_usage_by_model_day/migrations/0003 at omnimarket commit e0102e2ec,
the expand-only version: the three-column key is kept beside a four-column unique
index. Source and vendor were compared byte for byte before
pinning. Failure modes: the vendored file drifts from its source (the earlier
key-swapping 0003 among them), loses its checksum-bound ledger row, or is declared
with a class other than the one the class checker reads from its statements.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_projection_usage_by_model_day"
_NAME = "0003_usage_by_model_day_tier_share.sql"
_SHA256 = "b920108965aa83ec3437a4c97306e9212775a5a9b5ffd07ad9e43ad9a68f6ed5"


def test_vendored_bytes_match_the_pinned_omnimarket_source() -> None:
    artifact = _FORWARD / "nodes" / _NODE / _NAME
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == _SHA256


def test_the_vendor_keeps_the_three_column_key() -> None:
    sql = (_FORWARD / "nodes" / _NODE / _NAME).read_text()
    assert "DROP CONSTRAINT" not in sql.upper()
    assert "usage_by_model_day_tier_key" in sql


def test_the_vendor_has_exactly_one_checksum_bound_ledger_row() -> None:
    artifact = f"nodes/{_NODE}/{_NAME}"
    rows = [
        line.split("\t")
        for line in (_FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert [row for row in rows if row[0] == artifact] == [
        [
            artifact,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "tenant",
            f"node:{_NODE}:{_NAME}",
            _SHA256,
        ]
    ]


def test_the_vendor_is_declared_forward_only() -> None:
    # The four-column unique index is built on an existing table; the class checker
    # reads that as not provably additive, so expand-only would be refused.
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{_NODE}/{_NAME}"] == "forward-only"
