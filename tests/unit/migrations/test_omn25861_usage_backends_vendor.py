# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the usage node migration 0004 (OMN-25861), not page acceptance.

The digest was measured from the omnimarket source the vendor is copied from,
node_projection_usage_by_model_day/migrations/0004 at omnimarket commit
d3f010f155c0ea43edb7f011321a9c3f9e6417fe, which adds the nullable usage_by_model_day_calls.backend and
usage_by_model_day.backends columns. The usage writer stores a call's backend once
the first exists and counts a row's backends once both do. Source and vendor were compared byte for byte before
pinning. Failure modes: the vendored file drifts from its source, gives a column a
default (a manufactured value where the source stated none), loses its
checksum-bound ledger row, or is declared with a class other than the one the class
checker reads from its statements.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_projection_usage_by_model_day"
_NAME = "0004_usage_by_model_day_backends.sql"
_SHA256 = "5aeef6c0fbfefb21810276985ac27ba555ed796fa7d7408071b6e3fb3ca81cb1"


def test_vendored_bytes_match_the_pinned_omnimarket_source() -> None:
    artifact = _FORWARD / "nodes" / _NODE / _NAME
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == _SHA256


def test_the_vendor_adds_both_columns_nullable_with_no_default() -> None:
    sql = (_FORWARD / "nodes" / _NODE / _NAME).read_text()
    statements = [
        line.strip()
        for line in sql.splitlines()
        if line.strip() and not line.lstrip().startswith("--")
    ]
    assert statements == [
        "ALTER TABLE public.usage_by_model_day_calls ADD COLUMN IF NOT EXISTS backend TEXT;",
        "ALTER TABLE public.usage_by_model_day ADD COLUMN IF NOT EXISTS backends JSONB;",
    ]


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
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    key = f"forward/nodes/{_NODE}/{_NAME}"
    assert classes["migrations"][key] == "forward-only"
