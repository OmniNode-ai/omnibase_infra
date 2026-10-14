# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the AC3 nullable columns, not monetary acceptance.

The digest was measured from the preserved omnimarket source at Git tree
97efd93ff89c071afafc54482e4cc5f0e21a78c7, under
src/omnimarket/nodes/node_projection_delegation/migrations/. The source and
vendor were compared byte for byte before pinning this deployment check.
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_projection_delegation"
_NAME = "0059_delegation_events_modelled_run_savings.sql"
_SHA256 = "fb68ed708582d20a9838ef9eeedd55778c8b677fff16a0a8786fc84402c4ffe6"
_ARTIFACT = f"nodes/{_NODE}/{_NAME}"


def test_vendored_bytes_match_the_pinned_omnimarket_source() -> None:
    assert hashlib.sha256((_FORWARD / _ARTIFACT).read_bytes()).hexdigest() == _SHA256


def test_the_vendor_has_exactly_one_checksum_bound_ledger_row() -> None:
    rows = [
        line.split("\t")
        for line in (_FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    assert [row for row in rows if row[0] == _ARTIFACT] == [
        [
            _ARTIFACT,
            f"node:{_NODE}",
            f"node:{_NODE}",
            "tenant",
            f"node:{_NODE}:{_NAME}",
            _SHA256,
        ]
    ]


def test_the_vendor_is_declared_expand_only() -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{_ARTIFACT}"] == "expand-only"


def test_only_the_four_nullable_text_columns_are_added() -> None:
    statements = "\n".join(
        line
        for line in (_FORWARD / _ARTIFACT).read_text().splitlines()
        if not line.lstrip().startswith("--")
    )
    assert re.findall(r"ADD COLUMN IF NOT EXISTS (\w+) (\w+)", statements) == [
        ("modelled_savings_usd", "TEXT"),
        ("savings_baseline_model", "TEXT"),
        ("savings_pricing_manifest_version", "TEXT"),
        ("savings_measurement", "TEXT"),
    ]
    for forbidden in ("NOT NULL", "DEFAULT", "DROP", "UPDATE", "DELETE", "TRUNCATE"):
        assert not re.search(rf"\b{forbidden}\b", statements.upper())
