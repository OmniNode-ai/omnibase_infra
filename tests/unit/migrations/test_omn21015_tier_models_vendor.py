# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the AC2 node migration, not page acceptance.

The digest was measured from the omnimarket source the vendor was copied from,
omnimarket commit 565f9784d. Source and vendor were compared byte for byte
before pinning. Failure modes: a vendored file drifts
from its source, loses its checksum-bound ledger row, or is declared with a
class other than the one the class checker reads from its statements.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"

_VENDORED = [
    (
        "node_projection_delegation",
        "0060_model_routing_tier_models.sql",
        "7c6307ab2cbb23537d0911a79fdb238ea72d90a8194f4f642e7d7e6570647c68",
    ),
]


@pytest.mark.parametrize(("node", "name", "sha256"), _VENDORED)
def test_vendored_bytes_match_the_pinned_omnimarket_source(
    node: str, name: str, sha256: str
) -> None:
    artifact = _FORWARD / "nodes" / node / name
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == sha256


@pytest.mark.parametrize(("node", "name", "sha256"), _VENDORED)
def test_the_vendor_has_exactly_one_checksum_bound_ledger_row(
    node: str, name: str, sha256: str
) -> None:
    artifact = f"nodes/{node}/{name}"
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
            f"node:{node}",
            f"node:{node}",
            "tenant",
            f"node:{node}:{name}",
            sha256,
        ]
    ]


@pytest.mark.parametrize(("node", "name", "sha256"), _VENDORED)
def test_the_vendor_is_declared_forward_only(node: str, name: str, sha256: str) -> None:
    # 0060 replaces a view: the class checker reads it as forward-only, so
    # expand-only would be a false rollback promise.
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{node}/{name}"] == "forward-only"
