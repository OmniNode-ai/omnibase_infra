# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the pr_merged_events unique-key node migration 0003.

The digest was measured from the omnimarket source the vendor was copied from,
node_pr_merged_projection/migrations/0003 on branch
omn-19998-pr-merged-dedupe-repo-pr (commit d346106ab). Source and vendor were
compared byte for byte before pinning. Failure modes: the vendored file drifts
from its source, loses its checksum-bound ledger row, or is declared with a class
other than the one the class checker reads from its statements.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_pr_merged_projection"
_NAME = "0003_pr_merged_events_unique_repo_pr_number.sql"
_SHA256 = "68c4220235df46969883a9febc3887cc7b867a5d18b928aaeba6cc2c72be6634"


def test_vendored_bytes_match_the_pinned_omnimarket_source() -> None:
    artifact = _FORWARD / "nodes" / _NODE / _NAME
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == _SHA256


def test_the_vendor_collapses_duplicates_before_adding_the_key() -> None:
    sql = (_FORWARD / "nodes" / _NODE / _NAME).read_text()
    assert sql.index("DELETE FROM pr_merged_events") < sql.index(
        "UNIQUE (repo, pr_number)"
    )


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
            "omninode_internal",
            f"node:{_NODE}:{_NAME}",
            _SHA256,
        ]
    ]


def test_the_vendor_is_declared_forward_only() -> None:
    # The DELETE rewrites data and the constraint is added to an existing table;
    # the class checker reads both as not provably additive.
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{_NODE}/{_NAME}"] == "forward-only"
