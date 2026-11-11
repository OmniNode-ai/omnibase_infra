# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the goal-trace migrations to the measured omnimarket bytes (OMN-20034).

The digests were measured from the omnimarket branch carrying node_projection_goal_trace
at commit cd37ae71d5270a45c8ffcc7fe6b3451039c7d721. These checks cover deployment
identity, ledger checksums, migration classes and the absence of a fence.
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
        "node_projection_goal_trace/0000_create_goal_trace_events.sql",
        "44eb77617d5adfffabdcb0273ac4eacd1ae3af204da44bb6b66da09d0abe2e85",
        "forward-only",
    ),
    (
        "node_projection_goal_trace/0001_grant_omninode_runtime_goal_trace.sql",
        "e5cf4212a07d1b9d7fa07b83113834cedcf66a14dd4c43d1605e1fedcf215692",
        "expand-only",
    ),
]


@pytest.mark.parametrize(("path", "sha256", "klass"), _VENDORED)
def test_vendored_bytes_match_the_pinned_omnimarket_source(
    path: str, sha256: str, klass: str
) -> None:
    assert (
        hashlib.sha256((_FORWARD / "nodes" / path).read_bytes()).hexdigest() == sha256
    )


@pytest.mark.parametrize(("path", "sha256", "klass"), _VENDORED)
def test_the_vendor_has_exactly_one_checksum_bound_ledger_row(
    path: str, sha256: str, klass: str
) -> None:
    node, name = path.split("/")
    artifact = f"nodes/{path}"
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
            "omninode_internal",
            f"node:{node}:{name}",
            sha256,
        ]
    ]


@pytest.mark.parametrize(("path", "sha256", "klass"), _VENDORED)
def test_the_vendor_is_declared_with_its_class(
    path: str, sha256: str, klass: str
) -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{path}"] == klass


def test_neither_migration_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    for path, _, _ in _VENDORED:
        node, name = path.split("/")
        assert f"node:{node}:{name}" not in ids
