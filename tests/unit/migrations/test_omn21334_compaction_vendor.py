# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the barrier migrations to the measured omnimarket bytes.

The barrier digests were measured from the OMN-21334 omnimarket worktree at
026846208b5b5c0cf8f2ea51639cc5e3874364e4. Source and vendor were compared
byte for byte before pinning. These checks cover deployment identity, ledger
checksums, migration classes and the absence of a FORCE fence.
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
        "node_projection_claude_hook_events/0008_add_producer_sequence.sql",
        "cc816ad877ebbecb6db91474c33d152e667c2df0779adc7f7eb9ea70784566ca",
        "expand-only",
    ),
    (
        "node_projection_session_content/0002_add_session_barrier_stamps.sql",
        "0a0ca09e9c147efa4c05fa030b32184779734068681834c7555be3fa0fcb942d",
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


def test_none_of_the_two_migrations_is_fenced() -> None:
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    for path, _, _ in _VENDORED:
        node, name = path.split("/")
        assert f"node:{node}:{name}" not in ids
