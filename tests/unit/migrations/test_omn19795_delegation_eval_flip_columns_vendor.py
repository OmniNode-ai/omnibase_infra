# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19795 vendor identity for the delegation_eval_results generation-stability columns.

The digest was measured from the omnimarket source the vendor was copied from
(node_projection_delegation_eval/0007). Failure modes: the vendored file drifts
from its source, loses its checksum-bound ledger row, is declared with a class
other than expand-only, or is fenced although it touches no row-level security.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_projection_delegation_eval"
_NAME = "0007_add_generation_stability_to_delegation_eval_results.sql"
_SHA256 = "0cfb689054f4c741adcd0b7b0c0413f57cf0875dc36de209597a4590443a9e1e"


def test_vendored_bytes_match_the_pinned_omnimarket_source() -> None:
    artifact = _FORWARD / "nodes" / _NODE / _NAME
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == _SHA256


def test_the_vendor_has_exactly_one_checksum_bound_ledger_row() -> None:
    artifact = f"nodes/{_NODE}/{_NAME}"
    live = hashlib.sha256((_FORWARD / artifact).read_bytes()).hexdigest()
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
            live,
        ]
    ]


def test_the_vendor_is_expand_only_and_unfenced() -> None:
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/nodes/{_NODE}/{_NAME}"] == "expand-only"
    fenced = yaml.safe_load((_FORWARD / "fenced-node-migrations.yaml").read_text())
    ids = {entry["id"] for entry in fenced["fenced_node_migrations"]}
    assert f"node:{_NODE}:{_NAME}" not in ids


def test_the_vendor_only_adds_the_four_nullable_columns() -> None:
    body = (_FORWARD / "nodes" / _NODE / _NAME).read_text()
    statements = []
    for chunk in body.split(";"):
        kept = [
            line for line in chunk.splitlines() if not line.strip().startswith("--")
        ]
        text = " ".join(" ".join(kept).split())
        if text:
            statements.append(text)
    assert statements == [
        f"ALTER TABLE public.delegation_eval_results ADD COLUMN IF NOT EXISTS {column}"
        for column in (
            "flip_rate DOUBLE PRECISION",
            "flip_rate_wilson_low DOUBLE PRECISION",
            "flip_rate_wilson_high DOUBLE PRECISION",
            "flip_measured_backend_id TEXT",
        )
    ]
