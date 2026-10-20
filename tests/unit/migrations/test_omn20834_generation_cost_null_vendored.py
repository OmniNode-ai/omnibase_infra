# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20834: generation_events.cost_inference_usd takes NULL, vendored and bound.

A generation run on a model the pricing surface reports as UNKNOWN records a
null cost, and the projection stores it as NULL. omnimarket's 0063 drops the
column's NOT NULL and DEFAULT 0; this is its vendored copy. Failure modes this
is written against:

- the vendored file drifts from the omnimarket source (its checksum changes);
- the file has no ledger row, or a row that names another checksum or domain;
- the file is not declared forward-only;
- the file keeps the NOT NULL or the DEFAULT, or rewrites existing rows.
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FORWARD = ROOT / "docker/migrations/forward"
RELATIVE = (
    "nodes/node_projection_delegation/"
    "0063_generation_events_cost_inference_usd_nullable.sql"
)
NODE = "node:node_projection_delegation"
# The checksum of the omnimarket source this file is copied from.
SOURCE_SHA256 = "baf6c484400d471535117c76493bae4763da501237d58f1a504350fe2294e16a"


def test_0063_is_the_vendored_forward_only_manifest_entry() -> None:
    """Changing the SQL, omitting its ledger row or misclassifying it fails."""
    digest = hashlib.sha256((FORWARD / RELATIVE).read_bytes()).hexdigest()
    assert digest == SOURCE_SHA256
    rows = [
        line.split("\t")
        for line in (FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.split("\t")[0] == RELATIVE
    ]
    assert rows == [
        [
            RELATIVE,
            NODE,
            NODE,
            "omninode_internal",
            f"{NODE}:{Path(RELATIVE).name}",
            digest,
        ]
    ]
    classes = yaml.safe_load((ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{RELATIVE}"] == "forward-only"


def _statements(sql: str) -> str:
    """The SQL with comments removed, so prose cannot satisfy or trip a check."""
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def test_0063_drops_the_not_null_and_the_default_and_rewrites_no_row() -> None:
    sql = _statements((FORWARD / RELATIVE).read_text())
    assert re.search(r"ALTER\s+TABLE\s+generation_events\b", sql)
    assert "ALTER COLUMN cost_inference_usd DROP NOT NULL" in sql
    assert "ALTER COLUMN cost_inference_usd DROP DEFAULT" in sql
    assert not re.search(r"\b(UPDATE|DELETE|INSERT)\b", sql, re.IGNORECASE)
