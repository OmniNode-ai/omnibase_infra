# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18534: the delegation turn count column, vendored and bound.

omnimarket's node_projection_delegation adds the nullable turn_count column to
delegation_events in 0069. This is its vendored copy. Failure modes this is
written against:

- the vendored file drifts from the omnimarket source (its checksum changes);
- the file has no ledger row, or a row that names another checksum or domain;
- the file is not declared expand-only;
- the file adds the column NOT NULL or with a default, which would report a
  turn count for rows that never recorded one, or rewrites existing rows.
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FORWARD = ROOT / "docker/migrations/forward"
RELATIVE = "nodes/node_projection_delegation/0069_delegation_events_turn_count.sql"
NODE = "node:node_projection_delegation"
# The checksum of the omnimarket source this file is copied from.
SOURCE_SHA256 = "43a0a993223d7c71e392860334aaa9c47a81b1bf7afadd7f4884112dd2d4c0e3"


def test_0069_is_the_vendored_expand_only_manifest_entry() -> None:
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
            "tenant",
            f"{NODE}:{Path(RELATIVE).name}",
            digest,
        ]
    ]
    classes = yaml.safe_load((ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{RELATIVE}"] == "expand-only"


def _statements(sql: str) -> str:
    """The SQL with comments removed, so prose cannot satisfy or trip a check."""
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def test_0069_adds_one_nullable_column_and_rewrites_no_row() -> None:
    sql = _statements((FORWARD / RELATIVE).read_text())
    assert re.search(
        r"ALTER\s+TABLE\s+delegation_events\s+"
        r"ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+turn_count\s+INTEGER\s*;",
        sql,
    )
    assert not re.search(r"\b(NOT\s+NULL|DEFAULT)\b", sql, re.IGNORECASE)
    assert not re.search(r"\b(UPDATE|DELETE|INSERT)\b", sql, re.IGNORECASE)
