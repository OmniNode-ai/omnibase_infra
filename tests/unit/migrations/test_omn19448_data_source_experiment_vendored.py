# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19448: delegation_events accepts experiment data, vendored and bound.

omnimarket#4343 at dd4d9349e7cf adds experiment to the data_source check in
node_projection_delegation's 0064 without scanning or rewriting existing rows.
This is its vendored copy. Failure modes this is written against:

- the vendored file drifts from the omnimarket source (its checksum changes);
- the file has no ledger row, or a row that names another checksum or domain;
- the file is not declared forward-only;
- the file fails to replace the check with experiment allowed and NOT VALID,
  or rewrites existing rows.
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
    "nodes/node_projection_delegation/0064_delegation_events_data_source_experiment.sql"
)
NODE = "node:node_projection_delegation"
# The checksum of the omnimarket source this file is copied from.
SOURCE_SHA256 = "834545f00a20b83cf817282cdff9f05921797df13b375493cfbeefb3fa42277f"


def test_0064_is_the_vendored_forward_only_manifest_entry() -> None:
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
    assert classes["migrations"][f"forward/{RELATIVE}"] == "forward-only"


def _statements(sql: str) -> str:
    """The SQL with comments removed, so prose cannot satisfy or trip a check."""
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def test_0064_replaces_the_data_source_check_and_rewrites_no_row() -> None:
    sql = _statements((FORWARD / RELATIVE).read_text())
    assert re.search(
        r"ALTER\s+TABLE\s+delegation_events\s+"
        r"DROP\s+CONSTRAINT\s+IF\s+EXISTS\s+delegation_events_data_source_check\s*;",
        sql,
    )
    assert re.search(
        r"ALTER\s+TABLE\s+delegation_events\s+"
        r"ADD\s+CONSTRAINT\s+delegation_events_data_source_check\s+"
        r"CHECK\s*\(\s*data_source\s+IN\s*\(\s*"
        r"'real'\s*,\s*'fixture'\s*,\s*'experiment'\s*\)\s*\)\s+NOT\s+VALID\s*;",
        sql,
    )
    assert not re.search(r"\b(UPDATE|DELETE|INSERT)\b", sql, re.IGNORECASE)
