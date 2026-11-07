# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22648: the bounded quality-gate failure list, vendored and bound.

omnimarket#5170 (OMN-22573) replaces projection_delegation_quality_gate in
node_projection_delegation's 0065 so failure_categories holds at most 50
families of 120 characters plus one folded row. This is its vendored copy.
Failure modes this is written against:

- the vendored file drifts from the omnimarket source (its checksum changes);
- the file has no ledger row, or a row that names another checksum or domain;
- the file is not declared forward-only;
- the file stops bounding the list, drops the invoker rights, or rewrites rows.
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
    "nodes/node_projection_delegation/0065_quality_gate_failure_categories_bounded.sql"
)
NODE = "node:node_projection_delegation"
# The checksum of the omnimarket source this file is copied from.
SOURCE_SHA256 = "24366a4524d8f8d0fe26e246e4860720a21b5695f1a02d80289f183234a2d2ff"


def test_0065_is_the_vendored_forward_only_manifest_entry() -> None:
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


def test_0065_bounds_the_failure_list_and_keeps_invoker_rights() -> None:
    sql = _statements((FORWARD / RELATIVE).read_text())
    assert re.search(
        r"CREATE\s+OR\s+REPLACE\s+VIEW\s+projection_delegation_quality_gate\s+AS",
        sql,
    )
    assert re.search(r"LEFT\s*\(\s*quality_gate_detail\s*,\s*120\s*\)", sql)
    assert re.search(r"family_rank\s*<=\s*50\b", sql)
    assert re.search(r"family_rank\s*>\s*50\b", sql)
    assert re.search(
        r"ALTER\s+VIEW\s+projection_delegation_quality_gate\s+"
        r"SET\s*\(\s*security_invoker\s*=\s*true\s*\)\s*;",
        sql,
    )
    assert not re.search(r"\b(UPDATE|DELETE|INSERT)\b", sql, re.IGNORECASE)
