# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23169: the delegation summary view that leaves experiment runs out, vendored.

omnimarket's node_projection_delegation 0068 replaces projection_delegation_summary
so a run typed experiment (a synthetic run) is left out of the run, quality, token,
spend and savings totals. This is its vendored copy. Failure modes this is written
against:

- the vendored file drifts from the omnimarket source (its checksum changes);
- the file has no ledger row, or a row that names another checksum or domain;
- the file is not declared forward-only;
- the file stops excluding experiment rows, drops the invoker rights, or rewrites rows.
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
    "nodes/node_projection_delegation/0068_delegation_summary_excludes_experiment.sql"
)
NODE = "node:node_projection_delegation"
# The checksum of the omnimarket source this file is copied from.
SOURCE_SHA256 = "da43281cc16cc5f9e5cec5301b65eefa858abe3d66656fc604f125448f7c04b6"


def test_0068_is_the_vendored_forward_only_manifest_entry() -> None:
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


def test_0068_leaves_experiment_rows_out_and_keeps_invoker_rights() -> None:
    sql = _statements((FORWARD / RELATIVE).read_text())
    assert re.search(
        r"CREATE\s+OR\s+REPLACE\s+VIEW\s+projection_delegation_summary\s+AS", sql
    )
    assert re.search(r"data_source\s*<>\s*'experiment'", sql)
    assert re.search(
        r"ALTER\s+VIEW\s+projection_delegation_summary\s+"
        r"SET\s*\(\s*security_invoker\s*=\s*true\s*\)\s*;",
        sql,
    )
    assert not re.search(r"\b(UPDATE|DELETE|INSERT)\b", sql, re.IGNORECASE)
