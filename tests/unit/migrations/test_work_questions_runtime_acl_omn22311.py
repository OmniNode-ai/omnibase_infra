# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The ACL repair is delivered by the production migration ledger."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT = (
    "nodes/node_projection_work_questions/0001_revoke_undeclared_runtime_grants.sql"
)


def test_acl_repair_revokes_only_the_five_tables_and_checks_before_commit() -> None:
    sql = (_ROOT / "docker/migrations/forward" / _ARTIFACT).read_text()
    sql = "\n".join(line for line in sql.splitlines() if not line.startswith("--"))
    tables = {
        "work_outcome_units",
        "orchestration_style_outcomes",
        "lane_drift_and_stalls",
        "brief_quality_vs_outcome",
        "subagent_waste",
    }
    revoke = re.search(
        r"REVOKE ALL PRIVILEGES ON\s+(.*?)FROM omninode_runtime;", sql, re.S
    )
    assert revoke is not None
    assert {part.strip() for part in revoke[1].split(",")} == {
        f"omninode_internal.{table}" for table in tables
    }
    assert set(re.findall(r"'([a-z_]+)'", sql)) == tables | {
        "omninode_internal",
        "omninode_runtime",
    }
    assert "a.grantee = 'omninode_runtime'::regrole" in sql
    assert "1 / (count(*) = 0)::int" in sql
    assert (
        sql.index("BEGIN;")
        < sql.index("REVOKE")
        < sql.index("SELECT")
        < sql.index("COMMIT;")
    )
    assert not re.search(r"\b(GRANT|DROP|DELETE|TRUNCATE)\b", sql)


def test_acl_repair_is_bound_to_its_bytes_and_node_stream() -> None:
    forward = _ROOT / "docker/migrations/forward"
    migration = forward / _ARTIFACT
    digest = hashlib.sha256(migration.read_bytes()).hexdigest()
    rows = [
        line.split("\t")
        for line in (forward / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.split("\t", 1)[0] == _ARTIFACT
    ]
    assert rows == [
        [
            _ARTIFACT,
            "node:node_projection_work_questions",
            "node:node_projection_work_questions",
            "omninode_internal",
            "node:node_projection_work_questions:0001_revoke_undeclared_runtime_grants.sql",
            digest,
        ]
    ]
    classes = yaml.safe_load((_ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{_ARTIFACT}"] == "forward-only"
