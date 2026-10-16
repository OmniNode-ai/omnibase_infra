# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deployment identity of the pr_merged_events unique-key node migration 0003.

The source is node_pr_merged_projection/migrations/0003 on branch
omn-19998-pr-merged-dedupe-repo-pr (commit d346106ab). The deployment copy
qualifies its two catalog reads with pg_catalog. Both the deployed digest and
the original source digest are pinned, so no other source changes are accepted.
The ledger checksum and forward-only class are checked separately.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    lint_application_database_sql,
)

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[3]
_FORWARD = _ROOT / "docker/migrations/forward"
_NODE = "node_pr_merged_projection"
_NAME = "0003_pr_merged_events_unique_repo_pr_number.sql"
_SOURCE_SHA256 = "68c4220235df46969883a9febc3887cc7b867a5d18b928aaeba6cc2c72be6634"
_SHA256 = "2a1141966f9b9cdc83caf6e60d64cf866953312453d560f32efcd00a59651309"


def test_vendored_bytes_match_the_pinned_deployment_artifact() -> None:
    artifact = _FORWARD / "nodes" / _NODE / _NAME
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == _SHA256


def test_only_catalog_qualification_differs_from_the_pinned_omnimarket_source() -> None:
    sql = (_FORWARD / "nodes" / _NODE / _NAME).read_text()
    for catalog in ("pg_constraint", "pg_attribute"):
        assert sql.count(f"pg_catalog.{catalog}") == 1
        sql = sql.replace(f"pg_catalog.{catalog}", catalog)
    assert hashlib.sha256(sql.encode()).hexdigest() == _SOURCE_SHA256


def test_catalog_reads_pass_application_database_domain_enforcement() -> None:
    sql = (_FORWARD / "nodes" / _NODE / _NAME).read_text()
    assert not lint_application_database_sql(sql, load_topology_profile("local"))


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
