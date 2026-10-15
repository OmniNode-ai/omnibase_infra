# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-20717: the unpriced-cost migrations ship vendored and checksum-bound.

Failure modes this is written against:

- a vendored file drifts from the omnimarket source (its checksum changes);
- a file has no ledger row, or a row that names another checksum;
- a file is not declared forward-only;
- 0061 lifts and restores the table's forced row-level security, which the
  forward runner refuses for a migration outside the operator fence manifest,
  or restates tenant_isolation, whose uuid predicate breaks the shape-drift
  corpus on a still-text tenant_id;
- 0061's backfill helper policy is not dropped in the same block.
"""

import hashlib
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FORWARD = ROOT / "docker/migrations/forward"

# The checksums of the omnimarket sources these files are copied from.
VENDORED = {
    "nodes/node_projection_delegation/0061_delegation_events_cost_usd_nullable.sql": (
        "node:node_projection_delegation",
        "c784d907eba59133e902c7f192a314035f56f8a2e66f9bd4e8806f71fefc8d95",
    ),
    "nodes/node_projection_delegation/0062_delegation_token_usage_unpriced_calls.sql": (
        "node:node_projection_delegation",
        "0512d4962588a0a03d32996ebf859d297c933bbe17a4f462877fa44e55ab1729",
    ),
    "nodes/node_projection_savings/096_delegation_savings_unpriced_calls.sql": (
        "node:node_projection_savings",
        "8364fb62069d8ecb2cc7f6bf4a6767e600f6300d61d285b2e7057962538dd60e",
    ),
}
MIGRATION_0061 = FORWARD / next(iter(VENDORED))


@pytest.mark.parametrize("relative", sorted(VENDORED))
def test_each_file_is_the_vendored_forward_only_manifest_entry(relative: str) -> None:
    """Changing the SQL, omitting its ledger row or misclassifying it fails."""
    node, expected = VENDORED[relative]
    digest = hashlib.sha256((FORWARD / relative).read_bytes()).hexdigest()
    assert digest == expected
    rows = [
        line.split("\t")
        for line in (FORWARD / "_ledger/application-migrations.tsv")
        .read_text()
        .splitlines()
        if line.split("\t")[0] == relative
    ]
    assert rows == [
        [relative, node, node, "tenant", f"{node}:{Path(relative).name}", digest]
    ]
    classes = yaml.safe_load((ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"][f"forward/{relative}"] == "forward-only"


def _statements(sql: str) -> str:
    """The SQL with comments removed, so prose cannot satisfy or trip a check."""
    return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())


def test_0061_backfills_under_a_dropped_helper_policy_and_leaves_rls_alone() -> None:
    """0061 never toggles forcing or restates the tenant policy; its helper goes."""
    sql = _statements(MIGRATION_0061.read_text())
    assert "ALTER COLUMN cost_usd DROP NOT NULL" in sql
    assert re.search(r"WHERE\s+cost_measurement_source\s*=\s*'no_cost_model'", sql)
    assert not re.search(r"FORCE\s+ROW\s+LEVEL\s+SECURITY", sql, re.IGNORECASE)
    assert not re.search(r"POLICY\s+tenant_isolation", sql, re.IGNORECASE)
    created = sql.index("CREATE POLICY omn20717_unpriced_backfill")
    dropped = sql.rindex("DROP POLICY omn20717_unpriced_backfill")
    assert created < sql.index("UPDATE delegation_events") < dropped
