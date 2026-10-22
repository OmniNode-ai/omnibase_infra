# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17882 migration 099 is vendored as declared.

Failure modes this delivery check catches:

- migration 099 is missing from the infrastructure forward tree;
- a vendored byte differs from the approved omnimarket source;
- the exact tenant ledger row is missing, duplicated, or has a wrong field;
- the migration is missing its forward-only classification.

This test checks delivery metadata and bytes only; SQL behavior is proved by
the source repository's PostgreSQL tests.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FORWARD = ROOT / "docker/migrations/forward"

RELATIVE = "nodes/node_projection_savings/099_savings_series_tenant_bucket.sql"
NODE = "node:node_projection_savings"
EXPECTED_SHA256 = "7d8d13f32f2e800059a2a3b0407564891c47f7cfbbe740489e3bf56afaee17ee"


def test_savings_series_migration_is_byte_identical_and_forward_only() -> None:
    """The vendored bytes, exact tenant ledger identity, and class match source."""
    migration = FORWARD / RELATIVE
    assert migration.is_file(), f"missing vendored migration: {RELATIVE}"

    digest = hashlib.sha256(migration.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA256

    ledger = FORWARD / "_ledger/application-migrations.tsv"
    rows = [
        line.split("\t")
        for line in ledger.read_text().splitlines()
        if line.split("\t")[0] == RELATIVE
    ]
    assert rows == [
        [RELATIVE, NODE, NODE, "tenant", f"{NODE}:{Path(RELATIVE).name}", digest]
    ]

    classes_path = ROOT / "config/migration_classes.yaml"
    classes = yaml.safe_load(classes_path.read_text())
    assert classes["migrations"][f"forward/{RELATIVE}"] == "forward-only"
