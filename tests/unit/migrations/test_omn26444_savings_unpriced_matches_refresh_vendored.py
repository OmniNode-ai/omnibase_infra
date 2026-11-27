# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26444: migration 102 is vendored as declared.

Failure modes this delivery check catches:
- the migration is absent from the infrastructure forward tree;
- the vendored SQL file differs by even one byte from its approved source,
  omnimarket#6629 at 5b056878ad221aecc47dc99a2c89ac7de8e813f1;
- its exact tenant ledger row is missing, duplicated, or has a wrong field;
- its forward-only migration classification is missing or changed;
- it stops granting tenant_projection_writer the table the topology declares.

This test checks delivery metadata and bytes only; SQL behavior is covered by
the source repository's PostgreSQL tests.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FORWARD = ROOT / "docker/migrations/forward"
RELATIVE = (
    "nodes/node_projection_savings/102_delegation_savings_unpriced_matches_refresh.sql"
)
NODE = "node:node_projection_savings"
# SHA-256 digest of the approved omnimarket source migration.
DIGEST = "699950784b59f19483ed26a6f68117f3292fd7a461a9e55efe9b8a3a20d445ba"


def test_migration_102_is_byte_identical_ledgered_and_forward_only() -> None:
    migration = FORWARD / RELATIVE
    assert migration.is_file(), f"missing vendored migration: {RELATIVE}"
    assert hashlib.sha256(migration.read_bytes()).hexdigest() == DIGEST

    ledger = FORWARD / "_ledger/application-migrations.tsv"
    rows = [
        line.split("\t")
        for line in ledger.read_text().splitlines()
        if line.split("\t")[0] == RELATIVE
    ]
    assert rows == [
        [RELATIVE, NODE, NODE, "tenant", f"{NODE}:{Path(RELATIVE).name}", DIGEST]
    ]

    classes = yaml.safe_load((ROOT / "config/migration_classes.yaml").read_text())
    flat: dict[str, str] = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str):
                    flat[str(key)] = value
                walk(value)

    walk(classes)
    assert flat.get(f"forward/{RELATIVE}") == "forward-only"


def test_migration_102_delivers_the_writer_grant_on_savings_priced_models() -> None:
    sql = (FORWARD / RELATIVE).read_text(encoding="utf-8")
    assert (
        "GRANT SELECT, INSERT, UPDATE ON public.savings_priced_models "
        "TO tenant_projection_writer;" in sql
    )
    assert (
        "GRANT DELETE ON public.savings_priced_models "
        "TO tenant_projection_writer;" in sql
    )
