# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22675: migration 101 is vendored as declared.

Failure modes this delivery check catches:

- the migration is absent from the infrastructure forward tree;
- the vendored SQL file differs by even one byte from its approved source,
  omnimarket#5195 at 9681dee8b41dce3b97cd9827955be79919b740e9;
- its exact tenant ledger row is missing, duplicated, or has a wrong field;
- its forward-only migration classification is missing or changed.

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

# SHA-256 digest of the approved omnimarket source migration.
VENDORED = {
    "nodes/node_projection_savings/101_delegation_savings_unpriced_single_pass.sql": (
        "node:node_projection_savings",
        "c242d8f2626cbf63de22c08684ef2a6f88e49d7754899e6eb6c36d73cbddab76",
    ),
}


@pytest.mark.parametrize("relative", sorted(VENDORED))
def test_savings_unpriced_single_pass_is_byte_identical_and_forward_only(
    relative: str,
) -> None:
    """The vendored bytes, exact ledger identity and class match the source."""
    node, expected_digest = VENDORED[relative]
    migration = FORWARD / relative
    assert migration.is_file(), f"missing vendored migration: {relative}"

    digest = hashlib.sha256(migration.read_bytes()).hexdigest()
    assert digest == expected_digest

    ledger = FORWARD / "_ledger/application-migrations.tsv"
    rows = [
        line.split("\t")
        for line in ledger.read_text().splitlines()
        if line.split("\t")[0] == relative
    ]
    assert rows == [
        [relative, node, node, "tenant", f"{node}:{Path(relative).name}", digest]
    ]

    classes_path = ROOT / "config/migration_classes.yaml"
    classes = yaml.safe_load(classes_path.read_text())
    assert classes["migrations"][f"forward/{relative}"] == "forward-only"
