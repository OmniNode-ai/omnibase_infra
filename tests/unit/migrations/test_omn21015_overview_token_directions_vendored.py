# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-21015: migration 099 is vendored as declared.

Failure modes this delivery check catches:

- the migration is absent from the infrastructure forward tree;
- the vendored SQL file differs by even one byte from its approved source;
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
    "nodes/node_projection_savings/099_savings_overview_token_directions.sql": (
        "node:node_projection_savings",
        "f0c50314b5e0f4ed58361b41ceb22def397a9dd3c31ee12c1659b27df5796450",
    ),
}


@pytest.mark.parametrize("relative", sorted(VENDORED))
def test_overview_token_directions_view_is_byte_identical_and_forward_only(
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
