# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-21206: migrations 097 and 098 are vendored as declared.

Failure modes this delivery check catches:

- either migration is absent from the infrastructure forward tree;
- a vendored SQL file differs by even one byte from its approved source;
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

# SHA-256 digests of the two approved omnimarket source migrations.
VENDORED = {
    "nodes/node_projection_savings/097_savings_overview_all_runs.sql": (
        "node:node_projection_savings",
        "920506d0fa24de62fee31341bba986da13485531a2c80263f17db7b93e7c90d1",
    ),
    "nodes/node_projection_savings/098_delegation_savings_measured_population.sql": (
        "node:node_projection_savings",
        "4a818362a4784690d38b8e832bdc99c7a94eef149f3e5e8c31c42706eba30cc2",
    ),
}


@pytest.mark.parametrize("relative", sorted(VENDORED))
def test_each_savings_view_is_byte_identical_and_forward_only(relative: str) -> None:
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
