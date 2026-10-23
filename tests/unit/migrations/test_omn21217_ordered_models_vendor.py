# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""T5 deployment identity, defined before adding the vendored migration.

Catch a missing or edited vendor, an absent/wrong tenant-ledger binding, and
an incorrect expand-only class for a backfill that adds constraints.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
NODE = "node_delegation_routing_reducer"
FILENAME = "0007_add_credential_ordered_models.sql"
ARTIFACT = f"nodes/{NODE}/{FILENAME}"
SHA256 = "8f8a591ca8969219d32e4095804f83846aa11f3e2bde8680f518b7dc1b7e1865"
BLOB = "50de5a6c37d0c6cbfb0d46c563f349ae509cbfd4"
FORWARD = ROOT / "docker/migrations/forward"


def test_vendor_is_the_exact_ordered_models_source() -> None:
    path = FORWARD / ARTIFACT
    assert path.is_file(), "ordered-model migration was not vendored"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256
    assert (
        hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
        == BLOB
    )


def test_tenant_ledger_binds_the_exact_vendor_once() -> None:
    lines = (FORWARD / "_ledger/application-migrations.tsv").read_text().splitlines()
    rows = [line.split("\t") for line in lines if line.split("\t", 1)[0] == ARTIFACT]
    assert rows == [
        [
            ARTIFACT,
            f"node:{NODE}",
            f"node:{NODE}",
            "tenant",
            f"node:{NODE}:{FILENAME}",
            SHA256,
        ]
    ]


def test_backfill_and_constraints_are_classified_forward_only() -> None:
    classes = yaml.safe_load((ROOT / "config/migration_classes.yaml").read_text())
    assert classes["migrations"].get(f"forward/{ARTIFACT}") == "forward-only"
