# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19996 deployment identity for the credential last-used migration.

Catch a missing or edited vendor, an absent or wrong tenant-ledger binding, and
a class other than expand-only for a migration that only adds a nullable column.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
NODE = "node_projection_tenant_credentials"
FILENAME = "0010_tenant_credentials_last_used_at.sql"
ARTIFACT = f"nodes/{NODE}/{FILENAME}"
SHA256 = "b3ec863feb49351e56732ea8dff2e3910ef63453c5e11ca60cd80b4138adcf13"
BLOB = "33af17028e3b3332b1b7717269674842ea206a0e"
FORWARD = ROOT / "docker/migrations/forward"


def test_vendor_is_the_exact_last_used_source() -> None:
    path = FORWARD / ARTIFACT
    assert path.is_file(), "last-used migration was not vendored"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == SHA256
    assert (
        hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
        == BLOB
    )


def test_tenant_ledger_binds_the_exact_vendor_once() -> None:
    lines = (
        (FORWARD / "_ledger/application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
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


def test_a_nullable_column_add_is_classified_expand_only() -> None:
    classes = yaml.safe_load(
        (ROOT / "config/migration_classes.yaml").read_text(encoding="utf-8")
    )
    assert classes["migrations"].get(f"forward/{ARTIFACT}") == "expand-only"


def test_ledger_sha_is_the_vendored_file_hash() -> None:
    """The ledger row's sha256 is the vendored file's own hash, not just a constant."""
    data = (FORWARD / ARTIFACT).read_bytes()
    lines = (FORWARD / "_ledger/application-migrations.tsv").read_text(encoding="utf-8")
    rows = [line.split("\t") for line in lines.splitlines()]
    [row] = [row for row in rows if row[0] == ARTIFACT]
    assert row[-1] == hashlib.sha256(data).hexdigest()
