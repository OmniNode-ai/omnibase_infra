# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""T5 deployment identity for both vendored migrations.

0007 carries the ordered models; 0009 the credential models and version.

Catch a missing or edited vendor, an absent/wrong tenant-ledger binding, and
a class other than forward-only for a backfill that adds constraints.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FORWARD = ROOT / "docker/migrations/forward"


@dataclass(frozen=True)
class Vendor:
    node: str
    filename: str
    sha256: str
    blob: str


VENDORS = (
    Vendor(
        node="node_delegation_routing_reducer",
        filename="0007_add_credential_ordered_models.sql",
        sha256="52508b6a27a8956ad0739940364377b96bb1773588974b8a026a890f75da2233",
        blob="2cb17409d7b5974a3d06bd6f732c93ffc6fb8c12",
    ),
    Vendor(
        node="node_projection_tenant_credentials",
        filename="0009_tenant_credentials_models_and_version.sql",
        sha256="b3f77ca598d34478246cd7d96ed3075289e3e942121b40296cb3f0f5c2750da9",
        blob="66acfaddab3721d4ea5e052ac291908cacd5458e",
    ),
)


@pytest.mark.parametrize("vendor", VENDORS, ids=[v.filename for v in VENDORS])
def test_vendor_is_the_exact_ordered_models_source(vendor: Vendor) -> None:
    artifact = f"nodes/{vendor.node}/{vendor.filename}"
    path = FORWARD / artifact
    assert path.is_file(), "ordered-model migration was not vendored"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == vendor.sha256
    assert (
        hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
        == vendor.blob
    )


@pytest.mark.parametrize("vendor", VENDORS, ids=[v.filename for v in VENDORS])
def test_tenant_ledger_binds_the_exact_vendor_once(vendor: Vendor) -> None:
    artifact = f"nodes/{vendor.node}/{vendor.filename}"
    lines = (
        (FORWARD / "_ledger/application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    rows = [line.split("\t") for line in lines if line.split("\t", 1)[0] == artifact]
    assert rows == [
        [
            artifact,
            f"node:{vendor.node}",
            f"node:{vendor.node}",
            "tenant",
            f"node:{vendor.node}:{vendor.filename}",
            vendor.sha256,
        ]
    ]


@pytest.mark.parametrize("vendor", VENDORS, ids=[v.filename for v in VENDORS])
def test_backfill_and_constraints_are_classified_forward_only(vendor: Vendor) -> None:
    artifact = f"nodes/{vendor.node}/{vendor.filename}"
    classes = yaml.safe_load(
        (ROOT / "config/migration_classes.yaml").read_text(encoding="utf-8")
    )
    assert classes["migrations"].get(f"forward/{artifact}") == "forward-only"


@pytest.mark.parametrize("vendor", VENDORS, ids=[v.filename for v in VENDORS])
def test_ledger_sha_is_the_vendored_file_hash(vendor: Vendor) -> None:
    """The ledger row's sha256 is the vendored file's own hash, not just a constant."""
    artifact = f"nodes/{vendor.node}/{vendor.filename}"
    data = (FORWARD / artifact).read_bytes()
    lines = (FORWARD / "_ledger/application-migrations.tsv").read_text(encoding="utf-8")
    rows = [line.split("\t") for line in lines.splitlines()]
    [row] = [row for row in rows if row[0] == artifact]
    assert row[-1] == hashlib.sha256(data).hexdigest()
