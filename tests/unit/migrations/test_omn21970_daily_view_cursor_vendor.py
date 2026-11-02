# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21970 deployment identity for the per-day view and the walk cursor.

Catch a missing or edited vendor of node_projection_delegation 0065 (the
delegation_events walk cursor) or 0066 (the per-day view), an absent or wrong
tenant-ledger binding, a class other than forward-only, and a topology that
does not let the delegation writer read the new view.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
NODE = "node_projection_delegation"
FORWARD = ROOT / "docker/migrations/forward"
VENDORED = {
    "0065_delegation_events_projection_cursor.sql": (
        "e94046838ad555f063a6fbe524f6b033868be28ceae4014f85e022140231de2f",
        "6856eeccfe4af74462e2f25ad347ad94952f3eb2",
    ),
    "0066_delegation_daily_view.sql": (
        "c5ef1655d66bd0ce386a76e6a1307bb0ee9fad13dccf156f3cf252fc28fb8edb",
        "28fb12081de5e66c544b7ff58b3012e22a1bb79f",
    ),
}


@pytest.mark.parametrize("filename", sorted(VENDORED))
def test_vendor_is_the_exact_omnimarket_source(filename: str) -> None:
    sha256, blob = VENDORED[filename]
    path = FORWARD / "nodes" / NODE / filename
    assert path.is_file(), f"{filename} was not vendored"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha256
    assert (
        hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()
        == blob
    )


@pytest.mark.parametrize("filename", sorted(VENDORED))
def test_tenant_ledger_binds_the_exact_vendor_once(filename: str) -> None:
    artifact = f"nodes/{NODE}/{filename}"
    data = (FORWARD / artifact).read_bytes()
    lines = (
        (FORWARD / "_ledger/application-migrations.tsv")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    rows = [line.split("\t") for line in lines if line.split("\t", 1)[0] == artifact]
    assert rows == [
        [
            artifact,
            f"node:{NODE}",
            f"node:{NODE}",
            "tenant",
            f"node:{NODE}:{filename}",
            hashlib.sha256(data).hexdigest(),
        ]
    ]


@pytest.mark.parametrize("filename", sorted(VENDORED))
def test_each_migration_is_classified_forward_only(filename: str) -> None:
    classes = yaml.safe_load(
        (ROOT / "config/migration_classes.yaml").read_text(encoding="utf-8")
    )
    assert (
        classes["migrations"].get(f"forward/nodes/{NODE}/{filename}") == "forward-only"
    )


INSTANCES = sorted((ROOT / "src/omnibase_infra/topology/instances").glob("*.yaml"))
VIEW = "projection_delegation_daily"
WRITER = "tenant_projection_writer"


def _table_grants(instance: Path) -> dict[str, list[dict[str, object]]]:
    """Each application principal's TABLE grant blocks in one shipped instance."""
    data = yaml.safe_load(instance.read_text(encoding="utf-8"))
    principals = data["databases"]["application"]["principals"]
    return {
        name: [g for g in (spec.get("grants") or []) if g.get("object_type") == "TABLE"]
        for name, spec in principals.items()
    }


def test_every_topology_instance_is_checked() -> None:
    assert len(INSTANCES) >= 3, "positive control: the shipped instances are found"


@pytest.mark.parametrize("instance", INSTANCES, ids=lambda path: path.stem)
def test_the_delegation_writer_may_select_the_per_day_view(instance: Path) -> None:
    grants = _table_grants(instance)
    blocks = [g for g in grants[WRITER] if VIEW in (g.get("objects") or [])]
    assert len(blocks) == 1, f"{WRITER} names {VIEW} in {len(blocks)} TABLE blocks"
    assert blocks[0]["schema"] == "public"
    assert blocks[0]["privileges"] == ["SELECT"], "the view is read only"


@pytest.mark.parametrize("instance", INSTANCES, ids=lambda path: path.stem)
def test_no_other_principal_is_granted_the_per_day_view(instance: Path) -> None:
    others = sorted(
        name
        for name, blocks in _table_grants(instance).items()
        if name != WRITER and any(VIEW in (g.get("objects") or []) for g in blocks)
    )
    assert others == [], f"{VIEW} is granted to {others}"
