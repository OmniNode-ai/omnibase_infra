# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Pin the shipped view seam readings grant in instances and catalogs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_INSTANCES = sorted((_ROOT / "src/omnibase_infra/topology/instances").glob("*.yaml"))
_CATALOGS = sorted((_ROOT / "docker/catalog/database-topology").glob("*.yaml"))


def _assert_table_grant(path: Path) -> None:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    grants = (
        document.get("databases", {})
        .get("application", {})
        .get("principals", {})
        .get("tenant_projection_writer", {})
        .get("grants", [])
    )
    matching = [
        grant
        for grant in grants
        if grant.get("object_type") == "TABLE"
        and grant.get("schema") == "public"
        and "view_seam_readings" in grant.get("objects", [])
    ]
    assert len(matching) == 1, f"{path}: expected exactly one view seam readings grant"
    assert set(matching[0].get("privileges", [])) == {"SELECT", "INSERT", "UPDATE"}, (
        f"{path}: unexpected view seam readings privileges"
    )


@pytest.mark.parametrize("path", _INSTANCES, ids=[path.stem for path in _INSTANCES])
def test_view_seam_readings_instance_grant(path: Path) -> None:
    _assert_table_grant(path)


@pytest.mark.parametrize("path", _CATALOGS, ids=[path.stem for path in _CATALOGS])
def test_view_seam_readings_catalog_grant(path: Path) -> None:
    _assert_table_grant(path)
