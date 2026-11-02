# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rendered database-topology catalogs carry the routing-decisions grant (OMN-21935)."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_CATALOG_DIR = (
    Path(__file__).resolve().parents[3] / "docker" / "catalog" / "database-topology"
)
_TABLE = "infra_routing_decisions"
_PRINCIPAL = "role_omnibase_infra"
_PRIVILEGES = {"INSERT", "SELECT", "UPDATE"}
_CATALOGS = sorted(_CATALOG_DIR.glob("*.yaml"))


def test_the_rendered_catalogs_are_present() -> None:
    assert _CATALOGS


@pytest.mark.parametrize("path", _CATALOGS, ids=lambda p: p.stem)
def test_the_rendered_catalog_grants_the_table_to_the_principal_only(
    path: Path,
) -> None:
    document = yaml.safe_load(path.read_text())
    database = document["databases"].get("omnibase_infra")
    assert database is not None, f"{path.name} renders no omnibase_infra database"
    principals = database["principals"]
    assert _PRINCIPAL in principals
    for name, principal in principals.items():
        granted = [
            grant
            for grant in principal.get("grants", [])
            if _TABLE in grant.get("objects", [])
        ]
        if name != _PRINCIPAL:
            assert not granted
            continue
        (grant,) = granted
        assert grant["object_type"] == "TABLE"
        assert grant["schema"] == "public"
        assert set(grant["privileges"]) == _PRIVILEGES
