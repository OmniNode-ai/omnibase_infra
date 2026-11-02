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
_CATALOGS = sorted(_CATALOG_DIR.glob("*.yaml"))


def test_the_rendered_catalogs_are_present() -> None:
    assert _CATALOGS


@pytest.mark.parametrize("path", _CATALOGS, ids=lambda p: p.stem)
def test_the_rendered_catalog_grants_the_table_to_the_principal_only(
    path: Path,
) -> None:
    document = yaml.safe_load(path.read_text())
    principals = document["databases"]["omnibase_infra"]["principals"]
    for name, principal in principals.items():
        granted = any(
            _TABLE in grant.get("objects", []) for grant in principal.get("grants", [])
        )
        assert granted == (name == _PRINCIPAL)
