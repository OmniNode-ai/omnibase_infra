# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pinned omnimarket contract replaced the delegation_eval_items bridge.

Retiring the OMN-19790 bridge removes only its hand-authored declaration. The
vendored migration lineage and the exact grants in every shipped instance must
survive. This proof uses committed repository state and needs no foreign checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODE_MIGRATIONS = Path(
    "docker/migrations/forward/nodes/node_projection_delegation_eval"
)
_MIGRATIONS = (
    "0000_create_delegation_eval_items.sql",
    "0001_grant_tenant_projection_writer_delegation_eval_items.sql",
    "0002_force_rls_delegation_eval_items.sql",
)
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")
_TABLE = "delegation_eval_items"
_SCHEMA = "public"
_PRINCIPAL = "tenant_projection_writer"
_REQUIRED_PRIVILEGES = frozenset({"SELECT", "INSERT", "UPDATE"})


def test_no_supplemental_bridge_remains() -> None:
    carried = {
        declaration.table.name for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
    }
    assert _TABLE not in carried, (
        f"{_TABLE} still has a supplemental declaration, although the pinned "
        "omnimarket contract now declares it"
    )


@pytest.mark.parametrize("filename", _MIGRATIONS)
def test_vendored_migration_lineage_remains(filename: str) -> None:
    migration = _NODE_MIGRATIONS / filename
    assert (_REPO_ROOT / migration).is_file(), (
        f"{migration} is missing; retiring {_TABLE}'s bridge must leave its "
        "vendored migration in the tree"
    )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
def test_shipped_instance_retains_exact_writer_privileges(profile: str) -> None:
    path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    )
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    grants = document["databases"]["application"]["principals"][_PRINCIPAL]["grants"]
    matching = [
        grant
        for grant in grants
        if grant["object_type"] == "TABLE"
        and grant["schema"] == _SCHEMA
        and _TABLE in grant["objects"]
    ]
    assert len(matching) == 1, (
        f"{profile} ships {len(matching)} TABLE grants for {_SCHEMA}.{_TABLE} "
        f"to {_PRINCIPAL}; expected exactly one"
    )
    assert frozenset(matching[0]["privileges"]) == _REQUIRED_PRIVILEGES
