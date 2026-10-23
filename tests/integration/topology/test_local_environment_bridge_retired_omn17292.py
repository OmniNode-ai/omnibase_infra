# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The pinned omnimarket contract replaced the local_environment bridge.

OMN-19988 vendored node_projection_local_environment ahead of the omnimarket
node contract and carried a supplemental declaration until the pin advanced
past it. Retiring that bridge removes only the hand-authored declaration: the
vendored migration lineage and the writer grant in every shipped instance must
survive. This proof uses committed repository state and needs no foreign
checkout.
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
    "docker/migrations/forward/nodes/node_projection_local_environment"
)
_INSTANCES_DIR = _REPO_ROOT / "src" / "omnibase_infra" / "topology" / "instances"
_SHIPPED_INSTANCES = tuple(sorted(path.stem for path in _INSTANCES_DIR.glob("*.yaml")))
_RELATION = "local_environment"
_SCHEMA = "public"
_PRINCIPAL = "tenant_projection_writer"
_REQUIRED_PRIVILEGES = frozenset({"SELECT", "INSERT", "UPDATE"})
_VENDORED_MIGRATIONS = (
    "0000_create_local_environment.sql",
    "0001_grant_tenant_projection_writer_local_environment.sql",
    "0002_force_rls_local_environment.sql",
)


def _instance_document(profile: str) -> dict[str, Any]:
    path = _INSTANCES_DIR / f"{profile}.yaml"
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path} is not a topology mapping"
    return document


def test_shipped_instances_are_discovered() -> None:
    assert _SHIPPED_INSTANCES, f"no shipped topology instance under {_INSTANCES_DIR}"


def test_no_supplemental_bridge_remains() -> None:
    carried = {
        declaration.table.name for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
    }
    assert _RELATION not in carried, (
        f"{_RELATION} still has a supplemental declaration, although the pinned "
        "omnimarket contract now declares it"
    )


@pytest.mark.parametrize("filename", _VENDORED_MIGRATIONS)
def test_vendored_migration_lineage_remains(filename: str) -> None:
    migration = _NODE_MIGRATIONS / filename
    assert (_REPO_ROOT / migration).is_file(), (
        f"{migration} is missing; retiring {_RELATION}'s bridge must leave its "
        "vendored migration in the tree"
    )


@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
def test_shipped_instance_retains_exact_writer_privileges(profile: str) -> None:
    application = _instance_document(profile)["databases"]["application"]
    grants = application["principals"][_PRINCIPAL]["grants"]
    matching = [
        grant
        for grant in grants
        if grant["object_type"] == "TABLE"
        and grant["schema"] == _SCHEMA
        and _RELATION in grant["objects"]
    ]
    assert len(matching) == 1, (
        f"{profile} ships {len(matching)} TABLE grants for {_SCHEMA}.{_RELATION} "
        f"to {_PRINCIPAL}; expected exactly one"
    )
    granted = frozenset(matching[0]["privileges"])
    assert granted == _REQUIRED_PRIVILEGES, (
        f"{profile} grants {sorted(granted)} on {_SCHEMA}.{_RELATION} to "
        f"{_PRINCIPAL}; expected exactly {sorted(_REQUIRED_PRIVILEGES)}"
    )
