# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The metering_summary and lab_proof_receipts bridges were retired (OMN-17292).

The omnimarket contract pin advancing to 532ec1f53834 carries omnimarket#3079
and omnimarket#3043, so the pinned contracts now declare both relations. The
hand-authored ``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries are deleted; the
vendored migration lineage and the exact shipped grants must survive.
This proof uses committed repository state alone.
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
_MIGRATIONS_ROOT = Path("docker/migrations/forward/nodes")
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})

# (relation, schema, principal, node package, migrations)
_RETIRED_BRIDGES = (
    (
        "metering_summary",
        "public",
        "tenant_projection_writer",
        "node_projection_metering_summary",
        (
            "0000_create_metering_summary.sql",
            "0001_grant_tenant_projection_writer_metering_summary.sql",
        ),
    ),
    (
        "lab_proof_receipts",
        "omninode_internal",
        "omninode_runtime",
        "node_projection_lab_proof_receipts",
        (
            "0000_create_lab_proof_receipts.sql",
            "0001_grant_omninode_runtime_lab_proof_receipts.sql",
        ),
    ),
)


def _committed_privileges(
    profile: str, relation: str, schema: str, principal: str
) -> frozenset[str]:
    path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    )
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    granted: set[str] = set()
    for database in document.get("databases", {}).values():
        entry = database.get("principals", {}).get(principal)
        if entry is None:
            continue
        for grant in entry.get("grants", ()):
            if (
                grant.get("object_type") == "TABLE"
                and grant.get("schema") == schema
                and relation in tuple(grant.get("objects") or ())
            ):
                granted.update(grant.get("privileges", ()))
    return frozenset(granted)


@pytest.mark.parametrize(
    ("relation", "schema", "principal", "package", "migrations"),
    _RETIRED_BRIDGES,
    ids=[bridge[0] for bridge in _RETIRED_BRIDGES],
)
class TestTheBridgeWasRetired:
    def test_no_supplemental_bridge_remains(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
    ) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        assert relation not in carried

    def test_the_migration_lineage_is_still_in_the_tree(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
    ) -> None:
        for name in migrations:
            assert (_REPO_ROOT / _MIGRATIONS_ROOT / package / name).is_file()

    @pytest.mark.parametrize("instance", _SHIPPED_INSTANCES)
    def test_the_shipped_grant_survived(
        self,
        relation: str,
        schema: str,
        principal: str,
        package: str,
        migrations: tuple[str, ...],
        instance: str,
    ) -> None:
        granted = _committed_privileges(instance, relation, schema, principal)
        assert granted >= _REQUIRED_PRIVILEGES
