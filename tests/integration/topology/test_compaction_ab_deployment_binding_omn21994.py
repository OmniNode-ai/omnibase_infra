# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compaction A/B declarations reach deployment catalogs and runtime binding."""

from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_projection_database_target,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.application_database import SUPPORTED_TOPOLOGY_PROFILES
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    WRITE_PRIVILEGES,
    derive_table_grants,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_TABLES = ("compaction_ab_rows", "compaction_ab_build_series")
_PRINCIPAL = "omninode_runtime"


@pytest.mark.parametrize("profile", sorted(SUPPORTED_TOPOLOGY_PROFILES))
@pytest.mark.parametrize("table_name", _TABLES)
def test_deployment_catalog_and_runtime_binding_keep_derived_compaction_grants(
    profile: str, table_name: str
) -> None:
    (declaration,) = (
        item
        for item in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if item.table.name == table_name
    )
    topology = load_topology_profile(profile)
    derived = derive_table_grants(topology, (declaration,), database_ref="application")
    assert not derived.unmappable
    assert set(derived.grants) == {_PRINCIPAL}
    (grant,) = derived.grants[_PRINCIPAL]
    assert grant.schema == "omninode_internal"
    assert set(grant.objects) == {table_name}
    assert set(grant.privileges) == WRITE_PRIVILEGES

    catalog_path = _ROOT / "docker/catalog/database-topology" / f"{profile}.yaml"
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    principal = catalog["databases"]["application"]["principals"][_PRINCIPAL]
    (catalog_grant,) = (
        item for item in principal["grants"] if table_name in item.get("objects", [])
    )
    assert catalog_grant["object_type"] == "TABLE"
    assert catalog_grant["schema"] == grant.schema
    assert set(catalog_grant["privileges"]) == {
        privilege.value for privilege in grant.privileges
    }

    target = _resolve_projection_database_target((declaration.table,), topology)
    (resolved,) = target.table_targets
    assert resolved.table.name == table_name
    for binding in (resolved.read_binding, resolved.write_binding):
        assert binding is not None
        assert binding.principal == _PRINCIPAL

    # A missing shipped grant must fail runtime admission even while derivation
    # and the deployment catalog still contain the positive control.
    without_grant = topology.model_copy(deep=True)
    runtime = without_grant.databases["application"].principals[_PRINCIPAL]
    principals = dict(without_grant.databases["application"].principals)
    principals[_PRINCIPAL] = runtime.model_copy(
        update={
            "grants": tuple(g for g in runtime.grants if table_name not in g.objects)
        }
    )
    databases = dict(without_grant.databases)
    databases["application"] = databases["application"].model_copy(
        update={"principals": principals}
    )
    without_grant = without_grant.model_copy(update={"databases": databases})
    with pytest.raises(ValueError, match="lacks declared read privileges"):
        _resolve_projection_database_target((declaration.table,), without_grant)
