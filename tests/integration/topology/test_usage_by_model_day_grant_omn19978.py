# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The usage-by-model-day bridges derive the grants they ship (OMN-19978).

omnibase_infra vendors node_projection_usage_by_model_day 0000 and 0001 ahead of
omnimarket#3073. Until the omnimarket pin carries that node contract, two
interim declarations in table_grant_derivation keep both relations derivable.
This proves the real derivation and each shipped instance give
tenant_projection_writer exactly SELECT, INSERT and UPDATE on each relation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    derive_table_grants,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELATIONS = ("usage_by_model_day", "usage_by_model_day_calls")
_SCHEMA = "public"
_PRINCIPAL = "tenant_projection_writer"
_REQUIRED_PRIVILEGES = frozenset({"INSERT", "SELECT", "UPDATE"})
_SHIPPED_INSTANCES = ("local", "onex-dev", "onex-prod")


def _derived_privileges(profile: str, relation: str) -> frozenset[str]:
    derived = derive_table_grants(
        load_topology_profile(profile), LEGACY_MIGRATION_TABLE_DECLARATIONS
    )
    matching = tuple(
        grant
        for grant in derived.grants.get(_PRINCIPAL, ())
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and grant.schema == _SCHEMA
        and relation in grant.objects
    )
    assert len(matching) == 1, (
        f"real derivation for {profile} produced {len(matching)} {_PRINCIPAL} "
        f"TABLE grants for {_SCHEMA}.{relation}, not one. Keep exactly one "
        "bridge declaration and regenerate the topology grants."
    )
    return frozenset(privilege.value for privilege in matching[0].privileges)


def _shipped_grants(profile: str, relation: str) -> list[dict[str, Any]]:
    path = (
        _REPO_ROOT
        / "src"
        / "omnibase_infra"
        / "topology"
        / "instances"
        / f"{profile}.yaml"
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    principal = document["databases"]["application"]["principals"][_PRINCIPAL]
    return [
        grant
        for grant in principal["grants"]
        if grant.get("object_type") == "TABLE"
        and grant.get("schema") == _SCHEMA
        and relation in tuple(grant.get("objects") or ())
    ]


@pytest.mark.parametrize("relation", _RELATIONS)
@pytest.mark.parametrize("profile", _SHIPPED_INSTANCES)
class TestTheUsageByModelDayBridgesProduceTheShippedGrant:
    def test_real_derivation_grants_exact_writer_privileges(
        self, profile: str, relation: str
    ) -> None:
        assert _derived_privileges(profile, relation) == _REQUIRED_PRIVILEGES

    def test_shipped_instance_grants_exact_writer_privileges(
        self, profile: str, relation: str
    ) -> None:
        matching = _shipped_grants(profile, relation)
        assert len(matching) == 1, (
            f"{profile} ships {len(matching)} {_PRINCIPAL} TABLE grants for "
            f"{_SCHEMA}.{relation}, not one. Regenerate the instance."
        )
        assert frozenset(matching[0]["privileges"]) == _REQUIRED_PRIVILEGES
