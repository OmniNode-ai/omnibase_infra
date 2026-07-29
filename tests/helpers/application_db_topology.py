# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed application-database targets shared by projection wiring tests."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProjectionDatabaseTarget,
    _resolve_projection_database_target,
)

_TOPOLOGY_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "application_relation_ownership"
    / "topology.yaml"
)


@lru_cache(maxsize=1)
def application_topology() -> ModelDeploymentTopology:
    return ModelDeploymentTopology.from_yaml(_TOPOLOGY_PATH)


def projection_database_target(
    *table_names: str,
    schema: str = "tenant",
    physical_database: str = "omnidash_analytics",
) -> ProjectionDatabaseTarget:
    names = table_names or ("projection_fixture",)
    tables = tuple(
        ModelDbTableDeclaration(
            name=name,
            database_ref="application",
            schema=schema,
            migration=f"tests/{name}.sql",
            access="read_write",
            role=f"{name}_projection",
        )
        for name in names
    )
    topology = application_topology()
    if physical_database != topology.databases["application"].physical_name:
        database = topology.databases["application"].model_copy(
            update={"physical_name": physical_database}
        )
        topology = topology.model_copy(update={"databases": {"application": database}})
    return _resolve_projection_database_target(tables, topology)


__all__ = ["application_topology", "projection_database_target"]
