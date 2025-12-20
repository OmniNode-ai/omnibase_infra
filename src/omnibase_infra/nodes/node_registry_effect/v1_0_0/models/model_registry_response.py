# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registry response model for node registration operations."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.nodes.node_registry_effect.v1_0_0.models.model_consul_operation_result import (
    ModelConsulOperationResult,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models.model_node_registration import (
    ModelNodeRegistration,
)
from omnibase_infra.nodes.node_registry_effect.v1_0_0.models.model_postgres_operation_result import (
    ModelPostgresOperationResult,
)


class ModelRegistryResponse(BaseModel):
    """Response model for registry operations."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    operation: Literal["register", "deregister", "discover", "request_introspection"]
    success: bool
    status: Literal["success", "partial", "failed"]
    consul_result: ModelConsulOperationResult | None = None
    postgres_result: ModelPostgresOperationResult | None = None
    nodes: list[ModelNodeRegistration] | None = None
    error: str | None = None
    processing_time_ms: float
    correlation_id: UUID
