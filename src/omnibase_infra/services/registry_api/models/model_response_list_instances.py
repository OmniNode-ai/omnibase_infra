# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Response model for list_instances endpoint.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
    - PR #182: Add Pydantic response models for API endpoints
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.registry_api.models.model_registry_instance_view import (
    ModelRegistryInstanceView,
)
from omnibase_infra.services.registry_api.models.model_warning import ModelWarning


class ModelResponseListInstances(BaseModel):
    """Response model for the GET /registry/instances endpoint.

    Provides a list of live Consul service instances with optional warnings
    for partial success scenarios.

    Attributes:
        instances: List of live Consul service instances
        warnings: List of warnings for partial success scenarios
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    instances: list[ModelRegistryInstanceView] = Field(
        default_factory=list,
        description="List of live Consul service instances",
    )
    warnings: list[ModelWarning] = Field(
        default_factory=list,
        description="Warnings for partial success scenarios",
    )


__all__ = ["ModelResponseListInstances"]
