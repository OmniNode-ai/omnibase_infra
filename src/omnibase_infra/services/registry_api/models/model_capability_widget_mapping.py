# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Capability widget mapping model for dashboard configuration.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.registry_api.models.model_widget_defaults import (
    ModelWidgetDefaults,
)


class ModelCapabilityWidgetMapping(BaseModel):
    """Mapping from a capability to its widget configuration.

    Attributes:
        widget_type: Type of widget to render
        defaults: Default widget configuration
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    widget_type: str = Field(
        ...,
        description="Type of widget to render",
    )
    defaults: ModelWidgetDefaults = Field(
        default_factory=ModelWidgetDefaults,
        description="Default widget configuration",
    )


__all__ = ["ModelCapabilityWidgetMapping"]
