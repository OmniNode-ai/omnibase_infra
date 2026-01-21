# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Widget mapping model for complete dashboard configuration.

Related Tickets:
    - OMN-1278: Contract-Driven Dashboard - Registry Discovery
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.registry_api.models.model_capability_widget_mapping import (
    ModelCapabilityWidgetMapping,
)


class ModelWidgetMapping(BaseModel):
    """Complete widget mapping configuration.

    Loaded from widget_mapping.yaml, provides the mapping from
    capabilities and semantic roles to widget types.

    Attributes:
        version: Configuration version for compatibility checking
        capability_mappings: Map of capability tags to widget configs
        semantic_mappings: Map of semantic roles to widget configs
        fallback: Default widget config when no mapping matches
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = Field(
        ...,
        description="Configuration version",
    )
    capability_mappings: dict[str, ModelCapabilityWidgetMapping] = Field(
        default_factory=dict,
        description="Map of capability tags to widget configs",
    )
    semantic_mappings: dict[str, ModelCapabilityWidgetMapping] = Field(
        default_factory=dict,
        description="Map of semantic roles to widget configs",
    )
    fallback: ModelCapabilityWidgetMapping = Field(
        ...,
        description="Default widget config when no mapping matches",
    )


__all__ = ["ModelWidgetMapping"]
