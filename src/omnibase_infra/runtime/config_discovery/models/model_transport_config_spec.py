# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Transport configuration specification model.

Defines the mapping between a transport type and its expected configuration
keys in Infisical. Each spec represents a single transport's config needs,
including the Infisical folder path and the set of expected keys.

.. versionadded:: 0.10.0
    Created as part of OMN-2287.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumInfraTransportType


class ModelTransportConfigSpec(BaseModel):
    """Specification for a transport type's configuration in Infisical.

    Attributes:
        transport_type: The infrastructure transport type.
        infisical_folder: The Infisical folder path where config lives.
            Shared config: ``/shared/<transport>/``
            Per-service: ``/services/<service>/<transport>/``
        keys: Expected configuration key names at this path.
        required: Whether this transport's config is required for startup.
        service_slug: If set, indicates this is per-service config.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    transport_type: EnumInfraTransportType = Field(
        ..., description="Infrastructure transport type."
    )
    infisical_folder: str = Field(
        ..., description="Infisical folder path for this transport's secrets."
    )
    keys: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Expected configuration key names.",
    )
    required: bool = Field(
        default=False,
        description="Whether this transport config is required for startup.",
    )
    service_slug: str = Field(
        default="",
        description="Service slug for per-service config. Empty for shared.",
    )
