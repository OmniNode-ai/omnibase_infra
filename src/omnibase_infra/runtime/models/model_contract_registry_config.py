# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Registry Configuration Model.

This module provides the Pydantic model for contract registry event processing
configuration, controlling the staleness tick timer behavior.

Related:
    - OMN-1869: Wire ServiceKernel to Kafka event bus
    - ContractRegistrationEventRouter: Uses this config for tick interval
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__: list[str] = ["ModelContractRegistryConfig"]


class ModelContractRegistryConfig(BaseModel):
    """Configuration for contract registry event processing.

    Controls the behavior of the contract registry staleness tick timer
    and whether contract registry processing is enabled.

    Attributes:
        tick_interval_seconds: Interval for staleness tick in seconds (min 5s)
        enabled: Whether contract registry processing is enabled
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tick_interval_seconds: int = Field(
        default=60,
        ge=5,
        description="Interval for staleness tick in seconds (min 5s)",
    )
    enabled: bool = Field(
        default=True,
        description="Whether contract registry processing is enabled",
    )
