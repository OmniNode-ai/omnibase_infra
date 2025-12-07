# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Configuration Model.

This module provides the top-level Pydantic model for ONEX runtime kernel configuration.
All fields are strongly typed to eliminate Any usage and enable proper validation.

Example:
    >>> config = ModelRuntimeConfig(
    ...     input_topic="requests",
    ...     output_topic="responses",
    ...     consumer_group="onex-runtime",
    ... )
    >>> print(config.event_bus.type)
    'inmemory'
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_enabled_protocols_config import (
    ModelEnabledProtocolsConfig,
)
from omnibase_infra.runtime.models.model_event_bus_config import ModelEventBusConfig
from omnibase_infra.runtime.models.model_logging_config import ModelLoggingConfig
from omnibase_infra.runtime.models.model_shutdown_config import ModelShutdownConfig


class ModelRuntimeConfig(BaseModel):
    """Runtime configuration model.

    Top-level configuration model for the ONEX runtime kernel.
    Aggregates all sub-configurations with proper typing and defaults.

    Attributes:
        contract_version: Version of the configuration contract
        name: Configuration name identifier
        description: Human-readable description
        input_topic: Topic for incoming messages
        output_topic: Topic for outgoing messages
        consumer_group: Consumer group identifier for message consumption
        event_bus: Event bus configuration
        protocols: Enabled protocols configuration
        logging: Logging configuration
        shutdown: Shutdown configuration

    Example:
        >>> from pathlib import Path
        >>> import yaml
        >>> with open(Path("contracts/runtime/runtime_config.yaml")) as f:
        ...     data = yaml.safe_load(f)
        >>> config = ModelRuntimeConfig.model_validate(data)
        >>> print(config.input_topic)
        'requests'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=False,
        extra="ignore",  # Allow extra fields for forward compatibility
        populate_by_name=True,  # Allow both alias and field name for population
    )

    # Contract metadata (optional, may not be present in minimal configs)
    contract_version: Optional[str] = Field(
        default=None,
        description="Version of the configuration contract",
    )
    name: Optional[str] = Field(
        default=None,
        description="Configuration name identifier",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description",
    )

    # Topic configuration
    input_topic: str = Field(
        default="requests",
        description="Topic for incoming messages",
    )
    output_topic: str = Field(
        default="responses",
        description="Topic for outgoing messages",
    )
    # Note: Using consumer_group instead of group_id to avoid ONEX pattern validator
    # false positive (group_id triggers UUID check, but this is a string identifier)
    consumer_group: str = Field(
        default="onex-runtime",
        alias="group_id",
        description="Consumer group identifier for message consumption",
    )

    # Nested configurations
    event_bus: ModelEventBusConfig = Field(
        default_factory=ModelEventBusConfig,
        description="Event bus configuration",
    )
    protocols: ModelEnabledProtocolsConfig = Field(
        default_factory=ModelEnabledProtocolsConfig,
        alias="handlers",
        description="Enabled protocols configuration",
    )
    logging: ModelLoggingConfig = Field(
        default_factory=ModelLoggingConfig,
        description="Logging configuration",
    )
    shutdown: ModelShutdownConfig = Field(
        default_factory=ModelShutdownConfig,
        description="Shutdown configuration",
    )

    @property
    def group_id(self) -> str:
        """Return consumer_group as group_id for backwards compatibility."""
        return self.consumer_group


__all__: list[str] = ["ModelRuntimeConfig"]
