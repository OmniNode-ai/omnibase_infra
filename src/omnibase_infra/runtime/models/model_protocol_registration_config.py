# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Protocol Registration Configuration Model.

This module provides the Pydantic model for individual protocol registration
configuration within contract-driven runtime setup.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.model_enabled_protocols_config import ProtocolName


class ModelProtocolRegistrationConfig(BaseModel):
    """Protocol registration configuration model.

    Defines configuration for a single protocol registration as specified
    in runtime contracts.

    Attributes:
        type: Protocol type identifier (e.g., "http", "db", "kafka")
        protocol_class: Fully qualified protocol class name for instantiation
        enabled: Whether this protocol is enabled for registration
        options: Additional protocol-specific configuration options

    Example:
        >>> config = ModelProtocolRegistrationConfig(
        ...     type="http",
        ...     protocol_class="omnibase_infra.handlers.handler_http.HttpHandler",
        ...     enabled=True,
        ...     options={"timeout": "30"},
        ... )
        >>> print(config.type)
        'http'
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        populate_by_name=True,  # Allow both alias and field name
        from_attributes=True,  # Support pytest-xdist compatibility
    )

    type: ProtocolName = Field(
        description="Protocol type identifier (e.g., 'http', 'db', 'kafka')",
    )
    protocol_class: str | None = Field(
        default=None,
        alias="class",
        description="Fully qualified protocol class name for instantiation",
    )
    enabled: bool = Field(
        default=True,
        description="Whether this protocol is enabled for registration",
    )
    options: dict[str, str] = Field(
        default_factory=dict,
        description="Additional protocol-specific configuration options",
    )


__all__: list[str] = ["ModelProtocolRegistrationConfig"]
