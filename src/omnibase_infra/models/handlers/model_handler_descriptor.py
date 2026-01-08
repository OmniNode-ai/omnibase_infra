# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Descriptor Model for Filesystem Discovery.

This module provides ModelHandlerDescriptor, which represents a discovered handler
from a contract file. Part of OMN-1097 HandlerContractSource filesystem discovery.

The descriptor captures essential metadata about a handler discovered from a
handler_contract.yaml file, suitable for registration and validation.

See Also:
    - HandlerContractSource: Source that discovers these descriptors
    - ModelHandlerContract: Contract model from omnibase_core
    - ModelContractDiscoveryResult: Result model with graceful error handling

.. versionadded:: 0.6.2
    Created as part of OMN-1097 filesystem handler discovery.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Type alias for valid ONEX handler kinds
LiteralHandlerKind = Literal["compute", "effect", "reducer", "orchestrator"]


class ModelHandlerDescriptor(BaseModel):
    """Handler descriptor model representing a discovered handler.

    This model captures the essential metadata about a handler discovered
    from a contract file, suitable for registration and validation.

    Attributes:
        handler_id: Unique identifier for the handler.
        name: Human-readable name for the handler.
        version: Semantic version string.
        handler_kind: Handler kind (compute, effect, reducer, orchestrator).
        input_model: Fully qualified input model class path.
        output_model: Fully qualified output model class path.
        description: Optional description of the handler.
        contract_path: Path to the source contract file.

    Example:
        >>> descriptor = ModelHandlerDescriptor(
        ...     handler_id="auth.handler",
        ...     name="Auth Handler",
        ...     version="1.0.0",
        ...     handler_kind="compute",
        ...     input_model="auth.models.AuthInput",
        ...     output_model="auth.models.AuthOutput",
        ... )

    .. versionadded:: 0.6.2
        Created as part of OMN-1097 filesystem handler discovery.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    handler_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the handler",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name for the handler",
    )
    version: str = Field(
        ...,
        min_length=1,
        pattern=r"^\d+\.\d+\.\d+(-[a-zA-Z0-9.]+)?(\+[a-zA-Z0-9.]+)?$",
        description="Semantic version string (e.g., '1.0.0', '1.0.0-beta.1', '2.1.0+build.123')",
    )
    handler_kind: LiteralHandlerKind = Field(
        ...,
        description="Handler architectural kind (compute, effect, reducer, orchestrator)",
    )
    input_model: str = Field(
        ...,
        min_length=3,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$",
        description="Fully qualified input model class path (e.g., 'myapp.models.User')",
    )
    output_model: str = Field(
        ...,
        min_length=3,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$",
        description="Fully qualified output model class path (e.g., 'myapp.models.Result')",
    )
    description: str | None = Field(
        default=None,
        description="Optional description of the handler",
    )
    contract_path: str | None = Field(
        default=None,
        description="Path to the source contract file",
    )


__all__ = ["LiteralHandlerKind", "ModelHandlerDescriptor"]
