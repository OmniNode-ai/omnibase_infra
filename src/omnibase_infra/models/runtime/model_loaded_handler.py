# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Loaded Handler Model for Handler Plugin Loader.

This module provides ModelLoadedHandler, which represents a handler that has been
successfully discovered, validated, and registered by the HandlerPluginLoader.

The model captures essential metadata about loaded handlers for runtime tracking,
discovery filtering, and diagnostic purposes.

See Also:
    - HandlerPluginLoader: Loader that creates these descriptors
    - ModelHandlerDescriptor: Contract-level handler descriptor
    - EnumHandlerTypeCategory: Handler behavioral classification

.. versionadded:: 0.7.0
    Created as part of OMN-1132 Handler Plugin Loader implementation.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.primitives import ModelSemVer
from omnibase_infra.enums.enum_handler_type_category import EnumHandlerTypeCategory


class ModelLoadedHandler(BaseModel):
    """Represents a handler loaded from a contract.

    Tracks metadata about handlers that have been successfully discovered,
    validated, and registered by the HandlerPluginLoader.

    This model is used for:
    - Runtime handler registry tracking
    - Handler discovery and filtering by capability tags
    - Diagnostics and introspection of loaded handlers
    - Audit logging of handler loading events

    Attributes:
        handler_name: Unique identifier for the handler. Used as the primary
            key for handler registry lookups.
        handler_type: Behavioral classification of the handler (COMPUTE, EFFECT,
            or NONDETERMINISTIC_COMPUTE). Determines policy envelope and runtime
            behavior.
        handler_class: Fully qualified class name of the handler implementation
            (e.g., 'myapp.handlers.AuthHandler'). Used for dynamic instantiation.
        contract_path: Absolute path to the contract YAML file from which this
            handler was loaded. Used for reloading and validation.
        capability_tags: List of tags for handler discovery and filtering.
            Examples: ['auth', 'validation', 'http-client'].
        loaded_at: Timestamp when the handler was successfully loaded.
            Used for diagnostics and cache invalidation.
        handler_version: Semantic version of the handler from the contract.
            Used for version tracking and compatibility checks.

    Example:
        >>> from datetime import datetime, UTC
        >>> from pathlib import Path
        >>> from omnibase_core.models.primitives import ModelSemVer
        >>> from omnibase_infra.enums import EnumHandlerTypeCategory
        >>> handler = ModelLoadedHandler(
        ...     handler_name="auth.validate_token",
        ...     handler_type=EnumHandlerTypeCategory.COMPUTE,
        ...     handler_class="myapp.handlers.TokenValidator",
        ...     contract_path=Path("/app/handlers/auth/handler_contract.yaml"),
        ...     capability_tags=["auth", "validation", "jwt"],
        ...     loaded_at=datetime.now(UTC),
        ...     handler_version=ModelSemVer(major=1, minor=0, patch=0),
        ... )
        >>> handler.handler_name
        'auth.validate_token'

    See Also:
        - :class:`ModelHandlerDescriptor`: Contract-level handler metadata.
        - :class:`EnumHandlerTypeCategory`: Handler behavioral classification.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    handler_name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the handler",
    )
    protocol_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Protocol type identifier for registry lookup (e.g., 'db', 'http'). "
            "Used as the key for handler registry registration."
        ),
    )
    handler_type: EnumHandlerTypeCategory = Field(
        ...,
        description="Behavioral classification (COMPUTE, EFFECT, NONDETERMINISTIC_COMPUTE)",
    )
    handler_class: str = Field(
        ...,
        min_length=3,
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$",
        description="Fully qualified handler class name (e.g., 'myapp.handlers.AuthHandler')",
    )
    contract_path: Path = Field(
        ...,
        description="Absolute path to the source contract YAML file",
    )
    capability_tags: list[str] = Field(
        default_factory=list,
        description="Tags for handler discovery and filtering",
    )
    loaded_at: datetime = Field(
        ...,
        description="Timestamp when the handler was successfully loaded",
    )
    handler_version: ModelSemVer = Field(
        ...,
        description="Handler semantic version from contract",
    )


__all__ = ["ModelLoadedHandler"]
