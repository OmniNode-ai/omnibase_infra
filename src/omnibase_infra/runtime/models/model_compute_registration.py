# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Compute Registration Model.

This module provides the Pydantic model for compute plugin registration parameters,
used to register compute plugins with the RegistryCompute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils import validate_version_lenient

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolPluginCompute

    ComputePluginClass = type[ProtocolPluginCompute]
else:
    # NOTE: At runtime we use generic type to avoid circular import.
    # Type checkers see proper protocol type via TYPE_CHECKING block.
    ComputePluginClass = type  # type: ignore[assignment,misc]


class ModelComputeRegistration(BaseModel):
    """Model for compute plugin registration parameters.

    Encapsulates all parameters needed to register a compute plugin with
    the RegistryCompute.

    Attributes:
        plugin_id: Unique identifier for the plugin (e.g., 'json_normalizer')
        plugin_class: Plugin implementation class
        version: Semantic version string (default: "1.0.0")
        description: Human-readable description of the plugin
        deterministic_async: If True, allows async interface (MUST be explicitly flagged)

    Example:
        >>> from omnibase_infra.runtime.models import ModelComputeRegistration
        >>> registration = ModelComputeRegistration(
        ...     plugin_id="json_normalizer",
        ...     plugin_class=JsonNormalizerPlugin,
        ...     version="1.0.0",
        ...     description="Normalizes JSON for deterministic comparison",
        ... )
    """

    model_config = ConfigDict(
        strict=False,
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Required for type[ProtocolPluginCompute]
        from_attributes=True,  # pytest-xdist compatibility
    )

    plugin_id: str = Field(
        ...,
        min_length=1,
        description="Unique compute plugin identifier (e.g., 'json_normalizer')",
    )
    plugin_class: ComputePluginClass = Field(
        ...,
        description="Compute plugin implementation class",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version string",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the plugin",
    )
    deterministic_async: bool = Field(
        default=False,
        description="If True, allows async interface. MUST be explicitly flagged for async plugins.",
    )

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format.

        Args:
            v: The version string to validate

        Returns:
            The validated version string

        Raises:
            ValueError: If version format is invalid
        """
        return validate_version_lenient(v)


__all__: list[str] = ["ModelComputeRegistration"]
