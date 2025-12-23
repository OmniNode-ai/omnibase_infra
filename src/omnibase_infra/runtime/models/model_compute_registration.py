# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Compute Registration Model.

This module provides the Pydantic model for compute plugin registration parameters,
used to register compute plugins with the RegistryCompute.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from omnibase_infra.protocols import ProtocolPluginCompute

    ComputePluginClass = type[ProtocolPluginCompute]
else:
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
        if not v or not v.strip():
            raise ValueError("Version cannot be empty")

        v = v.strip()
        if "-" in v:
            version_part, prerelease = v.split("-", 1)
            if not prerelease:
                raise ValueError(
                    f"Invalid version '{v}': prerelease cannot be empty after '-'"
                )
        else:
            version_part = v

        parts = version_part.split(".")
        if len(parts) < 1 or len(parts) > 3:
            raise ValueError(
                f"Invalid version '{v}': expected format 'major.minor.patch'"
            )

        for part in parts:
            if not part:
                raise ValueError(f"Invalid version '{v}': empty component")
            try:
                num = int(part)
                if num < 0:
                    raise ValueError(f"Invalid version '{v}': negative component")
            except ValueError as e:
                if "negative component" in str(e):
                    raise
                raise ValueError(
                    f"Invalid version '{v}': non-integer component '{part}'"
                ) from None

        return v


__all__: list[str] = ["ModelComputeRegistration"]
