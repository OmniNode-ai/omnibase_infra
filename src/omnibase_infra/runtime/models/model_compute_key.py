# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Compute Registry Key Model.

Strongly-typed key for RegistryCompute dict operations.
Replaces primitive tuple[str, str] pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.utils import validate_version_lenient


class ModelComputeKey(BaseModel):
    """Strongly-typed compute registry key.

    Replaces tuple[str, str] pattern with named fields,
    validation, and self-documenting structure.

    Attributes:
        plugin_id: Unique identifier for the compute plugin (e.g., 'json_normalizer')
        version: Semantic version string (e.g., '1.0.0')

    Example:
        >>> key = ModelComputeKey(plugin_id="json_normalizer", version="1.0.0")
        >>> print(key.to_tuple())
        ('json_normalizer', '1.0.0')
        >>> # Create from tuple for backward compatibility
        >>> key2 = ModelComputeKey.from_tuple(("xml_parser", "2.1.0"))
        >>> print(key2.plugin_id)
        'xml_parser'
    """

    plugin_id: str = Field(
        ...,
        min_length=1,
        description="Unique compute plugin identifier",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version string",
    )

    model_config = ConfigDict(
        frozen=True,  # Make hashable for dict keys
        str_strip_whitespace=True,
    )

    @field_validator("version")
    @classmethod
    def validate_semver(cls, v: str) -> str:
        """Validate semantic version format.

        Accepts formats like:
        - "1.0.0" (major.minor.patch)
        - "1.0" (major.minor)
        - "1" (major only)
        - "1.2.3-alpha" (with prerelease)
        - "1.2.3-beta.1" (with prerelease segments)

        Args:
            v: The version string to validate

        Returns:
            The validated version string

        Raises:
            ValueError: If version format is invalid
        """
        return validate_version_lenient(v)

    def to_tuple(self) -> tuple[str, str]:
        """Convert to tuple for backward compatibility.

        Returns:
            Tuple of (plugin_id, version)
        """
        return (self.plugin_id, self.version)

    @classmethod
    def from_tuple(cls, key_tuple: tuple[str, str]) -> ModelComputeKey:
        """Create from tuple for backward compatibility.

        Args:
            key_tuple: Tuple of (plugin_id, version)

        Returns:
            ModelComputeKey instance
        """
        return cls(
            plugin_id=key_tuple[0],
            version=key_tuple[1],
        )


__all__ = ["ModelComputeKey"]
