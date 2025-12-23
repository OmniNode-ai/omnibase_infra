# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Compute Registry Key Model.

Strongly-typed key for RegistryCompute dict operations.
Replaces primitive tuple[str, str] pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
        if not v or not v.strip():
            raise ValueError("Version cannot be empty")

        v = v.strip()

        # Split off prerelease suffix
        version_part = v
        if "-" in v:
            version_part, prerelease = v.split("-", 1)
            if not prerelease:
                raise ValueError(
                    f"Invalid version '{v}': prerelease cannot be empty after '-'"
                )

        # Parse major.minor.patch components
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
