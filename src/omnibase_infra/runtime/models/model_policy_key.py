# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registry Key Model.

Strongly-typed key for PolicyRegistry dict operations.
Replaces primitive tuple[str, str, str] pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.models.model_semver import ModelSemVer
from omnibase_infra.utils.util_semver import validate_version_lenient


class ModelPolicyKey(BaseModel):
    """Strongly-typed policy registry key.

    Replaces tuple[str, str, str] pattern with named fields,
    validation, and self-documenting structure.

    Attributes:
        policy_id: Unique identifier for the policy (e.g., 'exponential_backoff')
        policy_type: Policy category (EnumPolicyType or 'orchestrator'/'reducer' string)
        version: Semantic version string (e.g., '1.0.0')

    Example:
        >>> from omnibase_infra.enums import EnumPolicyType
        >>> key = ModelPolicyKey(
        ...     policy_id="retry_backoff",
        ...     policy_type=EnumPolicyType.ORCHESTRATOR,
        ...     version="1.0.0"
        ... )
        >>> print(key.policy_id)
        'retry_backoff'
        >>> # Backward compatible with string
        >>> key2 = ModelPolicyKey(
        ...     policy_id="state_merger",
        ...     policy_type="reducer",
        ...     version="1.0.0"
        ... )
    """

    policy_id: str = Field(..., description="Unique policy identifier")
    policy_type: str | EnumPolicyType = Field(
        ...,
        description="Policy type (EnumPolicyType or 'orchestrator'/'reducer' string)",
    )
    version: str = Field(default="1.0.0", description="Semantic version string")

    model_config = ConfigDict(
        frozen=True,  # Make hashable for dict keys
        str_strip_whitespace=True,
    )

    @field_validator("version", mode="before")
    @classmethod
    def normalize_version(cls, v: str) -> str:
        """Normalize version string for consistent lookups using ModelSemVer.

        This validator ensures that equivalent version strings are normalized
        to a canonical form, preventing lookup mismatches where "1.0.0" and
        "1.0" might be treated as different versions.

        Uses ModelSemVer for proper semantic version validation and normalization,
        avoiding duplicated string manipulation logic.

        Normalization rules:
            1. Strip leading/trailing whitespace
            2. Strip leading 'v' or 'V' prefix (e.g., "v1.0.0" -> "1.0.0")
            3. Validate format with validate_version_lenient (accepts 1, 1.0, 1.0.0)
            4. Expand to three-part version (e.g., "1" -> "1.0.0", "1.2" -> "1.2.0")
            5. Parse with ModelSemVer.from_string() for final validation
            6. Return normalized string via str(ModelSemVer)

        Args:
            v: The version string to normalize

        Returns:
            Normalized version string in "x.y.z" or "x.y.z-prerelease" format,
            or the original value if empty (let downstream validation handle it)

        Raises:
            ValueError: If the version string is invalid and cannot be parsed

        Examples:
            >>> ModelPolicyKey(policy_id="test", policy_type="orchestrator", version="1.0")
            ModelPolicyKey(policy_id='test', policy_type='orchestrator', version='1.0.0')
            >>> ModelPolicyKey(policy_id="test", policy_type="orchestrator", version="v2.1")
            ModelPolicyKey(policy_id='test', policy_type='orchestrator', version='2.1.0')
        """
        # Don't normalize empty/whitespace-only - let downstream validation handle it
        if not v or not v.strip():
            return v

        # Strip whitespace
        normalized = v.strip()

        # Strip leading 'v' or 'V' prefix
        if normalized.startswith(("v", "V")):
            normalized = normalized[1:]

        # Validate format with lenient parsing (accepts 1, 1.0, 1.0.0)
        # This will raise ValueError for invalid formats
        validate_version_lenient(normalized)

        # Split on first hyphen to handle prerelease suffix
        parts = normalized.split("-", 1)
        version_part = parts[0]
        prerelease = parts[1] if len(parts) > 1 else None

        # Expand to three-part version (x.y.z) for ModelSemVer parsing
        version_nums = version_part.split(".")
        while len(version_nums) < 3:
            version_nums.append("0")
        expanded_version = ".".join(version_nums)

        # Re-add prerelease if present
        if prerelease:
            expanded_version = f"{expanded_version}-{prerelease}"

        # Parse with ModelSemVer for final validation and canonical form
        semver = ModelSemVer.from_string(expanded_version)
        return str(semver)

    @field_validator("policy_type")
    @classmethod
    def validate_policy_type(cls, v: str | EnumPolicyType) -> str | EnumPolicyType:
        """Validate policy_type is a valid EnumPolicyType value.

        Args:
            v: The policy_type value to validate

        Returns:
            The validated policy_type value

        Raises:
            ValueError: If policy_type is not a valid EnumPolicyType value
        """
        if isinstance(v, EnumPolicyType):
            return v
        # If it's a string, validate it's a valid EnumPolicyType value
        valid_values = {e.value for e in EnumPolicyType}
        if v not in valid_values:
            raise ValueError(f"policy_type must be one of {valid_values}, got '{v}'")
        return v

    def to_tuple(self) -> tuple[str, str, str]:
        """Convert to tuple for backward compatibility.

        Returns:
            Tuple of (policy_id, policy_type, version)
        """
        policy_type_str = (
            self.policy_type.value
            if isinstance(self.policy_type, EnumPolicyType)
            else self.policy_type
        )
        return (self.policy_id, policy_type_str, self.version)

    @classmethod
    def from_tuple(cls, key_tuple: tuple[str, str, str]) -> ModelPolicyKey:
        """Create from tuple for backward compatibility.

        Args:
            key_tuple: Tuple of (policy_id, policy_type, version)

        Returns:
            ModelPolicyKey instance
        """
        return cls(
            policy_id=key_tuple[0],
            policy_type=key_tuple[1],
            version=key_tuple[2],
        )
