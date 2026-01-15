# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registry Key Model.

Strongly-typed key for RegistryPolicy dict operations.
Replaces primitive tuple[str, str, str] pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.runtime.util_version import normalize_version


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
    def validate_and_normalize_version(cls, v: str) -> str:
        """Normalize version string for consistent lookups.

        Delegates to the shared normalize_version utility which is the
        SINGLE SOURCE OF TRUTH for version normalization in omnibase_infra.

        Converts version strings to canonical x.y.z format. This ensures consistent
        version handling across all ONEX components, preventing lookup mismatches
        where "1.0.0" and "1.0" might be treated as different versions.

        Normalization rules:
            1. Strip leading/trailing whitespace
            2. Strip leading 'v' or 'V' prefix
            3. Expand partial versions (1 -> 1.0.0, 1.0 -> 1.0.0)
            4. Parse with ModelSemVer.parse() for validation
            5. Preserve prerelease suffix if present

        Args:
            v: The version string to normalize

        Returns:
            Normalized version string in "x.y.z" or "x.y.z-prerelease" format

        Raises:
            ValueError: If the version string is invalid and cannot be parsed

        Examples:
            >>> ModelPolicyKey(policy_id="test", policy_type="orchestrator", version="1.0")
            ModelPolicyKey(policy_id='test', policy_type='orchestrator', version='1.0.0')
            >>> ModelPolicyKey(policy_id="test", policy_type="orchestrator", version="v2.1")
            ModelPolicyKey(policy_id='test', policy_type='orchestrator', version='2.1.0')
        """
        return normalize_version(v)

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
