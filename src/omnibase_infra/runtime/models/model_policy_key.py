# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ModelPolicyKey - Strongly-typed PolicyRegistry key.

Defines ModelPolicyKey for PolicyRegistry dict operations.
Replaces primitive tuple[str, str, str] pattern with named fields.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.runtime.util_version import normalize_version
from omnibase_infra.types import PolicyTypeInput
from omnibase_infra.utils import validate_policy_type_value

# NOTE: PolicyTypeInput is the INPUT type (str | EnumPolicyType) for API flexibility.
# The validator coerces strings to EnumPolicyType, so the actual stored value is
# always an enum. This ensures type-safe access after model instantiation.


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
        retry_backoff
        >>> # Also accepts string policy_type
        >>> key2 = ModelPolicyKey(
        ...     policy_id="state_merger",
        ...     policy_type="reducer",
        ...     version="1.0.0"
        ... )
    """

    policy_id: str = Field(..., description="Unique policy identifier")
    # NOTE: Field accepts PolicyTypeInput (str | EnumPolicyType) via mode="before" coercion,
    # but the validator always coerces to EnumPolicyType. We annotate with the coerced type.
    policy_type: EnumPolicyType = Field(
        ...,
        description="Policy type (accepts string or enum, always stored as EnumPolicyType)",
    )
    version: str = Field(default="1.0.0", description="Semantic version string")

    model_config = ConfigDict(
        frozen=True,  # Make hashable for dict keys
        from_attributes=True,
        str_strip_whitespace=True,
        extra="forbid",
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

    @field_validator("policy_type", mode="before")
    @classmethod
    def validate_policy_type(cls, v: PolicyTypeInput) -> EnumPolicyType:
        """Validate and coerce policy_type to EnumPolicyType.

        Delegates to shared utility for consistent validation across all models.
        String values are coerced to EnumPolicyType, ensuring type-safe access.

        Note: mode="before" allows accepting str input before Pydantic validation.
        """
        return validate_policy_type_value(v)

    def to_tuple(self) -> tuple[str, str, str]:
        """Convert to tuple representation.

        Returns:
            Tuple of (policy_id, policy_type, version)
        """
        # policy_type is always EnumPolicyType after validation coercion
        return (self.policy_id, self.policy_type.value, self.version)

    @classmethod
    def from_tuple(cls, key_tuple: tuple[str, str, str]) -> ModelPolicyKey:
        """Create from tuple representation.

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
