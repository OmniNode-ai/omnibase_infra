# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registry Key Model.

Strongly-typed key for PolicyRegistry dict operations.
Replaces primitive tuple[str, str, str] pattern.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelPolicyKey(BaseModel):
    """Strongly-typed policy registry key.

    Replaces tuple[str, str, str] pattern with named fields,
    validation, and self-documenting structure.

    Attributes:
        policy_id: Unique identifier for the policy (e.g., 'exponential_backoff')
        policy_type: Policy category ('orchestrator' or 'reducer')
        version: Semantic version string (e.g., '1.0.0')

    Example:
        >>> key = ModelPolicyKey(
        ...     policy_id="retry_backoff",
        ...     policy_type="orchestrator",
        ...     version="1.0.0"
        ... )
        >>> print(key.policy_id)
        'retry_backoff'
    """

    policy_id: str = Field(..., description="Unique policy identifier")
    policy_type: str = Field(..., description="Policy type (orchestrator or reducer)")
    version: str = Field(default="1.0.0", description="Semantic version string")

    model_config = ConfigDict(
        frozen=True,  # Make hashable for dict keys
        str_strip_whitespace=True,
    )

    def to_tuple(self) -> tuple[str, str, str]:
        """Convert to tuple for backward compatibility.

        Returns:
            Tuple of (policy_id, policy_type, version)
        """
        return (self.policy_id, self.policy_type, self.version)

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
