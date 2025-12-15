# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registration Model.

This module provides the Pydantic model for policy registration parameters,
used to register policy plugins with the PolicyRegistry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums import EnumPolicyType

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_policy import ProtocolPolicy


class ModelPolicyRegistration(BaseModel):
    """Model for policy registration parameters.

    Encapsulates all parameters needed to register a policy plugin with
    the PolicyRegistry. This model reduces the number of parameters in
    the register() method signature.

    Attributes:
        policy_id: Unique identifier for the policy (e.g., 'exponential_backoff')
        policy_class: Policy implementation class that implements ProtocolPolicy
        policy_type: Policy type - either orchestrator or reducer
        version: Semantic version string (default: "1.0.0")
        description: Human-readable description of the policy
        deterministic_async: If True, allows async interface (must be explicit)

    Example:
        >>> from omnibase_infra.runtime.models import ModelPolicyRegistration
        >>> from omnibase_infra.enums import EnumPolicyType
        >>> registration = ModelPolicyRegistration(
        ...     policy_id="exponential_backoff",
        ...     policy_class=ExponentialBackoffPolicy,
        ...     policy_type=EnumPolicyType.ORCHESTRATOR,
        ...     version="1.0.0",
        ...     description="Calculates exponential backoff delays for retries",
        ... )
    """

    model_config = ConfigDict(
        strict=False,  # Allow type coercion for policy_class
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Required for type[ProtocolPolicy]
    )

    policy_id: str = Field(
        ...,
        description="Unique policy identifier (e.g., 'exponential_backoff')",
    )
    policy_class: type = Field(
        ...,
        description="Policy implementation class that implements ProtocolPolicy",
    )
    policy_type: Union[str, EnumPolicyType] = Field(
        ...,
        description="Policy type - either 'orchestrator' or 'reducer'",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version string",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the policy",
    )
    deterministic_async: bool = Field(
        default=False,
        description="If True, allows async interface (must be explicit for async policies)",
    )


__all__: list[str] = ["ModelPolicyRegistration"]
