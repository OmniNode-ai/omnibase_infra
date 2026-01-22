# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registry Error Class.

This module defines the PolicyRegistryError for policy registry operations.
"""

from typing import Any, cast

from omnibase_infra.enums import EnumPolicyType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.types import PolicyTypeInput


class PolicyRegistryError(RuntimeHostError):
    """Error raised when policy registry operations fail.

    Used for:
    - Attempting to get an unregistered policy
    - Registration failures (async validation, duplicate registration)
    - Invalid policy type identifiers
    - Policy validation failures during registration

    Extends RuntimeHostError as this is an infrastructure-layer runtime concern.

    Type Safety:
        The `policy_type` parameter accepts both `EnumPolicyType` enum values
        and string literals for backward compatibility. Using `EnumPolicyType`
        is strongly recommended for type safety and IDE autocomplete support.

    Example:
        >>> from omnibase_infra.errors import PolicyRegistryError
        >>> from omnibase_infra.enums import EnumPolicyType
        >>> from omnibase_infra.errors import ModelInfraErrorContext
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> from uuid import uuid4

        >>> # RECOMMENDED: Using EnumPolicyType for type safety
        >>> try:
        ...     policy = registry.get("unknown_policy_id")
        ... except PolicyRegistryError as e:
        ...     print(f"Policy not found: {e}")

        >>> # With context and EnumPolicyType (preferred approach)
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="get_policy",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise PolicyRegistryError(
        ...     "Policy not registered",
        ...     policy_id="rate_limit_default",
        ...     policy_type=EnumPolicyType.ORCHESTRATOR,  # Type-safe enum
        ...     context=context,
        ... )

        >>> # Reducer policy example
        >>> raise PolicyRegistryError(
        ...     "Reducer policy validation failed",
        ...     policy_id="state_merger",
        ...     policy_type=EnumPolicyType.REDUCER,  # Type-safe enum
        ... )

        >>> # Backward compatible with string (legacy)
        >>> raise PolicyRegistryError(
        ...     "Policy not registered",
        ...     policy_id="rate_limit_default",
        ...     policy_type="orchestrator",  # String (legacy support)
        ... )
    """

    def __init__(
        self,
        message: str,
        policy_id: str | None = None,
        policy_type: PolicyTypeInput | None = None,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize PolicyRegistryError.

        Args:
            message: Human-readable error message
            policy_id: The policy ID that caused the error (if applicable)
            policy_type: The policy type that caused the error. Accepts:
                - EnumPolicyType.ORCHESTRATOR or EnumPolicyType.REDUCER (recommended)
                - "orchestrator" or "reducer" string (legacy support)
                The enum value will be automatically converted to its string
                representation for serialization.
            context: Bundled infrastructure context for correlation_id and structured fields
            **extra_context: Additional context information
        """
        # Add policy_id and policy_type to extra_context if provided
        if policy_id is not None:
            extra_context["policy_id"] = policy_id
        if policy_type is not None:
            # Convert EnumPolicyType to string for serialization
            extra_context["policy_type"] = (
                policy_type.value
                if isinstance(policy_type, EnumPolicyType)
                else policy_type
            )

        # NOTE: Cast required for mypy - **dict[str, object] doesn't satisfy **context: Any
        super().__init__(
            message=message,
            context=context,
            **cast("dict[str, Any]", extra_context),
        )


__all__ = ["PolicyRegistryError"]
