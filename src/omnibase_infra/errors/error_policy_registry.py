# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Policy Registry Error Class.

This module defines the PolicyRegistryError for policy registry operations.
"""

from typing import Optional

from omnibase_infra.errors.infra_errors import RuntimeHostError
from omnibase_infra.errors.model_infra_error_context import ModelInfraErrorContext


class PolicyRegistryError(RuntimeHostError):
    """Error raised when policy registry operations fail.

    Used for:
    - Attempting to get an unregistered policy
    - Registration failures (async validation, duplicate registration)
    - Invalid policy type identifiers
    - Policy validation failures during registration

    Extends RuntimeHostError as this is an infrastructure-layer runtime concern.

    Example:
        >>> from omnibase_infra.errors import PolicyRegistryError
        >>> try:
        ...     policy = registry.get("unknown_policy_id")
        ... except PolicyRegistryError as e:
        ...     print(f"Policy not found: {e}")

        >>> # With context for correlation tracking
        >>> context = ModelInfraErrorContext(
        ...     transport_type=EnumInfraTransportType.RUNTIME,
        ...     operation="get_policy",
        ...     correlation_id=uuid4(),
        ... )
        >>> raise PolicyRegistryError(
        ...     "Policy not registered",
        ...     policy_id="rate_limit_default",
        ...     policy_type="RateLimitPolicy",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        policy_id: Optional[str] = None,
        policy_type: Optional[str] = None,
        context: Optional[ModelInfraErrorContext] = None,
        **extra_context: object,
    ) -> None:
        """Initialize PolicyRegistryError.

        Args:
            message: Human-readable error message
            policy_id: The policy ID that caused the error (if applicable)
            policy_type: The policy type that caused the error (if applicable)
            context: Bundled infrastructure context for correlation_id and structured fields
            **extra_context: Additional context information
        """
        # Add policy_id and policy_type to extra_context if provided
        if policy_id is not None:
            extra_context["policy_id"] = policy_id
        if policy_type is not None:
            extra_context["policy_type"] = policy_type

        super().__init__(
            message=message,
            context=context,
            **extra_context,
        )


__all__ = ["PolicyRegistryError"]
