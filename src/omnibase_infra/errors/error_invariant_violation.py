# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Invariant violation error."""

from __future__ import annotations

from uuid import UUID, uuid4

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class InvariantViolation(RuntimeHostError):  # noqa: N818 - Linear ticket names this API
    """Raised when an adapter action violates its declared contract invariants."""

    def __init__(
        self,
        *,
        action_name: str,
        protocol_domain: str,
        message: str | None = None,
        allowed_actions: tuple[str, ...] = (),
        context: ModelInfraErrorContext | None = None,
        correlation_id: UUID | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize an invariant violation.

        Args:
            action_name: Action requested by the caller.
            protocol_domain: Protocol domain that owns the allowed action set.
            message: Optional human-readable error message.
            allowed_actions: Declared actions that are valid for the domain.
            context: Optional infrastructure error context.
            correlation_id: Optional correlation ID for distributed tracing.
            **extra_context: Additional diagnostic context.
        """
        self.action_name = action_name
        self.protocol_domain = protocol_domain
        self.allowed_actions = allowed_actions

        if correlation_id is None:
            correlation_id = uuid4()

        if context is None:
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_allowed_action",
                target_name=protocol_domain,
                correlation_id=correlation_id,
            )

        extra_context["action_name"] = action_name
        extra_context["protocol_domain"] = protocol_domain
        extra_context["allowed_actions"] = allowed_actions

        if message is None:
            message = (
                f"Action {action_name!r} is not allowed for protocol domain "
                f"{protocol_domain!r}"
            )

        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.CONTRACT_VIOLATION,
            context=context,
            **extra_context,
        )


__all__ = ["InvariantViolation"]
