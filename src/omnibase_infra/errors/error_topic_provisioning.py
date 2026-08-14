# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Topic-provisioning policy errors (OMN-15395).

``TopicReplicationPolicyError`` is raised by
:class:`~omnibase_infra.topics.model_topic_provisioning_policy.ModelTopicProvisioningPolicy`
when a topic spec cannot be provisioned under the resolved environment policy —
either because the owning contract declares no replication factor in an
environment that forbids implicit defaults, or because the resolved replication
factor is below the environment's durability floor (RF1 against managed
staging / MSK).

It is a distinct class (not a bare ``ProtocolConfigurationError``) so the
provisioning call sites can re-raise it past their best-effort ``except
Exception`` boundaries without also re-raising unrelated configuration errors.
A durability violation must never degrade into a warning-and-continue.
"""

from __future__ import annotations

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)


class TopicReplicationPolicyError(RuntimeHostError):
    """Raised when a topic spec violates the environment's replication policy.

    Example:
        >>> from omnibase_infra.enums import EnumInfraTransportType
        >>> context = ModelInfraErrorContext.with_correlation(
        ...     transport_type=EnumInfraTransportType.KAFKA,
        ...     operation="resolve_topic_spec",
        ... )
        >>> raise TopicReplicationPolicyError(  # doctest: +SKIP
        ...     "replication_factor=1 is rejected in managed staging",
        ...     context=context,
        ... )
    """

    def __init__(
        self,
        message: str,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        """Initialize with the INVALID_CONFIGURATION error code."""
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_CONFIGURATION,
            context=context,
            **extra_context,
        )


__all__: list[str] = ["TopicReplicationPolicyError"]
