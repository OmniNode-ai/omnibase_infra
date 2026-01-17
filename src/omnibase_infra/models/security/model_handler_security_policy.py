# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler-declared security policy model.

This module defines the security policy that handlers declare at registration
time. The policy specifies what security capabilities the handler requires,
including secret access, network access, and data classification levels.

Security Policy Components:
    - Secret scopes: What secrets the handler needs access to
    - Network access: What domains the handler may contact
    - Data classification: Maximum data sensitivity level processed
    - Adapter tag: Whether this is a platform adapter (stricter rules)

Validation:
    Security policies are validated against environment constraints at
    registration time. See ModelEnvironmentPolicy for environment-level
    constraints.

See Also:
    - ModelEnvironmentPolicy: Environment-level security constraints
    - EnumSecurityRuleId: Security validation rule identifiers
    - EnumHandlerTypeCategory: Handler behavioral classification
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumDataClassification
from omnibase_infra.enums import EnumHandlerTypeCategory


class ModelHandlerSecurityPolicy(BaseModel):
    """Security policy declared by a handler in its contract.

    This model captures the security requirements and constraints
    that a handler declares at registration time. The policy is
    validated against environment constraints before the handler
    is permitted to register.

    Security Requirements:
        Handlers must declare ALL security capabilities they require.
        Any undeclared capability will be denied at runtime.

    Adapter Handlers:
        Handlers marked as adapters (is_adapter=True) have stricter
        security constraints:
        - Cannot request secret scopes (must use injected credentials)
        - Must be EFFECT category (adapters perform I/O)
        - May require explicit domain allowlists in production

    Attributes:
        secret_scopes: Secret scopes this handler requires access to.
            Empty set means no secret access needed.
        allowed_domains: Outbound domains this handler may access.
            Empty list means no outbound access. Use ["*"] for unrestricted
            (only permitted in development).
        data_classification: Maximum data classification this handler
            processes. Cannot exceed environment maximum.
        is_adapter: Whether this is an adapter handler (platform plumbing).
            Adapters have stricter security rules applied.
        handler_type_category: Behavioral classification of the handler.
            Required when is_adapter=True (must be EFFECT).

    Example:
        >>> policy = ModelHandlerSecurityPolicy(
        ...     secret_scopes=frozenset({"database/readonly"}),
        ...     allowed_domains=["api.internal.example.com"],
        ...     data_classification=EnumDataClassification.CONFIDENTIAL,
        ...     is_adapter=False,
        ...     handler_type_category=EnumHandlerTypeCategory.EFFECT,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    secret_scopes: frozenset[str] = Field(
        default_factory=frozenset,
        description="Secret scopes this handler requires access to",
    )

    allowed_domains: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Outbound domains this handler may access (immutable tuple)",
    )

    data_classification: EnumDataClassification = Field(
        default=EnumDataClassification.INTERNAL,
        description="Maximum data classification this handler processes",
    )

    is_adapter: bool = Field(
        default=False,
        description="Whether this is an adapter handler (stricter security rules)",
    )

    handler_type_category: EnumHandlerTypeCategory | None = Field(
        default=None,
        description="Behavioral classification of the handler",
    )


__all__ = ["ModelHandlerSecurityPolicy"]
