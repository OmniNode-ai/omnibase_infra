# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Environment-level security policy model.

This module defines the security constraints that apply at the environment
level. These constraints determine what security capabilities handlers
are permitted to request in each deployment environment.

Environment Policy Components:
    - Permitted secret scopes: What secrets can be accessed
    - Maximum data classification: Highest sensitivity level allowed
    - Outbound domain constraints: Network access restrictions
    - Adapter overrides: Special rules for adapter handlers

Validation Flow:
    1. Handler declares security policy (ModelHandlerSecurityPolicy)
    2. Environment policy defines constraints (ModelEnvironmentPolicy)
    3. Registration validates handler policy against environment constraints
    4. Runtime enforces declared policy at invocation time

See Also:
    - ModelHandlerSecurityPolicy: Handler-declared security requirements
    - EnumSecurityRuleId: Security validation rule identifiers
    - EnumEnvironment: Deployment environment classification
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.enums import EnumDataClassification
from omnibase_infra.enums import EnumEnvironment


class ModelEnvironmentPolicy(BaseModel):
    """Environment-level security constraints.

    Defines what security capabilities are permitted in a given environment.
    Used to validate handler security policies at registration time.

    Environment Hierarchy (most to least permissive):
        1. DEVELOPMENT - Most permissive, allows debugging features
        2. CI - Automated testing, some elevated permissions
        3. STAGING - Production-like with some relaxations
        4. PRODUCTION - Most restrictive, full security enforcement

    Constraint Types:
        - Allowlists: Define what IS permitted (secret scopes, domains)
        - Maximums: Define upper bounds (data classification)
        - Requirements: Define what handlers MUST declare (domain allowlists)
        - Overrides: Special rules for specific handler types (adapters)

    Attributes:
        environment: Target deployment environment.
        permitted_secret_scopes: Secret scopes permitted in this environment.
            Handlers requesting scopes not in this set will be rejected.
        max_data_classification: Maximum data classification allowed.
            Handlers declaring higher classification will be rejected.
        allowed_outbound_domains: Reserved for future registration-time domain
            enforcement. Currently NOT validated at registration time. This field
            is intended for future validation where handlers requesting domains
            not in this list would be rejected (unless None/unrestricted).
            None means unrestricted, empty list would mean no outbound access.
            Note: Handler-level domain allowlists (allowed_domains) ARE validated
            via require_explicit_domain_allowlist constraint.
        require_explicit_domain_allowlist: Whether handlers must declare
            explicit domain allowlists. When True, handlers with empty
            domain lists or ["*"] will be rejected.
        adapter_secrets_override_allowed: Whether adapters can request
            secrets. Usually False - adapters should use injected credentials.

    Example:
        >>> # Production environment with strict constraints
        >>> prod_policy = ModelEnvironmentPolicy(
        ...     environment=EnumEnvironment.PRODUCTION,
        ...     permitted_secret_scopes=frozenset({
        ...         "database/readonly",
        ...         "api/service-account",
        ...     }),
        ...     max_data_classification=EnumDataClassification.CONFIDENTIAL,
        ...     allowed_outbound_domains=[
        ...         "api.internal.example.com",
        ...         "metrics.internal.example.com",
        ...     ],
        ...     require_explicit_domain_allowlist=True,
        ...     adapter_secrets_override_allowed=False,
        ... )

        >>> # Development environment with relaxed constraints
        >>> dev_policy = ModelEnvironmentPolicy(
        ...     environment=EnumEnvironment.DEVELOPMENT,
        ...     permitted_secret_scopes=frozenset({"*"}),  # All scopes
        ...     max_data_classification=EnumDataClassification.TOP_SECRET,
        ...     allowed_outbound_domains=None,  # Unrestricted
        ...     require_explicit_domain_allowlist=False,
        ...     adapter_secrets_override_allowed=True,
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    environment: EnumEnvironment = Field(
        description="Target environment",
    )

    permitted_secret_scopes: frozenset[str] = Field(
        default_factory=frozenset,
        description="Secret scopes permitted in this environment",
    )

    max_data_classification: EnumDataClassification = Field(
        default=EnumDataClassification.INTERNAL,
        description="Maximum data classification allowed",
    )

    allowed_outbound_domains: list[str] | None = Field(
        default=None,
        description=(
            "Reserved for future registration-time domain enforcement. "
            "Currently NOT validated - see require_explicit_domain_allowlist "
            "for handler-level domain validation. None = unrestricted."
        ),
    )

    require_explicit_domain_allowlist: bool = Field(
        default=False,
        description="Whether handlers must declare explicit domain allowlists",
    )

    adapter_secrets_override_allowed: bool = Field(
        default=False,
        description=(
            "Whether adapters can request secrets directly (usually False). "
            "SECURITY WARNING: Enabling in production violates least-privilege "
            "principles. Adapters should use platform secret management (e.g., Vault) "
            "rather than direct secret access. Only enable for development/testing "
            "environments where secret isolation is less critical."
        ),
    )


__all__ = ["ModelEnvironmentPolicy"]
