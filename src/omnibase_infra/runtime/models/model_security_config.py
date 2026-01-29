# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security Configuration Model for Runtime Handler Loading.

This module provides the Pydantic model for configuring trusted handler namespaces.
The model allows operators to extend the trusted namespace list via configuration
file while maintaining secure defaults.

Security Model:
    - Default: Only omnibase_core. and omnibase_infra. are trusted
    - Third-party: Requires allow_third_party_handlers=True AND
      explicit listing in allowed_handler_namespaces
    - Config file is auditable/reviewable (unlike env vars)

Example:
    >>> from omnibase_infra.runtime.models import ModelSecurityConfig
    >>> config = ModelSecurityConfig()  # Secure defaults
    >>> config.get_effective_namespaces()
    ('omnibase_core.', 'omnibase_infra.')

    >>> # Enable third-party handlers
    >>> config = ModelSecurityConfig(
    ...     allow_third_party_handlers=True,
    ...     allowed_handler_namespaces=(
    ...         "omnibase_core.",
    ...         "omnibase_infra.",
    ...         "mycompany.handlers.",
    ...     ),
    ... )
    >>> config.get_effective_namespaces()
    ('omnibase_core.', 'omnibase_infra.', 'mycompany.handlers.')

.. versionadded:: 0.2.8
    Created as part of OMN-1519 security hardening.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.constants_security import TRUSTED_HANDLER_NAMESPACE_PREFIXES


class ModelSecurityConfig(BaseModel):
    """Security configuration for runtime handler loading.

    This model allows operators to extend the trusted namespace list
    via configuration file. The defaults are secure - third-party
    namespaces require explicit opt-in.

    Security Model:
        - Default: Only omnibase_core. and omnibase_infra. are trusted
        - Third-party: Requires allow_third_party_handlers=True AND
          explicit listing in allowed_handler_namespaces
        - Config file is auditable/reviewable (unlike env vars)

    Attributes:
        allow_third_party_handlers: Enable loading handlers from third-party
            namespaces. When False, only TRUSTED_HANDLER_NAMESPACE_PREFIXES
            are allowed regardless of allowed_handler_namespaces setting.
        allowed_handler_namespaces: Allowed namespace prefixes for handler
            loading. Only effective when allow_third_party_handlers=True.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    allow_third_party_handlers: bool = Field(
        default=False,
        description="Enable loading handlers from third-party namespaces. "
        "When False, only TRUSTED_HANDLER_NAMESPACE_PREFIXES are allowed.",
    )

    allowed_handler_namespaces: tuple[str, ...] = Field(
        default=TRUSTED_HANDLER_NAMESPACE_PREFIXES,
        description="Allowed namespace prefixes for handler loading. "
        "Only effective when allow_third_party_handlers=True.",
    )

    def get_effective_namespaces(self) -> tuple[str, ...]:
        """Get the effective namespace allowlist based on configuration.

        Returns:
            Tuple of allowed namespace prefixes. If third-party handlers
            are disabled, returns only the trusted defaults regardless
            of the allowed_handler_namespaces setting.

        Example:
            >>> config = ModelSecurityConfig()
            >>> config.get_effective_namespaces()
            ('omnibase_core.', 'omnibase_infra.')

            >>> config = ModelSecurityConfig(
            ...     allow_third_party_handlers=True,
            ...     allowed_handler_namespaces=("custom.namespace.",),
            ... )
            >>> config.get_effective_namespaces()
            ('custom.namespace.',)
        """
        if not self.allow_third_party_handlers:
            return TRUSTED_HANDLER_NAMESPACE_PREFIXES
        return self.allowed_handler_namespaces


__all__: list[str] = ["ModelSecurityConfig"]
