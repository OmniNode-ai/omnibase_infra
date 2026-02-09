# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security Configuration Model for Runtime Handler and Plugin Loading.

This module provides the Pydantic model for configuring trusted handler and plugin
namespaces. The model allows operators to extend the trusted namespace list via
configuration file while maintaining secure defaults for both handler loading and
plugin discovery.

Security Model:
    - Default handlers: Only omnibase_core. and omnibase_infra. are trusted
    - Default plugins: omnibase_core., omnibase_infra., and omniclaude. are trusted
    - Third-party handlers: Requires allow_third_party_handlers=True AND
      explicit listing in allowed_handler_namespaces
    - Third-party plugins: Requires allow_third_party_plugins=True AND
      explicit listing in allowed_plugin_namespaces
    - Config file is auditable/reviewable (unlike env vars)

Example:
    >>> from omnibase_infra.runtime.models import ModelSecurityConfig
    >>> config = ModelSecurityConfig()  # Secure defaults
    >>> config.get_effective_namespaces()
    ('omnibase_core.', 'omnibase_infra.')
    >>> config.get_effective_plugin_namespaces()
    ('omnibase_core.', 'omnibase_infra.', 'omniclaude.')

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

    >>> # Enable third-party plugins
    >>> config = ModelSecurityConfig(
    ...     allow_third_party_plugins=True,
    ...     allowed_plugin_namespaces=(
    ...         "omnibase_core.",
    ...         "omnibase_infra.",
    ...         "mycompany.plugins.",
    ...     ),
    ... )
    >>> config.get_effective_plugin_namespaces()
    ('omnibase_core.', 'omnibase_infra.', 'mycompany.plugins.')

.. versionadded:: 0.2.8
    Created as part of OMN-1519 security hardening.

.. versionchanged:: 0.3.0
    Added plugin namespace fields as part of OMN-2015.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.constants_security import (
    TRUSTED_HANDLER_NAMESPACE_PREFIXES,
    TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
)


class ModelSecurityConfig(BaseModel):
    """Security configuration for runtime handler and plugin loading.

    This model allows operators to extend the trusted namespace list
    via configuration file. The defaults are secure - third-party
    namespaces require explicit opt-in for both handlers and plugins.

    Security Model:
        - Default handlers: Only omnibase_core. and omnibase_infra. are trusted
        - Default plugins: omnibase_core., omnibase_infra., and omniclaude.
          are trusted (omniclaude provides first-party domain plugins)
        - Third-party handlers: Requires allow_third_party_handlers=True AND
          explicit listing in allowed_handler_namespaces
        - Third-party plugins: Requires allow_third_party_plugins=True AND
          explicit listing in allowed_plugin_namespaces
        - Config file is auditable/reviewable (unlike env vars)

    Attributes:
        allow_third_party_handlers: Enable loading handlers from third-party
            namespaces. When False, only TRUSTED_HANDLER_NAMESPACE_PREFIXES
            are allowed regardless of allowed_handler_namespaces setting.
        allowed_handler_namespaces: Allowed namespace prefixes for handler
            loading. Only effective when allow_third_party_handlers=True.
        allow_third_party_plugins: Enable discovery of plugins from third-party
            namespaces via entry_points. When False, only
            TRUSTED_PLUGIN_NAMESPACE_PREFIXES are allowed regardless of
            allowed_plugin_namespaces setting.
        allowed_plugin_namespaces: Allowed namespace prefixes for plugin
            discovery. Only effective when allow_third_party_plugins=True.
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

    allow_third_party_plugins: bool = Field(
        default=False,
        description="Enable discovery of plugins from third-party namespaces "
        "via entry_points. When False, only TRUSTED_PLUGIN_NAMESPACE_PREFIXES "
        "are allowed.",
    )

    allowed_plugin_namespaces: tuple[str, ...] = Field(
        default=TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
        description="Allowed namespace prefixes for plugin discovery. "
        "Only effective when allow_third_party_plugins=True.",
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

    def get_effective_plugin_namespaces(self) -> tuple[str, ...]:
        """Get the effective plugin namespace allowlist based on configuration.

        Returns:
            Tuple of allowed namespace prefixes for plugin discovery. If
            third-party plugins are disabled, returns only the trusted defaults
            regardless of the allowed_plugin_namespaces setting.

        Example:
            >>> config = ModelSecurityConfig()
            >>> config.get_effective_plugin_namespaces()
            ('omnibase_core.', 'omnibase_infra.', 'omniclaude.')

            >>> config = ModelSecurityConfig(
            ...     allow_third_party_plugins=True,
            ...     allowed_plugin_namespaces=("custom.plugins.",),
            ... )
            >>> config.get_effective_plugin_namespaces()
            ('custom.plugins.',)
        """
        if not self.allow_third_party_plugins:
            return TRUSTED_PLUGIN_NAMESPACE_PREFIXES
        return self.allowed_plugin_namespaces


__all__: list[str] = ["ModelSecurityConfig"]
