# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Security constants for runtime handler and plugin loading.

This module defines the trusted namespace prefixes for dynamic handler and plugin
loading. These prefixes form a security boundary - only modules from these
namespaces can be dynamically imported as handlers or plugins.

Security Model:
    Namespace allowlisting is the first security boundary for dynamic loading.
    It prevents arbitrary module imports but does not prevent:
    - Dangerous submodules within an allowed namespace
    - Dependency confusion attacks
    - Side effects at import time

    Additional security layers include:
    - Contract validation (handler_class must match contract schema)
    - Protocol validation (class must implement ProtocolHandler)
    - Optional: signature verification / registry provenance

Design Decisions:
    - SPI is NOT included because it contains protocols, not handler implementations
    - Third-party namespaces require explicit config file, not env vars
    - Env vars are only acceptable to point to a config file path
    - Plugin namespace prefixes are a superset of handler prefixes
    - Domain plugin entry point group uses PEP 621 naming conventions

Example:
    >>> from omnibase_infra.runtime.constants_security import (
    ...     TRUSTED_HANDLER_NAMESPACE_PREFIXES,
    ...     TRUSTED_PLUGIN_NAMESPACE_PREFIXES,
    ... )
    >>> handler_module = "omnibase_infra.handlers.handler_db"
    >>> any(
    ...     handler_module.startswith(prefix)
    ...     for prefix in TRUSTED_HANDLER_NAMESPACE_PREFIXES
    ... )
    True
    >>> plugin_module = "omnibase_infra.plugins.plugin_registry"
    >>> any(
    ...     plugin_module.startswith(prefix)
    ...     for prefix in TRUSTED_PLUGIN_NAMESPACE_PREFIXES
    ... )
    True

.. versionadded:: 0.2.8
    Created as part of OMN-1519 security hardening.

.. versionadded:: 0.3.0
    Added plugin security constants as part of OMN-2010.

.. versionchanged:: 0.6.0
    Added omniclaude. to TRUSTED_PLUGIN_NAMESPACE_PREFIXES (OMN-2047).
"""

from __future__ import annotations

from typing import Final

# Default trusted namespace prefixes for handler loading.
#
# SECURITY: This is a security boundary. Changes require review.
#
# Why these specific namespaces:
# - omnibase_core.: Core framework components (may contain base handlers)
# - omnibase_infra.: Infrastructure handlers (db, consul, vault, etc.)
#
# Why NOT omnibase_spi.:
# - SPI contains protocols (interfaces), not implementations
# - Handlers are implementations that live in infra or application code
# - Loading protocols as handlers is architecturally incorrect
#
# Third-party namespaces must be explicitly configured via security config file.
TRUSTED_HANDLER_NAMESPACE_PREFIXES: Final[tuple[str, ...]] = (
    "omnibase_core.",
    "omnibase_infra.",
)

# Default trusted namespace prefixes for plugin loading.
#
# SECURITY: This is a security boundary. Changes require review.
#
# Plugin namespace prefixes are a superset of handler prefixes. The core trust
# boundary (omnibase_core., omnibase_infra.) is shared, but plugins may include
# additional first-party namespaces that provide domain plugins without providing
# handler implementations.
#
# Why omniclaude. is included:
# - omniclaude provides PluginClaude, a first-party domain plugin registered via
#   pyproject.toml entry_points (onex.domain_plugins group)
# - Without this prefix, RegistryDomainPlugin.discover_from_entry_points() rejects
#   the entry point with status namespace_rejected (OMN-2047)
# - omniclaude is a first-party package maintained in the same monorepo ecosystem
#
# Why omniintelligence. is included:
# - omniintelligence provides PluginIntelligence, a first-party domain plugin
#   registered via pyproject.toml entry_points (onex.domain_plugins group)
# - Without this prefix, discover_from_entry_points() rejects it (OMN-2192)
# - omniintelligence is a first-party package in the OmniNode ecosystem
#
# Third-party plugin namespaces must be explicitly configured via security config file.
TRUSTED_PLUGIN_NAMESPACE_PREFIXES: Final[tuple[str, ...]] = (
    "omnibase_core.",
    "omnibase_infra.",
    "omniclaude.",
    "omniintelligence.",
)

# PEP 621 entry_points group name for domain plugin discovery.
#
# This is the group name used in pyproject.toml [project.entry-points] to register
# domain plugins for automatic discovery via importlib.metadata.entry_points().
#
# Example pyproject.toml usage:
#   [project.entry-points."onex.domain_plugins"]
#   my_plugin = "my_package.plugins:MyPlugin"
#
# SECURITY: Only plugins from trusted namespaces (see above) will be loaded,
# even if they are registered under this entry point group.
DOMAIN_PLUGIN_ENTRY_POINT_GROUP: Final[str] = "onex.domain_plugins"

# Environment variable name for security config file path.
# The config file (not the env var) contains the actual security settings.
# This keeps security configuration auditable and reviewable.
SECURITY_CONFIG_PATH_ENV_VAR: Final[str] = "ONEX_SECURITY_CONFIG_PATH"

# Environment variable to explicitly opt-in to namespace override.
# Required for emergency operations; logs loudly at startup.
ALLOW_NAMESPACE_OVERRIDE_ENV_VAR: Final[str] = "ONEX_ALLOW_HANDLER_NAMESPACE_OVERRIDE"
