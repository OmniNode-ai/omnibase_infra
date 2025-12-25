# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Re-exports of introspection configuration from canonical location.

This module provides backward-compatible re-exports of ModelIntrospectionConfig
and related constants from their canonical location in
``omnibase_infra.models.discovery.model_introspection_config``.

Note:
    This module exists for backward compatibility with code that imports from
    ``omnibase_infra.mixins.model_introspection_config``. New code should import
    from ``omnibase_infra.models.discovery`` instead.

Example:
    Preferred (canonical) import::

        from omnibase_infra.models.discovery import (
            ModelIntrospectionConfig,
            DEFAULT_INTROSPECTION_TOPIC,
            DEFAULT_HEARTBEAT_TOPIC,
            DEFAULT_REQUEST_INTROSPECTION_TOPIC,
        )

    Backward-compatible import::

        from omnibase_infra.mixins.model_introspection_config import (
            ModelIntrospectionConfig,
            DEFAULT_INTROSPECTION_TOPIC,
            DEFAULT_HEARTBEAT_TOPIC,
            DEFAULT_REQUEST_INTROSPECTION_TOPIC,
        )
"""

from omnibase_infra.models.discovery.model_introspection_config import (
    DEFAULT_HEARTBEAT_TOPIC,
    DEFAULT_INTROSPECTION_TOPIC,
    DEFAULT_REQUEST_INTROSPECTION_TOPIC,
    INVALID_TOPIC_CHARS,
    TOPIC_PATTERN,
    VERSION_SUFFIX_PATTERN,
    ModelIntrospectionConfig,
)

__all__ = [
    "DEFAULT_HEARTBEAT_TOPIC",
    "DEFAULT_INTROSPECTION_TOPIC",
    "DEFAULT_REQUEST_INTROSPECTION_TOPIC",
    "INVALID_TOPIC_CHARS",
    "TOPIC_PATTERN",
    "VERSION_SUFFIX_PATTERN",
    "ModelIntrospectionConfig",
]
