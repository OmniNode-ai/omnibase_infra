# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Introspection Configuration Model.

This module provides the configuration model for auto-introspection behavior
in RuntimeHostProcess. It controls startup introspection timing, jitter for
stampede prevention, and throttling for rapid restart scenarios.

This model is distinct from ModelIntrospectionConfig (in models/discovery/)
which configures the MixinNodeIntrospection mixin initialization. This model
specifically configures RuntimeHostProcess startup behavior.

Related Tickets:
    - OMN-1930: Phase 1 - Fix Auto-Introspection (P0)

See Also:
    - ProtocolNodeIntrospection: Protocol interface for introspection
    - RuntimeHostProcess: Consumer of this configuration
    - ModelIntrospectionConfig: Mixin initialization configuration (different purpose)
    - ModelIntrospectionTaskConfig: Background task configuration

.. versionadded:: 0.4.1
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeIntrospectionConfig(BaseModel):
    """Configuration model for RuntimeHostProcess auto-introspection.

    This model controls how RuntimeHostProcess announces node presence on
    startup. Key features:

    - **Jitter**: Random delay before publishing to prevent thundering herd
      when many nodes restart simultaneously (e.g., cluster restart).
    - **Throttling**: Minimum interval between introspection events to prevent
      stampede on rapid restart cycles.

    Attributes:
        enabled: Whether to enable auto-introspection on startup. When False,
            RuntimeHostProcess will not publish introspection events. This is
            useful for testing or when manual introspection control is needed.
            Default: True.
        jitter_max_ms: Maximum jitter in milliseconds before publishing startup
            introspection. A random delay between 0 and this value is applied.
            This prevents thundering herd when many nodes restart together.
            Default: 5000 (5 seconds). Range: 0-30000.
        throttle_min_interval_s: Minimum interval in seconds between introspection
            events. If a node restarts rapidly, subsequent introspection events
            within this interval are skipped to prevent stampede. Default: 10
            seconds. Range: 1-60.

    Example:
        ```python
        from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig

        # Production configuration (conservative jitter, standard intervals)
        prod_config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=5000,
            throttle_min_interval_s=10,
        )

        # Development configuration (minimal jitter for faster feedback)
        dev_config = ModelRuntimeIntrospectionConfig(
            enabled=True,
            jitter_max_ms=500,
            throttle_min_interval_s=5,
        )

        # Testing configuration (introspection disabled)
        test_config = ModelRuntimeIntrospectionConfig(enabled=False)
        ```

    Configuration Guidelines:
        - **Production**: Use default values (5s jitter, 10s throttle)
        - **Development**: Lower jitter (500ms) for faster feedback
        - **Testing**: Disable introspection or use minimal values
        - **Large Clusters**: Consider higher jitter (10-15s) to spread startup load

    See Also:
        RuntimeHostProcess: Uses this configuration for startup introspection.
        ProtocolNodeIntrospection: Protocol for introspection operations.
    """

    enabled: bool = Field(
        default=True,
        description="Whether to enable auto-introspection on startup",
    )

    jitter_max_ms: int = Field(
        default=5000,
        ge=0,
        le=30000,
        description="Maximum jitter in milliseconds before publishing startup introspection",
    )

    throttle_min_interval_s: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Minimum interval in seconds between introspection events",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "enabled": True,
                    "jitter_max_ms": 5000,
                    "throttle_min_interval_s": 10,
                },
                {
                    "enabled": True,
                    "jitter_max_ms": 500,
                    "throttle_min_interval_s": 5,
                },
                {
                    "enabled": False,
                    "jitter_max_ms": 0,
                    "throttle_min_interval_s": 10,
                },
            ]
        },
    )


__all__ = ["ModelRuntimeIntrospectionConfig"]
