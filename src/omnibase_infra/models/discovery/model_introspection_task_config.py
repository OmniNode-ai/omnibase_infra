# Copyright 2025 OmniNode Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration model for introspection background tasks.

This module provides the configuration model used to control background
task behavior in MixinNodeIntrospection. Grouping parameters into a
configuration model follows ONEX patterns for reducing function parameter
count and union type patterns.

Background Tasks:
    - Heartbeat task: Periodically publishes heartbeat events for liveness monitoring
    - Registry listener: Responds to introspection request events from the registry

See Also:
    - MixinNodeIntrospection: The mixin that uses this configuration
    - ModelIntrospectionConfig: Main initialization configuration
"""

from pydantic import BaseModel, ConfigDict, Field


class ModelIntrospectionTaskConfig(BaseModel):
    """Configuration model for introspection background tasks.

    This model groups parameters for ``start_introspection_tasks()`` into a
    single configuration object, following ONEX conventions for reducing
    function parameter count and union type patterns.

    Using a config model instead of individual optional parameters reduces
    the number of union types in method signatures while providing better
    type safety and documentation.

    Attributes:
        enable_heartbeat: Whether to start the heartbeat background task.
            When enabled, periodically publishes heartbeat events to the
            heartbeat topic. Default: True.
        heartbeat_interval_seconds: Time between heartbeat publications
            in seconds. Must be >= 1.0 to prevent excessive traffic.
            Default: 30.0.
        enable_registry_listener: Whether to start the registry listener
            background task. When enabled, listens for introspection
            request events and responds with introspection data.
            Default: True.

    Example:
        ```python
        from omnibase_infra.models.discovery import ModelIntrospectionTaskConfig
        from omnibase_infra.mixins import MixinNodeIntrospection

        class MyNode(MixinNodeIntrospection):
            async def startup(self):
                # With config object
                task_config = ModelIntrospectionTaskConfig(
                    enable_heartbeat=True,
                    heartbeat_interval_seconds=15.0,
                    enable_registry_listener=True,
                )
                await self.start_introspection_tasks_from_config(task_config)

            async def shutdown(self):
                await self.stop_introspection_tasks()

        # Using defaults
        class SimpleNode(MixinNodeIntrospection):
            async def startup(self):
                config = ModelIntrospectionTaskConfig()  # All defaults
                await self.start_introspection_tasks_from_config(config)
        ```

    Configuration Guidelines:
        - Production: 30-60 second heartbeat interval
        - Development: 10-15 second heartbeat interval for faster feedback
        - Disable registry listener if node discovery is not needed

    See Also:
        MixinNodeIntrospection: The mixin that uses this configuration.
        ModelIntrospectionConfig: Main initialization configuration.
    """

    enable_heartbeat: bool = Field(
        default=True,
        description="Whether to start the heartbeat background task",
    )

    heartbeat_interval_seconds: float = Field(
        default=30.0,
        ge=1.0,
        description="Time between heartbeat publications in seconds",
    )

    enable_registry_listener: bool = Field(
        default=True,
        description="Whether to start the registry listener background task",
    )

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "enable_heartbeat": True,
                    "heartbeat_interval_seconds": 30.0,
                    "enable_registry_listener": True,
                },
                {
                    "enable_heartbeat": True,
                    "heartbeat_interval_seconds": 15.0,
                    "enable_registry_listener": False,
                },
            ]
        },
    )


__all__ = ["ModelIntrospectionTaskConfig"]
