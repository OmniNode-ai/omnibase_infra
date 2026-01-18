# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for structured logging sink.

This module defines the configuration model for creating SinkLoggingStructured
instances. The model validates configuration parameters and provides sensible
defaults for zero-config usage.

Usage:
    ```python
    from omnibase_infra.observability.models import ModelLoggingSinkConfig

    # Default configuration (JSON output, 1000 buffer)
    config = ModelLoggingSinkConfig()

    # Custom configuration (console output, larger buffer)
    config = ModelLoggingSinkConfig(
        max_buffer_size=5000,
        output_format="console",
    )
    ```
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelLoggingSinkConfig(BaseModel):
    """Configuration model for structured logging sink creation.

    This model defines all configurable parameters for creating a
    SinkLoggingStructured instance. All fields have sensible defaults
    allowing zero-config usage.

    Attributes:
        max_buffer_size: Maximum number of log entries to buffer before
            oldest entries are dropped. Higher values use more memory but
            reduce the chance of log loss during burst traffic.
        output_format: Output format for log entries.
            - "json": Machine-readable JSON format (default)
            - "console": Human-readable colored console output

    Example:
        ```python
        # Default configuration (JSON output, 1000 buffer)
        config = ModelLoggingSinkConfig()

        # Custom configuration (console output, larger buffer)
        config = ModelLoggingSinkConfig(
            max_buffer_size=5000,
            output_format="console",
        )
        ```
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_buffer_size: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of log entries to buffer.",
    )
    output_format: str = Field(
        default="json",
        pattern="^(json|console)$",
        description="Output format: 'json' or 'console'.",
    )


__all__: list[str] = [
    "ModelLoggingSinkConfig",
]
