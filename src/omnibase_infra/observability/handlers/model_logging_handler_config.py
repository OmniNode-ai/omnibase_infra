# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for the structured logging handler.

This module defines the configuration model for HandlerLoggingStructured,
specifying buffer size, flush intervals, output format, and drop policy.

Usage:
    >>> from omnibase_infra.observability.handlers import ModelLoggingHandlerConfig
    >>>
    >>> # Default configuration
    >>> config = ModelLoggingHandlerConfig()
    >>>
    >>> # Custom configuration
    >>> config = ModelLoggingHandlerConfig(
    ...     buffer_size=500,
    ...     flush_interval_seconds=10.0,
    ...     output_format="console",
    ...     drop_policy="drop_newest",
    ... )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ModelLoggingHandlerConfig(BaseModel):
    """Configuration model for the structured logging handler.

    Attributes:
        buffer_size: Maximum number of log entries to buffer before auto-flush
            or drop policy kicks in. Default: 1000.
        flush_interval_seconds: Interval between periodic background flushes.
            Set to 0 to disable periodic flush. Default: 5.0.
        output_format: Output format for log entries. Either "json" for
            machine-readable JSON or "console" for human-readable colored output.
            Default: "json".
        output_file: Optional file path for log output. When set, logs are
            written to both stdout and the specified file. Default: None.
        drop_policy: Policy for handling buffer overflow. "drop_oldest" removes
            the oldest entries when buffer is full (preserves recent logs).
            "drop_newest" rejects new entries (preserves historical logs).
            Default: "drop_oldest".
    """

    buffer_size: int = Field(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum buffer size before drop policy applies",
    )
    flush_interval_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=3600.0,
        description="Periodic flush interval (0 to disable)",
    )
    output_format: Literal["json", "console"] = Field(
        default="json",
        description="Output format for log entries",
    )
    output_file: str | None = Field(
        default=None,
        description="Optional file path for log output",
    )
    drop_policy: Literal["drop_oldest", "drop_newest"] = Field(
        default="drop_oldest",
        description="Buffer overflow handling policy",
    )


__all__: list[str] = [
    "ModelLoggingHandlerConfig",
]
