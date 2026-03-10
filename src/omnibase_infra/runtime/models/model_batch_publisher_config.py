# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Configuration model for BatchResponsePublisher (OMN-478)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Default configuration values
DEFAULT_BATCH_SIZE = 10
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 1000

DEFAULT_FLUSH_INTERVAL_MS = 100.0
MIN_FLUSH_INTERVAL_MS = 10.0
MAX_FLUSH_INTERVAL_MS = 5000.0


class ModelBatchPublisherConfig(BaseModel):
    """Configuration for BatchResponsePublisher.

    Attributes:
        batch_size: Maximum number of responses to buffer before flushing.
        flush_interval_ms: Maximum time in milliseconds to wait before flushing.
        enabled: Whether batch publishing is enabled. When False, responses
            are published immediately (passthrough mode).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(
        default=DEFAULT_BATCH_SIZE,
        ge=MIN_BATCH_SIZE,
        le=MAX_BATCH_SIZE,
        description="Maximum number of responses to buffer before flushing.",
    )
    flush_interval_ms: float = Field(
        default=DEFAULT_FLUSH_INTERVAL_MS,
        ge=MIN_FLUSH_INTERVAL_MS,
        le=MAX_FLUSH_INTERVAL_MS,
        description="Maximum time in milliseconds before flushing buffered responses.",
    )
    enabled: bool = Field(
        default=False,
        description="Whether batch publishing is enabled.",
    )
