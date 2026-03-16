# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Metrics model for BatchResponsePublisher (OMN-478)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBatchPublisherMetrics(BaseModel):
    """Metrics for BatchResponsePublisher.

    Attributes:
        total_enqueued: Total number of responses enqueued.
        total_published: Total number of responses successfully published.
        total_failed: Total number of responses that failed to publish.
        total_batches_flushed: Total number of batch flush operations.
        total_timeout_flushes: Number of flushes triggered by timeout.
        total_size_flushes: Number of flushes triggered by batch size threshold.
    """

    model_config = ConfigDict(frozen=False, extra="forbid")

    total_enqueued: int = Field(default=0)
    total_published: int = Field(default=0)
    total_failed: int = Field(default=0)
    total_batches_flushed: int = Field(default=0)
    total_timeout_flushes: int = Field(default=0)
    total_size_flushes: int = Field(default=0)
