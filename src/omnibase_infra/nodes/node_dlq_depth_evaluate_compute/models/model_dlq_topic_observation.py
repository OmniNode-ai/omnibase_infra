# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One read-only broker observation of a single DLQ topic (OMN-16769).

This is BROKER TRUTH, not a derived metric: every field is a value the
Kafka/Redpanda admin surface reports directly. All derivation
(retained depth, arrivals-in-window, rate) happens in the pure compute
handler so it is testable without a broker.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelDlqTopicObservation(BaseModel):
    """Per-topic offset triple summed across the topic's partitions.

    Offsets are summed across partitions deliberately: this monitor asks
    "how much landed in this sink", which is a whole-topic question. A
    per-partition breakdown would not change any verdict and would
    multiply the row count by the partition count for no decision value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., min_length=1, description="Fully-qualified DLQ topic name.")
    partition_count: int = Field(
        ..., ge=1, description="Number of partitions the offsets below are summed over."
    )
    log_start_offset: int = Field(
        ...,
        ge=0,
        description=(
            "Sum of per-partition log-start offsets. This MOVES as retention "
            "trims the log — it is not always 0. Proven live on the .201 dev "
            "lane 2026-08-27: onex.dlq.omnibase-infra.events.v1 reported a "
            "log-start of 8,157,557 against a high-water mark of 8,170,442. "
            "Treating the high-water mark as 'depth' would have overstated "
            "that topic's retained backlog by ~634x."
        ),
    )
    high_watermark: int = Field(
        ...,
        ge=0,
        description="Sum of per-partition high-water marks (lifetime records ever written).",
    )
    window_start_offset: int = Field(
        ...,
        ge=0,
        description=(
            "Sum of per-partition offsets as of the start of the evaluation "
            "window, resolved from the broker's own offset-for-timestamp "
            "index. Equals high_watermark when nothing arrived in the window "
            "(the broker returns no offset for a timestamp past the last "
            "record, and the probe normalizes that to the high-water mark "
            "rather than guessing)."
        ),
    )

    @model_validator(mode="after")
    def _validate_offset_ordering(self) -> ModelDlqTopicObservation:
        """Reject physically impossible offset triples rather than deriving nonsense.

        A negative retained depth or a negative arrival count is not a
        recoverable measurement — it means the probe mixed samples from
        different points in time, or read a partition set that changed
        mid-probe. Failing closed here keeps a garbage reading from
        being materialized as a confident-looking zero.
        """
        if self.high_watermark < self.log_start_offset:
            raise ValueError(
                f"{self.topic}: high_watermark ({self.high_watermark}) is behind "
                f"log_start_offset ({self.log_start_offset}) — impossible offset "
                "triple, refusing to derive a negative retained depth."
            )
        if self.window_start_offset > self.high_watermark:
            raise ValueError(
                f"{self.topic}: window_start_offset ({self.window_start_offset}) is "
                f"ahead of high_watermark ({self.high_watermark}) — impossible offset "
                "triple, refusing to derive a negative arrival count."
            )
        if self.window_start_offset < self.log_start_offset:
            raise ValueError(
                f"{self.topic}: window_start_offset ({self.window_start_offset}) is "
                f"behind log_start_offset ({self.log_start_offset}) — the window "
                "start has been retention-trimmed away; the probe must clamp to "
                "log_start_offset rather than report an arrival count that "
                "includes already-deleted records."
            )
        return self

    @property
    def retained_depth(self) -> int:
        """Records still on disk. NOT the high-water mark — retention moves log-start."""
        return self.high_watermark - self.log_start_offset

    @property
    def arrivals_in_window(self) -> int:
        """Records that landed in this sink during the evaluation window."""
        return self.high_watermark - self.window_start_offset


__all__ = ["ModelDlqTopicObservation"]
