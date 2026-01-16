# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Capture configuration model for corpus capture service.

.. versionadded:: 0.5.0
    Added for CorpusCapture (OMN-1203)
"""

from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_dedupe_strategy import EnumDedupeStrategy


class ModelCaptureConfig(BaseModel):
    """
    Configuration for corpus capture.

    Controls filtering, deduplication, and limits for captured executions.

    Example:
        >>> config = ModelCaptureConfig(
        ...     corpus_display_name="regression-suite-v1",
        ...     max_executions=50,
        ...     sample_rate=0.5,
        ...     handler_filter=("compute-handler", "effect-handler"),
        ... )

    .. versionadded:: 0.5.0
        Added for CorpusCapture (OMN-1203)
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    corpus_display_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable display name for the corpus.",
    )

    max_executions: int = Field(
        default=50,
        ge=1,
        le=10000,
        description="Maximum number of executions to capture. "
        "Recommended max is 50 for manageable corpus sizes.",
    )

    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of capturing each execution. "
        "1.0 = capture all, 0.1 = capture 10%.",
    )

    handler_filter: tuple[str, ...] | None = Field(
        default=None,
        description="Whitelist of handler IDs to capture. "
        "If None, all handlers are captured.",
    )

    time_window_start: datetime | None = Field(
        default=None,
        description="Start of time window for capture (inclusive). "
        "Executions before this time are skipped.",
    )

    time_window_end: datetime | None = Field(
        default=None,
        description="End of time window for capture (inclusive). "
        "Executions after this time are skipped.",
    )

    dedupe_strategy: EnumDedupeStrategy = Field(
        default=EnumDedupeStrategy.INPUT_HASH,
        description="Strategy for detecting duplicate executions.",
    )

    capture_timeout_ms: float = Field(
        default=50.0,
        ge=1.0,
        le=5000.0,
        description="Maximum time to wait for capture operation (ms). "
        "If exceeded, capture is skipped and recorded as missed.",
    )

    @field_validator("handler_filter", mode="before")
    @classmethod
    def _convert_list_to_tuple(cls, v: Sequence[str] | None) -> tuple[str, ...] | None:
        """Convert sequence to tuple for immutability."""
        if v is None:
            return None
        return tuple(v)

    def has_time_window(self) -> bool:
        """Check if a time window filter is configured."""
        return self.time_window_start is not None or self.time_window_end is not None

    def is_in_time_window(self, timestamp: datetime) -> bool:
        """
        Check if a timestamp falls within the configured time window.

        Args:
            timestamp: The timestamp to check.

        Returns:
            True if within window (or no window configured).
        """
        if self.time_window_start is not None and timestamp < self.time_window_start:
            return False
        if self.time_window_end is not None and timestamp > self.time_window_end:
            return False
        return True

    def is_handler_allowed(self, handler_id: str) -> bool:
        """
        Check if a handler is in the whitelist.

        Args:
            handler_id: The handler ID to check.

        Returns:
            True if allowed (or no filter configured).
        """
        if self.handler_filter is None:
            return True
        return handler_id in self.handler_filter
