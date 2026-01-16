# SPDX-FileCopyrightText: 2025 OmniNode Team <info@omninode.ai>
#
# SPDX-License-Identifier: Apache-2.0
"""
Capture result model for corpus capture service.

.. versionadded:: 0.5.0
    Added for CorpusCapture (OMN-1203)
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.enums.enum_capture_outcome import EnumCaptureOutcome


class ModelCaptureResult(BaseModel):
    """
    Result of a single capture attempt.

    Tracks whether an execution was captured or skipped, and why.

    .. versionadded:: 0.5.0
        Added for CorpusCapture (OMN-1203)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest_id: UUID = Field(
        ...,
        description="ID of the manifest that was processed.",
    )

    outcome: EnumCaptureOutcome = Field(
        ...,
        description="Result of the capture attempt.",
    )

    captured_at: datetime | None = Field(
        default=None,
        description="Timestamp when captured (None if skipped).",
    )

    dedupe_hash: str | None = Field(
        default=None,
        description="Hash used for deduplication (if computed).",
    )

    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to process the capture attempt.",
    )

    error_message: str | None = Field(
        default=None,
        description="Error message if capture failed.",
    )

    def __bool__(self) -> bool:
        """
        Return True if the execution was captured.

        Warning:
            This is non-standard Pydantic behavior. Normally ``bool(model)``
            returns True for any model instance. This override enables
            idiomatic ``if result:`` checks for capture success.
        """
        return self.outcome == EnumCaptureOutcome.CAPTURED

    @property
    def was_captured(self) -> bool:
        """Check if the execution was captured."""
        return self.outcome == EnumCaptureOutcome.CAPTURED

    @property
    def was_skipped(self) -> bool:
        """Check if the execution was skipped (intentionally not captured)."""
        return self.outcome.value.startswith("skipped_")

    @property
    def was_failed(self) -> bool:
        """Check if the capture failed due to an error."""
        return self.outcome == EnumCaptureOutcome.FAILED
