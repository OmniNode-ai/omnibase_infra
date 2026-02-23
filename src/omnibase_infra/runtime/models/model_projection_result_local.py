# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ModelProjectionResultLocal — local stub until OMN-2508 (omnibase_spi) merges.

OMN-2508 will implement NodeProjectionEffect in omnibase_spi, which includes
a ``ModelProjectionResult`` (success/failure with artifact_ref).  Until that
PR lands and the new omnibase-spi package is published, the runtime uses this
local stub so the pipeline can be fully wired and tested.

Migration path (OMN-2510 follow-up):
    Once omnibase_spi>=0.11.0 exposes ModelProjectionResult, delete this file
    and update all imports to the canonical location from omnibase_spi.

Note:
    Do not confuse this with ``omnibase_core.models.projectors.ModelProjectionResult``
    which tracks per-row SQL projection outcomes.  This local model tracks the
    higher-level effect result: did the synchronous projection effect succeed
    and where is the artifact?
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionResultLocal(BaseModel):
    """Result returned by NodeProjectionEffect.execute().

    Fields:
        success: True when the projection was persisted successfully.
        artifact_ref: Opaque reference to the persisted projection artifact
            (e.g., a database row key or object-store path).  ``None`` on
            failure.
        error: Human-readable error description.  ``None`` on success.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ..., description="True when projection persisted successfully"
    )
    artifact_ref: str | None = Field(
        default=None,
        description="Reference to the persisted projection artifact (None on failure)",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error description (None on success)",
    )

    @classmethod
    def success_result(
        cls, artifact_ref: str | None = None
    ) -> ModelProjectionResultLocal:
        """Build a successful result.

        Args:
            artifact_ref: Optional reference to the persisted artifact.

        Returns:
            A ModelProjectionResultLocal with success=True.
        """
        return cls(success=True, artifact_ref=artifact_ref)

    @classmethod
    def failure_result(cls, error: str) -> ModelProjectionResultLocal:
        """Build a failure result.

        Args:
            error: Human-readable description of what went wrong.

        Returns:
            A ModelProjectionResultLocal with success=False.
        """
        return cls(success=False, error=error)


__all__ = ["ModelProjectionResultLocal"]
