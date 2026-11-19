# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The running count of a runtime's heartbeat publishes (OMN-24559)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRuntimeHeartbeatAccount(BaseModel):
    """What one runtime's heartbeat publishes have done so far.

    A failed publish is counted, never silent: ``record_failure`` raises the
    consecutive-failure streak and keeps the error text, ``record_success``
    clears both and notes that a heartbeat has been published.
    """

    model_config = ConfigDict(extra="forbid")

    consecutive_failures: int = Field(
        default=0, ge=0, description="Heartbeat publishes that failed in a row"
    )
    published_once: bool = Field(
        default=False, description="Whether any heartbeat has been published"
    )
    last_error: str = Field(
        default="", description="Detail of the latest failed heartbeat publish"
    )

    def record_success(self) -> None:
        """Clear the failure streak after a heartbeat published."""
        self.consecutive_failures = 0
        self.published_once = True
        self.last_error = ""

    def record_failure(self, detail: str) -> None:
        """Count one failed heartbeat publish with its detail."""
        self.consecutive_failures += 1
        self.last_error = detail


__all__ = ["ModelRuntimeHeartbeatAccount"]
