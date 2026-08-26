# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Result model for the sync-revert watchdog sweep."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_sync_revert_watchdog_effect.models.model_sync_revert_watchdog_outcome import (
    ModelSyncRevertWatchdogOutcome,
)


class ModelSyncRevertWatchdogResult(BaseModel):
    """Result of one sync-revert watchdog sweep run."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Sweep run correlation ID.")
    dry_run: bool = Field(..., description="True unless the request set apply=True.")
    kill_switch_engaged: bool = Field(
        default=False,
        description=(
            "True when ONEX_SYNC_REVERT_WATCHDOG_DISABLED was set — the "
            "watchdog performed zero Linear I/O and returned immediately."
        ),
    )
    issues_scanned: int = Field(default=0, ge=0)
    reverts_detected: int = Field(
        default=0, ge=0, description="Candidate silent automation-reverts found."
    )
    tickets_reflipped: int = Field(default=0, ge=0)
    tickets_skipped: int = Field(default=0, ge=0)
    tickets_errored: int = Field(default=0, ge=0)
    outcomes: tuple[ModelSyncRevertWatchdogOutcome, ...] = Field(
        default_factory=tuple,
        description="One entry per ticket the sweep considered.",
    )
    success: bool = Field(
        default=True,
        description="False only on a sweep-level failure (issue enumeration failed).",
    )
    error_message: str = Field(default="", description="Sweep-level error, if any.")


__all__ = ["ModelSyncRevertWatchdogResult"]
