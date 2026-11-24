# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-resolved transport settings for durable dev boot reports."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class ModelDevBootReportConfig(BaseModel):
    """Configuration injected by the deploy agent's host adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    registry_root: Path
    plugin_dir: Path | None
    ledger_via_bus: bool
    # One delivery pass stops starting new incidents after this many seconds; the rest wait for
    # the next pass (OMN-26316). An incident already running is cut by the per-command timeout.
    pass_seconds: float = 30.0
    # An incident older than this moves to the delivered-abandoned state: its ledger STATUS and
    # exception are no longer worth another attempt (OMN-26316). Seven days.
    abandon_after_seconds: float = 7 * 24 * 3600.0
