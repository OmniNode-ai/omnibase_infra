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
