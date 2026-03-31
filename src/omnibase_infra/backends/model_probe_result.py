# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Probe result model for onex.backends health probes."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.backends.enum_probe_state import EnumProbeState


class ModelProbeResult(BaseModel):
    """Structured result from a backend health probe."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: EnumProbeState
    reason: str
    backend_label: str
