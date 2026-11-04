# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What one start-first replacement of a dev runtime did (OMN-22151)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelRuntimeStartFirstResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    service: str
    old_container: str
    replacement: str
    status: Literal["pass", "fail"]
    failure: Literal[
        "",
        "old-runtime-not-running",
        "replacement-not-started",
        "replacement-exited-before-ready",
        "replacement-never-ready",
        "old-runtime-not-stopped",
    ] = ""
    detail: str = ""
    probes: int = 0
    waited_seconds: float = 0.0
    old_running_at_first_ready: bool = False
