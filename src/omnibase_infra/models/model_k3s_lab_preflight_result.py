# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the lab k3s preflight found when it judged a server start (OMN-24956)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelK3sLabPreflightResult(BaseModel):
    """The refusal reading; ``verdict`` is ``pass`` when the server may start."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Literal["pass", "refuse"]
    reason: str = ""
    detail: str = ""
    node_ip: str = ""
    flannel_iface: str = ""
    live_interface: str = ""
