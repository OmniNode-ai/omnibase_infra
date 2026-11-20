# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the lab k3s preflight is handed before a server start (OMN-24956)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelK3sLabPreflightRequest(BaseModel):
    """The k3s server config text and, optionally, the host's interface table."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    config_text: str
    interface_table: str = ""
    require_rendered: bool = True
