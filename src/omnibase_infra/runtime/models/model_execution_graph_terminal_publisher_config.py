# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-bound signing identity and topic for graph workflow terminals."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelExecutionGraphTerminalPublisherConfig(BaseModel):
    """Terminal route and runtime signer identity supplied by workflow composition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal_topic: str = Field(min_length=1)
    runtime_id: str = Field(min_length=1)  # string-id-ok: named runtime signer
    realm: str = Field(min_length=1)
    bus_id: str = Field(min_length=1)  # string-id-ok: named message bus
    workflow_type: str = Field(min_length=1)


__all__ = ["ModelExecutionGraphTerminalPublisherConfig"]
