# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which node's broker to resolve, under which workspace root (OMN-22362)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelNodeBrokerResolveRequest(BaseModel):
    """Resolve ``node_id``'s broker; ``workspace_root`` locates the config owner and the lane declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overlay_key: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    workspace_root: Path | None = None


__all__: list[str] = ["ModelNodeBrokerResolveRequest"]
