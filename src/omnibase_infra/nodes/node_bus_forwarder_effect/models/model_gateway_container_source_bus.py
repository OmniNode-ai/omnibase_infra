# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay selector: the forwarder's source bus is the container's bus (OMN-21950)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelGatewayContainerSourceBus(BaseModel):
    """Select the hosting runtime's own ``ProtocolEventBus`` as the source leg.

    The forwarder runs inside the runtime it forwards for. That runtime's bus is
    whatever its overlay registered in the container for ``ProtocolEventBus`` --
    the in-memory bus by default, a hosted bus where the overlay says so -- so
    the source leg carries no broker address and no credential here. The node
    constructs no adapter for it; the runtime hands the resolved bus in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["container"]
    protocol: Literal["ProtocolEventBus"] = "ProtocolEventBus"


__all__ = ["ModelGatewayContainerSourceBus"]
