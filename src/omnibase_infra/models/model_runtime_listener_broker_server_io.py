# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The broker EFFECT node's injected transport lifecycle (OMN-22379)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict


class ModelRuntimeListenerBrokerServerIo(BaseModel):
    """Start and stop the node-owned Unix socket transport."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    start: Callable[[], None]
    stop: Callable[[], None]


__all__: list[str] = ["ModelRuntimeListenerBrokerServerIo"]
