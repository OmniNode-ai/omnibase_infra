# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where the broker's unix socket lives and the listening socket serving it (OMN-22379)."""

from __future__ import annotations

import socket
import threading
from pathlib import Path

from pydantic import BaseModel, ConfigDict, PrivateAttr

from omnibase_infra.models.handlers.model_value_holder import ModelValueHolder


class ModelRuntimeListenerEndpoint(BaseModel):
    """The broker socket path, its lock and the socket once it is listening."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    socket_path: Path
    _guard: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _listening: ModelValueHolder[socket.socket] = PrivateAttr(
        default_factory=ModelValueHolder
    )

    @property
    def guard(self) -> threading.Lock:
        return self._guard

    @property
    def listening(self) -> ModelValueHolder[socket.socket]:
        return self._listening


__all__: list[str] = ["ModelRuntimeListenerEndpoint"]
