# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How the broker's socket server reaches the broker's decisions (OMN-22379)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runtime_listener_broker_request import (
    ModelRuntimeListenerBrokerRequest,
)
from omnibase_infra.models.model_runtime_listener_broker_response import (
    ModelRuntimeListenerBrokerResponse,
)


class ModelRuntimeListenerBrokerPort(BaseModel):
    """``serve`` answers one broker request; the server needs nothing else of the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    serve: Callable[
        [ModelRuntimeListenerBrokerRequest], ModelRuntimeListenerBrokerResponse
    ]


__all__: list[str] = ["ModelRuntimeListenerBrokerPort"]
