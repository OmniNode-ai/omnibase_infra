# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The holder EFFECT node's injected operations (OMN-22379)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from omnibase_infra.models.model_runtime_listener_holder_request import (
    ModelRuntimeListenerHolderRequest,
)
from omnibase_infra.models.model_runtime_listener_holder_result import (
    ModelRuntimeListenerHolderResult,
)


class ModelRuntimeListenerHolderIo(BaseModel):
    """Typed operations supplied by the holder node's I/O runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    install_bind: Callable[
        [ModelRuntimeListenerHolderRequest], ModelRuntimeListenerHolderResult
    ]
    recover: Callable[[], ModelRuntimeListenerHolderResult]
    reboot_rebind: Callable[
        [ModelRuntimeListenerHolderRequest], ModelRuntimeListenerHolderResult
    ]
    prepare_reexec: Callable[[], ModelRuntimeListenerHolderResult]
    reexec: Callable[
        [ModelRuntimeListenerHolderRequest], ModelRuntimeListenerHolderResult
    ]
    drop: Callable[[], ModelRuntimeListenerHolderResult]


__all__: list[str] = ["ModelRuntimeListenerHolderIo"]
