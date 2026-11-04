# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The process-image replacement operation injected into the listener holder (OMN-22379)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict


class ModelRuntimeListenerProcessPort(BaseModel):
    """Replace the agent's image after its descriptors have been remapped."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    replace: Callable[[str, list[str]], None]


__all__: list[str] = ["ModelRuntimeListenerProcessPort"]
