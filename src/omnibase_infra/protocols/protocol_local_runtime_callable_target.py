# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for local runtime callable targets."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class ProtocolLocalRuntimeCallableTarget(Protocol):
    """Object exposing the local runtime call entry point."""

    handle: Callable[..., object]


__all__: list[str] = ["ProtocolLocalRuntimeCallableTarget"]
