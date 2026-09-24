# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The task-class contract's declared fallback, as ``onex delegate`` reads it (OMN-19407)."""

from __future__ import annotations

from typing import Protocol

__all__ = ["ProtocolSelectionFallback"]


class ProtocolSelectionFallback(Protocol):
    """The class an unclaimed prompt resolves to."""

    @property
    def task_class(self) -> str: ...
