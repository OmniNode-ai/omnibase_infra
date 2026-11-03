# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Service lookup the listener handlers resolve their collaborators from (OMN-22379)."""

from __future__ import annotations

from typing import Protocol


class ProtocolRuntimeListenerLookup(Protocol):
    """The slice of ModelONEXContainer the listener handlers use: resolve by type."""

    def get_service[T](self, protocol_type: type[T]) -> T:
        """Return the registered instance of ``protocol_type``."""
        ...


__all__: list[str] = ["ProtocolRuntimeListenerLookup"]
