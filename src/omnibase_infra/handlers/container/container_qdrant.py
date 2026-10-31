# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for HandlerQdrant (OMN-21683).

The similarity enrichment adapter reaches Qdrant through ``HandlerQdrant``. This
module is the one place that builds it from the ONEX container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.handlers.handler_qdrant import HandlerQdrant

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

__all__: list[str] = ["bind_handler_qdrant"]


def bind_handler_qdrant(container: ModelONEXContainer) -> HandlerQdrant:
    """Build the Qdrant handler on the given container."""
    return HandlerQdrant(container)
