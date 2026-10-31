# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for HandlerDb (OMN-21623).

The ledger and projection write handlers compose a PostgreSQL ``HandlerDb`` from
their container (OMN-14140). This module is the one place that builds it, so a
caller resolves the handler here and never constructs it by hand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.handlers.handler_db import HandlerDb

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

__all__: list[str] = ["bind_handler_db"]


def bind_handler_db(container: ModelONEXContainer) -> HandlerDb:
    """Build the container's ``HandlerDb``."""
    return HandlerDb(container)
