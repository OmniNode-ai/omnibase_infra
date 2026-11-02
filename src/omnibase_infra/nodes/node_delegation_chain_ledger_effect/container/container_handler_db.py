# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Binding site for this node's HandlerDb (OMN-21623).

The node's write handlers compose a PostgreSQL ``HandlerDb`` over their container
(OMN-14140). This module is the one place in the node that constructs it, so no
handler builds it by hand. The binding does not look anything up in the container
and does not share instances: every call returns a new ``HandlerDb`` that the
calling handler initializes with its own DSN and shuts down with itself, so a
handler's shutdown never closes a pool another handler is using.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_infra.handlers.handler_db import HandlerDb

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer

__all__: list[str] = ["bind_handler_db"]


def bind_handler_db(container: ModelONEXContainer) -> HandlerDb:
    """Build a new ``HandlerDb`` over ``container``, owned by the caller.

    The caller initializes the returned handler and shuts it down; no other
    handler holds a reference to it.
    """
    return HandlerDb(container)
