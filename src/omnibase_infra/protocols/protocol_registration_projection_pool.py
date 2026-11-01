# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Connection acquisition capability needed by the registration reader."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import asyncpg


class ProtocolRegistrationProjectionPool(Protocol):
    """Borrow a database connection for a registration projection query."""

    def acquire(self) -> AbstractAsyncContextManager[asyncpg.Connection]:
        """Return a context that releases the connection after the query."""
        ...
