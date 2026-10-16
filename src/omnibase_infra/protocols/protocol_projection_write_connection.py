# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cursor capability of an already attested projection database connection."""

from contextlib import AbstractContextManager
from typing import Protocol

from omnibase_infra.protocols.protocol_projection_driver_cursor import (
    ProtocolProjectionDriverCursor,
)


class ProtocolProjectionWriteConnection(Protocol):
    """Open a typed cursor on the existing transaction's connection."""

    def cursor(self) -> AbstractContextManager[ProtocolProjectionDriverCursor]:
        """Keep all derived-write statements on this connection."""
        ...
