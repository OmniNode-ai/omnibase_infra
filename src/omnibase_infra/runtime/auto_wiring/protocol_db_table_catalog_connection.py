# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Minimal catalog connection protocol for typed table validation."""

from typing import Protocol


class ProtocolDbTableCatalogConnection(Protocol):
    """Minimal asyncpg-compatible catalog query surface."""

    async def fetchval(self, query: str, *args: object) -> object | None: ...


__all__ = ["ProtocolDbTableCatalogConnection"]
