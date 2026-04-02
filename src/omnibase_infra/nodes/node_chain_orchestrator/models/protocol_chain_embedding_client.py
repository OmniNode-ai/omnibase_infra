# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for embedding client used by chain retrieval."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ProtocolChainEmbeddingClient(Protocol):
    """Protocol for embedding client used by chain retrieval."""

    async def get_embedding(
        self,
        text: str,
    ) -> list[float]: ...


__all__ = ["ProtocolChainEmbeddingClient"]
