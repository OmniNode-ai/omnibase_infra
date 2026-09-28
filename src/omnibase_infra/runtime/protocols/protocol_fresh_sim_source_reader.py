# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exact-coordinate source reader boundary for disposable replay validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omnibase_infra.runtime.sim_archive_source_receipt import RawSimArchiveRecord


class ProtocolFreshSimSourceReader(Protocol):
    """Trusted, group-less exact-coordinate broker reader injected by infra."""

    async def read_exact(
        self, topic: str, partition: int, offset: int
    ) -> RawSimArchiveRecord | None: ...


__all__ = ["ProtocolFreshSimSourceReader"]
