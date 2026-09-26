# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Market rehydration plan shape accepted by the infra outbox."""

from __future__ import annotations

from typing import Protocol


class ProtocolSimArchiveRehydrationPlan(Protocol):
    """The Market plan shape consumed without importing the Market package."""

    @property
    def target_topic(self) -> str: ...

    @property
    def source_key(self) -> tuple[str, int, int]: ...

    @property
    def key(self) -> bytes | None: ...

    @property
    def value(self) -> bytes: ...

    @property
    def headers(self) -> tuple[tuple[str, bytes | None], ...]: ...

    @property
    def timestamp_ms(self) -> int: ...


__all__ = ["ProtocolSimArchiveRehydrationPlan"]
