# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Protocol
from uuid import UUID


class ProtocolCommand(Protocol):
    """The rebuild command: what the rule reads and what the agent accepts."""

    @property
    def correlation_id(self) -> UUID: ...

    @property
    def requested_by(self) -> str: ...

    def model_dump(self, *, mode: str) -> dict[str, object]: ...

    def model_copy(self, *, update: dict[str, object]) -> ProtocolCommand: ...
