# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path
from typing import Protocol
from uuid import UUID


class ProtocolJobStore(Protocol):
    state_dir: Path

    def has_active_job(self) -> bool: ...

    def accept(self, correlation_id: UUID, command: dict[str, object]) -> object: ...
