# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from datetime import datetime
from typing import Protocol


class ProtocolLaneFlags(Protocol):
    def frozen_reason(self, now: datetime) -> str | None: ...
