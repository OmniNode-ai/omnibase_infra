# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Where the dev agent records the canary receipts its roll gate reads (OMN-26463)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

__all__: list[str] = ["ProtocolRuntimeCanaryReceiptDirectory"]


class ProtocolRuntimeCanaryReceiptDirectory(ABC):
    """The agent's state directory, bound by the host composition root."""

    @abstractmethod
    def state_dir(self) -> Path:
        """The deploy agent's state directory; the receipts go in its canary-receipts child."""
        ...
