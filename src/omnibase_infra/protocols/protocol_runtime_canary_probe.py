# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a runtime canary run reads from, and asks of, the canary container (OMN-26463)."""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__: list[str] = ["ProtocolRuntimeCanaryProbe"]


class ProtocolRuntimeCanaryProbe(ABC):
    """The canary container's own surface; never a dev container."""

    @abstractmethod
    def ready(self) -> bool:
        """True when the canary's /ready answers ready."""
        ...

    @abstractmethod
    def dormant_contracts(self) -> tuple[str, ...]:
        """The contracts the canary booted without waking, by name."""
        ...

    @abstractmethod
    def wake(self, contract: str) -> bool:
        """Send the contract one synthetic first request; True when it was started."""
        ...

    @abstractmethod
    def heartbeat_ok(self) -> bool:
        """True when the runtime's heartbeat is healthy now."""
        ...

    @abstractmethod
    def log_text(self) -> str:
        """The canary container's log from its start."""
        ...
