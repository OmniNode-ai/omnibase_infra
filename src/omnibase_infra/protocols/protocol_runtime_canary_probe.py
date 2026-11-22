# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What a runtime canary run reads from, and asks of, the canary container (OMN-26463)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__: list[str] = ["ProtocolRuntimeCanaryProbe"]


@runtime_checkable
class ProtocolRuntimeCanaryProbe(Protocol):
    """The canary container's own surface; never a dev container."""

    def ready(self) -> bool:
        """True when the canary's /ready answers ready."""
        ...

    def dormant_contracts(self) -> tuple[str, ...]:
        """The contracts the canary booted without waking, by name."""
        ...

    def wake(self, contract: str) -> bool:
        """Send the contract one synthetic first request; True when it was started."""
        ...

    def heartbeat_ok(self) -> bool:
        """True when the runtime's heartbeat is healthy now."""
        ...

    def log_text(self) -> str:
        """The canary container's log from its start."""
        ...
