# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Clock interface consumed by the delegate timing bus."""

from contextlib import AbstractContextManager
from typing import Protocol

from omnibase_infra.enums.enum_delegate_phase import EnumDelegatePhase


class ProtocolDelegatePhaseStopwatch(Protocol):
    """Provide timing spans without importing the CLI implementation."""

    def begin(self, phase: EnumDelegatePhase) -> None:
        """Begin a phase span."""
        ...

    def end(self, phase: EnumDelegatePhase) -> None:
        """End a phase span."""
        ...

    def is_running(self, phase: EnumDelegatePhase) -> bool:
        """Report whether a phase span is open."""
        ...

    def phase(self, phase: EnumDelegatePhase) -> AbstractContextManager[None]:
        """Measure a block, including one that raises."""
        ...
