# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Opt-in final-evidence observation at the delegation terminal boundary."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from omnibase_infra.runtime.models.model_delegation_terminal_evidence import (
    ModelDelegationTerminalEvidence,
)


class ProtocolDelegationTerminalEvidenceSink(Protocol):
    """Persist final-run terminal evidence after dispatch receives the terminal."""

    def __call__(
        self, evidence: ModelDelegationTerminalEvidence
    ) -> Awaitable[None]: ...


__all__ = [
    "ModelDelegationTerminalEvidence",
    "ProtocolDelegationTerminalEvidenceSink",
]
