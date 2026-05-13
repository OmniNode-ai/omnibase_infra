# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol key for delegation intent bridge dependency injection."""

from __future__ import annotations

from typing import Protocol

from omnibase_infra.nodes.node_delegation_orchestrator.delegation_intent_bridge import (
    ProtocolLlmCaller,
)


class ProtocolDelegationIntentBridge(Protocol):
    """Protocol key for delegation intent bridge DI lookup."""

    @property
    def llm_caller(self) -> ProtocolLlmCaller | None:
        pass  # Protocol stub — implementation provided by concrete class


__all__ = ["ProtocolDelegationIntentBridge"]
