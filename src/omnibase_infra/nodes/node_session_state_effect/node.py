# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Session State Effect — filesystem I/O for session state management.

This effect node owns ALL filesystem I/O for ``~/.claude/state/``.
No other node touches the session state directory.

Follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Zero custom logic — all behavior from handlers
    - Lightweight shell that delegates to handler implementations

Handlers:
    - HandlerSessionIndexRead: Read session.json
    - HandlerSessionIndexWrite: Atomic write session.json (with flock)
    - HandlerRunContextRead: Read runs/{run_id}.json
    - HandlerRunContextWrite: Atomic write runs/{run_id}.json
    - HandlerStaleRunGC: Remove run docs older than 4hr TTL

Design Decisions:
    - Write atomicity: write .tmp -> fsync -> rename (POSIX atomic)
    - session.json uses flock for concurrent pipeline access
    - Run context files are single-writer (no lock needed)
    - 100% Contract-Driven: All capabilities in YAML, not Python

Related:
    - contract.yaml: Capability definitions and IO operations
    - models/: Session index, run context, and result models
    - handlers/: Filesystem I/O handler implementations

Tracking:
    - OMN-2117: Canonical State Nodes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeSessionStateEffect(NodeEffect):
    """Effect node for session state filesystem operations.

    Capability: session.state

    Provides a capability-oriented interface for session state I/O.
    Uses atomic filesystem operations (write-tmp-fsync-rename) with
    flock-based locking for concurrent pipeline safety.

    This node is declarative — all behavior is defined in contract.yaml
    and implemented through handlers. No custom logic exists in this class.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the session state effect node.

        Args:
            container: ONEX dependency injection container.
        """
        super().__init__(container)


__all__: list[str] = ["NodeSessionStateEffect"]
