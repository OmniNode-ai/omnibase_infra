# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeBuildLoopWriteEffect - declarative EFFECT node.

Persists ModelPayloadBuildLoopAppend intents into the public.build_loop_runs
append-only audit table. All persistence logic lives in
HandlerBuildLoopAppend per the ONEX declarative pattern.

Ticket: OMN-9774
"""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodeBuildLoopWriteEffect(NodeEffect):
    """Declarative EFFECT node for build_loop_runs persistence.

    All behavior is defined in contract.yaml and delegated to
    HandlerBuildLoopAppend. This node contains no custom logic beyond the
    explicit DI constructor required by the nodes/*/node.py guideline.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodeBuildLoopWriteEffect"]
