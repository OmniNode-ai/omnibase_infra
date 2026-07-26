# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodePrStateWriteEffect - declarative EFFECT node.

Persists ModelPayloadPrStateUpsert intents into the public.pr_state
latest-known-state table. All persistence logic lives in
HandlerPrStateUpsert per the ONEX declarative pattern.

Ticket: OMN-14375
"""

from __future__ import annotations

from omnibase_core.container import ModelONEXContainer
from omnibase_core.nodes.node_effect import NodeEffect


class NodePrStateWriteEffect(NodeEffect):
    """Declarative EFFECT node for pr_state persistence.

    All behavior is defined in contract.yaml and delegated to
    HandlerPrStateUpsert. This node contains no custom logic beyond the
    explicit DI constructor required by the nodes/*/node.py guideline.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__ = ["NodePrStateWriteEffect"]
