# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""node_pr_state_write_effect - pr_state persistence node.

Declarative EFFECT node that consumes ModelPayloadPrStateUpsert intents
emitted by NodePrStateProjectionCompute and UPSERTs them into the
public.pr_state latest-known-state table.

Ticket: OMN-14375
"""

from omnibase_infra.nodes.node_pr_state_write_effect.handlers import (
    HandlerPrStateUpsert,
)
from omnibase_infra.nodes.node_pr_state_write_effect.models import (
    ModelPrStateUpsertResult,
)
from omnibase_infra.nodes.node_pr_state_write_effect.node import (
    NodePrStateWriteEffect,
)
from omnibase_infra.nodes.node_pr_state_write_effect.registry import (
    RegistryInfraPrStateWrite,
)

__all__ = [
    "HandlerPrStateUpsert",
    "ModelPrStateUpsertResult",
    "NodePrStateWriteEffect",
    "RegistryInfraPrStateWrite",
]
