# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""node_pr_state_projection_compute - GitHub PR status projection node.

Declarative COMPUTE node that subscribes to onex.evt.github.pr-status.v1
(published by node_github_pr_poller_effect) and emits ModelIntent payloads
for the EFFECT layer to persist into pr_state.

Ticket: OMN-14375
"""

from omnibase_infra.nodes.node_pr_state_projection_compute.handlers import (
    HandlerPrStateProjection,
)
from omnibase_infra.nodes.node_pr_state_projection_compute.models import (
    ModelPayloadPrStateUpsert,
)
from omnibase_infra.nodes.node_pr_state_projection_compute.node import (
    NodePrStateProjectionCompute,
)
from omnibase_infra.nodes.node_pr_state_projection_compute.registry import (
    RegistryInfraPrStateProjection,
)

__all__ = [
    "HandlerPrStateProjection",
    "ModelPayloadPrStateUpsert",
    "NodePrStateProjectionCompute",
    "RegistryInfraPrStateProjection",
]
