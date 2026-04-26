# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""node_build_loop_projection_compute - terminal-event projection node.

Declarative COMPUTE node that subscribes to
onex.evt.omnimarket.build-loop-orchestrator-completed.v1 and emits
ModelIntent payloads for the EFFECT layer to persist into build_loop_runs.

Ticket: OMN-9774
Parent epic: OMN-8943
"""

from omnibase_infra.nodes.node_build_loop_projection_compute.handlers import (
    HandlerBuildLoopProjection,
)
from omnibase_infra.nodes.node_build_loop_projection_compute.models import (
    ModelPayloadBuildLoopAppend,
)
from omnibase_infra.nodes.node_build_loop_projection_compute.node import (
    NodeBuildLoopProjectionCompute,
)
from omnibase_infra.nodes.node_build_loop_projection_compute.registry import (
    RegistryInfraBuildLoopProjection,
)

__all__ = [
    "HandlerBuildLoopProjection",
    "ModelPayloadBuildLoopAppend",
    "NodeBuildLoopProjectionCompute",
    "RegistryInfraBuildLoopProjection",
]
