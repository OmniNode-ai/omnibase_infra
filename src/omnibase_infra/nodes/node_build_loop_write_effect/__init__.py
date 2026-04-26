# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""node_build_loop_write_effect - build_loop_runs persistence node.

Declarative EFFECT node that consumes ModelPayloadBuildLoopAppend intents
emitted by NodeBuildLoopProjectionCompute and INSERTs them as one row each
into the public.build_loop_runs append-only audit table.

Ticket: OMN-9774
Parent epic: OMN-8943
"""

from omnibase_infra.nodes.node_build_loop_write_effect.handlers import (
    HandlerBuildLoopAppend,
)
from omnibase_infra.nodes.node_build_loop_write_effect.models import (
    ModelBuildLoopAppendResult,
)
from omnibase_infra.nodes.node_build_loop_write_effect.node import (
    NodeBuildLoopWriteEffect,
)
from omnibase_infra.nodes.node_build_loop_write_effect.registry import (
    RegistryInfraBuildLoopWrite,
)

__all__ = [
    "HandlerBuildLoopAppend",
    "ModelBuildLoopAppendResult",
    "NodeBuildLoopWriteEffect",
    "RegistryInfraBuildLoopWrite",
]
