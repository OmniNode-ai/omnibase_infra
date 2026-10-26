# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime membership, primary election, self-fencing and state rebuild (OMN-21421).

Implements the runtime primary election model of OMN-21418
(``formal/runtime_primary_election/RuntimePrimaryElection.tla``): one leader key
per ``primary`` node bound to the runtime's session lease, an election epoch
equal to the store revision of the grant, a local self-fence deadline of
``TTL - margin`` after each keepalive was sent, and a write target that
refuses an epoch below the highest it has accepted.
"""

from omnibase_infra.handlers.placement.enum_placement_health_verdict import (
    EnumPlacementHealthVerdict,
)
from omnibase_infra.handlers.placement.handler_placement_election import (
    HandlerPlacementElection,
)
from omnibase_infra.handlers.placement.handler_placement_epoch_guard import (
    HandlerPlacementEpochGuard,
)
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.handlers.placement.handler_placement_rebuild import (
    fold_projection_from_offsets,
)
from omnibase_infra.handlers.placement.handler_placement_runtime import (
    HandlerPlacementRuntime,
    ProtocolPlacedNode,
)
from omnibase_infra.handlers.placement.handler_placement_serving import (
    evaluate_serving_health,
    read_serving_report,
)
from omnibase_infra.handlers.placement.handler_placement_session import (
    HandlerPlacementSession,
)
from omnibase_infra.handlers.placement.model_placement_command import (
    ModelPlacementCommand,
)
from omnibase_infra.handlers.placement.model_placement_policy import (
    ModelPlacementPolicy,
)
from omnibase_infra.handlers.placement.model_placement_replay_event import (
    ModelPlacementReplayEvent,
)
from omnibase_infra.handlers.placement.model_placement_serving_count import (
    ModelPlacementServingCount,
)
from omnibase_infra.handlers.placement.model_placement_serving_report import (
    ModelPlacementServingReport,
)

__all__ = [
    "HandlerPlacementEtcd",
    "EnumPlacementHealthVerdict",
    "ModelPlacementCommand",
    "ModelPlacementPolicy",
    "ModelPlacementReplayEvent",
    "ModelPlacementServingCount",
    "ModelPlacementServingReport",
    "ProtocolPlacedNode",
    "HandlerPlacementElection",
    "HandlerPlacementEpochGuard",
    "HandlerPlacementRuntime",
    "HandlerPlacementSession",
    "evaluate_serving_health",
    "fold_projection_from_offsets",
    "read_serving_report",
]
