# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ONEX Projection Models Module.

Provides Pydantic models for projection storage, ordering, and snapshot
topic configuration. Used by projectors to persist materialized state and
by orchestrators to query current entity state.

Exports:
    ModelCapabilityFields: Container for capability fields in projection persistence
    ModelContractProjection: Contract projection for Registry API queries
    ModelCursorContract: Cursor mechanism contract for projection replay
    ModelProjectionContract: Freshness and degraded-semantics contract for projections
    ModelProjectionIntent: Intent emitted by reducer to trigger synchronous projection (omnibase_core.models.projectors)
    ModelRegistrationProjection: Registration projection for orchestrator state queries
    ModelRegistrationSnapshot: Compacted snapshot for read optimization
    ModelSequenceInfo: Sequence information for projection ordering and idempotency
    ModelSnapshotTopicConfig: Kafka topic configuration for snapshot publishing
    ModelTopicProjection: Topic projection for Registry API queries

Related Tickets:
    - OMN-1845: Create ProjectionReaderContract for contract/topic queries
    - OMN-1134: Registry Projection Extensions for Capabilities
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
    - OMN-2510: Runtime wires NodeProjectionEffect before Kafka publish
    - OMN-2718: Remove local stub, use omnibase_core canonical ModelProjectionIntent
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.projectors.model_projection_intent import (
        ModelProjectionIntent,
    )
    from omnibase_infra.models.projection.enum_projection_ordering_direction import (
        EnumProjectionOrderingDirection,
    )
    from omnibase_infra.models.projection.model_capability_fields import (
        ModelCapabilityFields,
    )
    from omnibase_infra.models.projection.model_contract_projection import (
        ModelContractProjection,
    )
    from omnibase_infra.models.projection.model_cursor_contract import (
        ModelCursorContract,
    )
    from omnibase_infra.models.projection.model_projected_flag_meta import (
        ModelProjectedFlagMeta,
    )
    from omnibase_infra.models.projection.model_projection_contract import (
        ModelProjectionContract,
    )
    from omnibase_infra.models.projection.model_projection_ordering_contract import (
        DISPATCH_EVAL_RESULTS_ORDERING_CONTRACT,
        PROJECTION_ORDERING_CONTRACTS,
        ModelProjectionOrderingContract,
        get_projection_ordering_contract,
    )
    from omnibase_infra.models.projection.model_registration_projection import (
        ModelRegistrationProjection,
    )
    from omnibase_infra.models.projection.model_registration_snapshot import (
        ModelRegistrationSnapshot,
    )
    from omnibase_infra.models.projection.model_sequence_info import ModelSequenceInfo
    from omnibase_infra.models.projection.model_snapshot_topic_config import (
        ModelSnapshotTopicConfig,
    )
    from omnibase_infra.models.projection.model_topic_projection import (
        ModelTopicProjection,
    )

# OMN-19444: Lazy imports keep `onex <cmd> --help` fast.
_LAZY_EXPORTS: dict[str, str] = {
    "DISPATCH_EVAL_RESULTS_ORDERING_CONTRACT": "omnibase_infra.models.projection.model_projection_ordering_contract",
    "EnumProjectionOrderingDirection": "omnibase_infra.models.projection.enum_projection_ordering_direction",
    "ModelCapabilityFields": "omnibase_infra.models.projection.model_capability_fields",
    "ModelContractProjection": "omnibase_infra.models.projection.model_contract_projection",
    "ModelCursorContract": "omnibase_infra.models.projection.model_cursor_contract",
    "ModelProjectedFlagMeta": "omnibase_infra.models.projection.model_projected_flag_meta",
    "ModelProjectionContract": "omnibase_infra.models.projection.model_projection_contract",
    "ModelProjectionIntent": "omnibase_core.models.projectors.model_projection_intent",
    "ModelProjectionOrderingContract": "omnibase_infra.models.projection.model_projection_ordering_contract",
    "ModelRegistrationProjection": "omnibase_infra.models.projection.model_registration_projection",
    "ModelRegistrationSnapshot": "omnibase_infra.models.projection.model_registration_snapshot",
    "ModelSequenceInfo": "omnibase_infra.models.projection.model_sequence_info",
    "ModelSnapshotTopicConfig": "omnibase_infra.models.projection.model_snapshot_topic_config",
    "ModelTopicProjection": "omnibase_infra.models.projection.model_topic_projection",
    "PROJECTION_ORDERING_CONTRACTS": "omnibase_infra.models.projection.model_projection_ordering_contract",
    "get_projection_ordering_contract": "omnibase_infra.models.projection.model_projection_ordering_contract",
}

__all__ = [
    "ModelCapabilityFields",
    "ModelContractProjection",
    "ModelCursorContract",
    "DISPATCH_EVAL_RESULTS_ORDERING_CONTRACT",
    "EnumProjectionOrderingDirection",
    "ModelProjectedFlagMeta",
    "ModelProjectionContract",
    "ModelProjectionIntent",
    "ModelProjectionOrderingContract",
    "PROJECTION_ORDERING_CONTRACTS",
    "ModelRegistrationProjection",
    "ModelRegistrationSnapshot",
    "ModelSequenceInfo",
    "ModelSnapshotTopicConfig",
    "ModelTopicProjection",
    "get_projection_ordering_contract",
]


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module = importlib.import_module(_LAZY_EXPORTS[name])
        value: object = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
