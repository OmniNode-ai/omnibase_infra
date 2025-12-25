# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for the registration orchestrator node.

This module exports all models used by the NodeRegistrationOrchestrator,
including configuration, input, output, intent, and timeout event models.
"""

from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_intent_payload import (
    ModelConsulIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_registration_intent import (
    ModelConsulRegistrationIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_intent_execution_result import (
    ModelIntentExecutionResult,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_node_liveness_expired import (
    ModelNodeLivenessExpired,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_node_registration_ack_timed_out import (
    ModelNodeRegistrationAckTimedOut,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_orchestrator_config import (
    ModelOrchestratorConfig,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_orchestrator_input import (
    ModelOrchestratorInput,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_orchestrator_output import (
    ModelOrchestratorOutput,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_intent_payload import (
    ModelPostgresIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_upsert_intent import (
    ModelPostgresUpsertIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_registration_intent import (
    IntentPayload,
    ModelRegistrationIntent,
)

__all__ = [
    "IntentPayload",
    "ModelConsulIntentPayload",
    "ModelConsulRegistrationIntent",
    "ModelIntentExecutionResult",
    "ModelNodeLivenessExpired",
    "ModelNodeRegistrationAckTimedOut",
    "ModelOrchestratorConfig",
    "ModelOrchestratorInput",
    "ModelOrchestratorOutput",
    "ModelPostgresIntentPayload",
    "ModelPostgresUpsertIntent",
    "ModelReducerState",
    "ModelRegistrationIntent",
]
