# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for the registration orchestrator node.

This module exports all models used by the NodeRegistrationOrchestrator,
including configuration, input, and output models.
"""

from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_intent_execution_result import (
    ModelIntentExecutionResult,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_orchestrator_config import (
    ModelOrchestratorConfig,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_orchestrator_input import (
    ModelOrchestratorInput,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_orchestrator_output import (
    ModelOrchestratorOutput,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_reducer_state import (
    ModelReducerState,
)
from omnibase_infra.nodes.node_registration_orchestrator.v1_0_0.models.model_registration_intent import (
    ModelRegistrationIntent,
)

__all__ = [
    "ModelIntentExecutionResult",
    "ModelOrchestratorConfig",
    "ModelOrchestratorInput",
    "ModelOrchestratorOutput",
    "ModelReducerState",
    "ModelRegistrationIntent",
]
