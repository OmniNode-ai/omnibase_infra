# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Models.

This module exports Pydantic models for the Registration Orchestrator.

Exports:
    ModelOrchestratorContext: Context for orchestrator handler execution

Note:
    Handlers return list[BaseModel] directly rather than a wrapper model.
    This simplifies the API and aligns with ONEX patterns.
"""

from omnibase_infra.orchestrators.registration.models.model_orchestrator_context import (
    ModelOrchestratorContext,
)

__all__: list[str] = [
    "ModelOrchestratorContext",
]
