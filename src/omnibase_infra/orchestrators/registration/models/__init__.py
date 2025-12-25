# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Models.

This module exports Pydantic models for the Registration Orchestrator.

Note:
    Handlers return list[BaseModel] directly rather than a wrapper model.
    This simplifies the API and aligns with ONEX patterns.

    For orchestrator context, use the canonical ModelOrchestratorContext from
    omnibase_core.models.orchestrator, which provides time injection and
    correlation tracking for orchestrator handler execution.
"""

__all__: list[str] = []
