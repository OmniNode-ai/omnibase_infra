# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Models Package.

This module serves as a namespace for Registration Orchestrator models.
Currently, all models are defined in their respective submodules and are not
re-exported from this package to avoid circular imports.

Note:
    Handlers return list[BaseModel] directly rather than a wrapper model.
    This simplifies the API and aligns with ONEX patterns.

    For orchestrator context, use the canonical ModelOrchestratorContext from
    omnibase_core.models.orchestrator, which provides time injection and
    correlation tracking for orchestrator handler execution.

Import Models Directly:
    Registration events and commands should be imported from their source modules:

    - omnibase_infra.models.registration.events for event models
    - omnibase_infra.models.registration.commands for command models
    - omnibase_core.models.orchestrator for ModelOrchestratorContext
"""

__all__: list[str] = []
