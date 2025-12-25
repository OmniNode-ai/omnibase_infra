# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Orchestrator Models Package.

This package exists as a namespace placeholder for the Registration Orchestrator.
It intentionally exports nothing (`__all__ = []`) because:

1. **No orchestrator-specific models**: The Registration Orchestrator uses shared
   models from the central registration models package, not local submodules.

2. **Circular import prevention**: Re-exporting models here would create circular
   dependencies between orchestrator handlers and model definitions.

3. **Single source of truth**: All registration models live in one canonical
   location for consistency and maintainability.

Note:
    Handlers return ``list[BaseModel]`` directly rather than a wrapper model.
    This simplifies the API and aligns with ONEX patterns.

Import Models From Canonical Locations:
    All registration-related models should be imported from their source packages:

    - ``omnibase_infra.models.registration.events`` - Registration event models
      (e.g., ModelNodeRegistrationInitiated, ModelNodeRegistrationAccepted)

    - ``omnibase_infra.models.registration.commands`` - Registration command models
      (e.g., ModelNodeRegistrationAcked)

    - ``omnibase_core.models.orchestrator`` - Orchestrator infrastructure
      (e.g., ModelOrchestratorContext for time injection and correlation tracking)

Example:
    >>> from omnibase_infra.models.registration.events import (
    ...     ModelNodeRegistrationInitiated,
    ...     ModelNodeRegistrationAccepted,
    ... )
    >>> from omnibase_core.models.orchestrator import ModelOrchestratorContext
"""

__all__: list[str] = []
