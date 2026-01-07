# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration Result Model - Re-exported from node model.

This module re-exports ModelRegistrationResult from the canonical location
in the node_service_discovery_effect models package.

Note:
    This re-export exists for backwards compatibility with existing handler
    imports. New code should import directly from
    ``omnibase_infra.nodes.node_service_discovery_effect.models``.

Related:
    - omnibase_infra.nodes.node_service_discovery_effect.models.model_registration_result:
        Canonical model definition
"""

from omnibase_infra.nodes.node_service_discovery_effect.models.model_registration_result import (
    ModelRegistrationResult,
)

__all__ = ["ModelRegistrationResult"]
