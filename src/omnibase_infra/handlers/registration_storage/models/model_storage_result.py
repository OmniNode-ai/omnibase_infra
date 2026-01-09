# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Storage Result Model - Re-exported from node model.

This module re-exports ModelStorageResult from the canonical location
in the node_registration_storage_effect models package.

Note:
    This re-export exists for backwards compatibility with existing handler
    imports. New code should import directly from
    ``omnibase_infra.nodes.node_registration_storage_effect.models``.

Related:
    - omnibase_infra.nodes.node_registration_storage_effect.models.model_storage_result:
        Canonical model definition
"""

from omnibase_infra.nodes.node_registration_storage_effect.models.model_storage_result import (
    ModelStorageResult,
)

__all__ = ["ModelStorageResult"]
