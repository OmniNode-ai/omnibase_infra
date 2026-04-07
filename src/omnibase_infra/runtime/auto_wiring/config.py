# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle hook configuration model for contract auto-wiring.

Re-exports ModelLifecycleHookConfig from the canonical models/ location
to maintain backward compatibility with imports from auto_wiring.config.

.. versionadded:: 0.35.0
    Created as part of OMN-7655 (Contract lifecycle hooks).
"""

from omnibase_infra.runtime.auto_wiring.models.model_lifecycle_hook_config import (
    ModelLifecycleHookConfig,
)

__all__ = [
    "ModelLifecycleHookConfig",
]
