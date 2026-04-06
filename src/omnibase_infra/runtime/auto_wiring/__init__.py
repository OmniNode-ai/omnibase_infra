# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring module for contract lifecycle hooks.

Provides schema models and execution logic for contract-level lifecycle hooks
(on_start, validate_handshake, on_shutdown) that replace Plugin.initialize()
and Plugin.shutdown() with declarative, contract-driven lifecycle management.

.. versionadded:: 0.35.0
    Created as part of OMN-7655 (Contract lifecycle hooks).
"""

from omnibase_infra.runtime.auto_wiring.context import ModelAutoWiringContext
from omnibase_infra.runtime.auto_wiring.config import ModelLifecycleHookConfig
from omnibase_infra.runtime.auto_wiring.models import ModelLifecycleHooks
from omnibase_infra.runtime.auto_wiring.result import ModelLifecycleHookResult
from omnibase_infra.runtime.auto_wiring.wiring import LifecycleHookExecutor

__all__ = [
    "LifecycleHookExecutor",
    "ModelAutoWiringContext",
    "ModelLifecycleHookConfig",
    "ModelLifecycleHookResult",
    "ModelLifecycleHooks",
]
