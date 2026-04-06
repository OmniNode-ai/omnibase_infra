# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Lifecycle hooks container model for contract auto-wiring.

Defines the top-level hooks container that declares which lifecycle
phases have hook callables attached.

.. versionadded:: 0.35.0
    Created as part of OMN-7655 (Contract lifecycle hooks).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.config import ModelLifecycleHookConfig


class ModelLifecycleHooks(BaseModel):
    """Contract-level lifecycle hooks for auto-wiring.

    Declares optional hooks that the auto-wiring engine invokes during
    node lifecycle transitions. These replace Plugin.initialize() and
    Plugin.shutdown() with contract-declared, auditable callables.

    Phase Ordering:
        1. on_start — called after container wiring, before consumers start
        2. validate_handshake — called after on_start, must pass for wiring
        3. on_shutdown — called during graceful shutdown, before resources close

    Attributes:
        on_start: Hook invoked during node startup after container wiring.
        validate_handshake: Hook invoked to validate runtime preconditions.
        on_shutdown: Hook invoked during graceful node shutdown.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    on_start: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked during node startup after container wiring",
    )
    validate_handshake: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked to validate runtime preconditions",
    )
    on_shutdown: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked during graceful node shutdown",
    )

    def has_hooks(self) -> bool:
        """Return True if any lifecycle hook is configured."""
        return any([self.on_start, self.validate_handshake, self.on_shutdown])


__all__ = [
    "ModelLifecycleHooks",
]
