# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-level lifecycle hooks configuration model (OMN-7655)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.models.model_handshake_config import (
    ModelHandshakeConfig,
)
from omnibase_infra.runtime.auto_wiring.models.model_lifecycle_hook_config import (
    ModelLifecycleHookConfig,
)


class ModelLifecycleHooks(BaseModel):
    """Contract-level lifecycle hooks for auto-wiring.

    Phase Ordering:
        1. on_start -- called after container wiring, before consumers start
        2. validate_handshake -- called after on_start, must pass for wiring
        3. on_shutdown -- called during graceful shutdown, before resources close
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
    handshake_config: ModelHandshakeConfig = Field(
        default_factory=ModelHandshakeConfig,
        description="Retry and timeout configuration for handshake validation",
    )
    on_shutdown: ModelLifecycleHookConfig | None = Field(
        default=None,
        description="Hook invoked during graceful node shutdown",
    )

    def has_hooks(self) -> bool:
        """Return True if any lifecycle hook is configured."""
        return any([self.on_start, self.validate_handshake, self.on_shutdown])
