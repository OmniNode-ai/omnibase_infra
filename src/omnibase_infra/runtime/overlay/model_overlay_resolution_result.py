# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Resolution result model for overlay config resolver."""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.overlay.model_overlay_env_injection_result import (
    ModelOverlayEnvInjectionResult,
)
from omnibase_infra.runtime.overlay.model_overlay_resolution_manifest import (
    ModelOverlayResolutionManifest,
)

logger = logging.getLogger(__name__)


class ModelOverlayResolutionResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    resolved: dict[str, str] = Field(default_factory=dict)
    missing: tuple[str, ...] = Field(default_factory=tuple)
    manifest: ModelOverlayResolutionManifest

    def apply_to_environment(self) -> ModelOverlayEnvInjectionResult:
        injected: list[str] = []
        skipped: list[str] = []
        for key, value in self.resolved.items():
            if key in os.environ:
                logger.warning(
                    "Overlay key %s already set in environment (existing=%s); skipping",
                    key,
                    os.environ[key],
                )
                skipped.append(key)
            else:
                os.environ[key] = value
                injected.append(key)
        return ModelOverlayEnvInjectionResult(
            injected_keys=tuple(injected),
            skipped_existing_keys=tuple(skipped),
        )
