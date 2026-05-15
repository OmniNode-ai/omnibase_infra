# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Frozen result of OverlayConfigResolver.resolve()."""

from __future__ import annotations

import logging
import os

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.models.overlay.model_overlay_resolution_manifest import (
    ModelOverlayResolutionManifest,
)
from omnibase_infra.runtime.overlay.model_overlay_env_injection_result import (
    ModelOverlayEnvInjectionResult,
)

logger = logging.getLogger(__name__)


class ModelOverlayResolutionResult(BaseModel):
    """Frozen result of resolving overlay config against contract requirements."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    resolved: dict[str, str] = Field(
        ..., description="Keys resolved from the overlay file."
    )
    missing: tuple[str, ...] = Field(
        ..., description="Optional keys that had no value in the overlay."
    )
    manifest: ModelOverlayResolutionManifest = Field(
        ..., description="Evidence artifact for this resolution."
    )

    def apply_to_environment(self) -> ModelOverlayEnvInjectionResult:
        """Inject resolved keys into os.environ. Never overwrites existing keys."""
        injected: list[str] = []
        skipped: list[str] = []
        for key, value in self.resolved.items():
            if key not in os.environ:
                os.environ[key] = value
                injected.append(key)
            else:
                logger.warning(
                    "Overlay key %s skipped: already set in environment", key
                )
                skipped.append(key)
        return ModelOverlayEnvInjectionResult(
            injected_keys=tuple(sorted(injected)),
            skipped_existing_keys=tuple(sorted(skipped)),
        )


__all__ = ["ModelOverlayResolutionResult"]
