# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay configuration loading, resolution, and error types."""

from omnibase_infra.runtime.overlay.errors import (
    OverlayMergeConflictError,
    OverlayNotFoundError,
    OverlayPermissionError,
    OverlaySchemaInvalidError,
    RequiredConfigMissingError,
    UnsupportedOverlayVersionError,
)
from omnibase_infra.runtime.overlay.model_overlay_env_injection_result import (
    ModelOverlayEnvInjectionResult,
)
from omnibase_infra.runtime.overlay.model_overlay_resolution_result import (
    ModelOverlayResolutionResult,
)
from omnibase_infra.runtime.overlay.overlay_config_resolver import OverlayConfigResolver

__all__ = [
    "ModelOverlayEnvInjectionResult",
    "ModelOverlayResolutionResult",
    "OverlayConfigResolver",
    "OverlayMergeConflictError",
    "OverlayNotFoundError",
    "OverlayPermissionError",
    "OverlaySchemaInvalidError",
    "RequiredConfigMissingError",
    "UnsupportedOverlayVersionError",
]
