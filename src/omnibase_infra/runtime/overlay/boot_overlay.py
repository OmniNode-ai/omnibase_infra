# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Extracted, unit-testable helper for overlay config loading at runtime boot.

Once Tasks 1-4 PRs merge, replace the local stub imports with:
  - ModelOverlayFile: omnibase_core.models.overlay.model_overlay_file
  - ModelOverlayResolutionManifest: omnibase_core.models.overlay.model_overlay_resolution_manifest
  - OverlayFileLoader: omnibase_infra.runtime.overlay.overlay_file_loader (real impl)
  - OverlayConfigResolver: omnibase_infra.runtime.overlay.overlay_config_resolver (real impl)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from omnibase_infra.runtime.config_discovery.contract_config_extractor import (
    ContractConfigExtractor,
)
from omnibase_infra.runtime.overlay.errors import OverlayNotFoundError
from omnibase_infra.runtime.overlay.model_overlay_resolution_result import (
    ModelOverlayResolutionResult,
)
from omnibase_infra.runtime.overlay.overlay_config_resolver import OverlayConfigResolver
from omnibase_infra.runtime.overlay.overlay_file_loader import (
    LEGACY_ENV_SENTINEL_KEYS,
    OverlayFileLoader,
)

logger = logging.getLogger(__name__)


def _legacy_env_vars_present() -> bool:
    return any(os.environ.get(k) for k in LEGACY_ENV_SENTINEL_KEYS)


def load_overlay_config(
    *,
    overlay_path: Path,
    contracts_dir: Path,
    require_overlay: bool = False,
) -> ModelOverlayResolutionResult | None:
    """Load and resolve overlay config for runtime boot.

    Returns a resolved result ready for env injection, or None when running
    in legacy env-var compat mode (overlay absent but legacy vars present).

    Raises OverlayNotFoundError when:
    - require_overlay=True and overlay file is missing (regardless of env vars)
    - overlay file is missing AND no legacy env vars detected (cold-start)
    """
    try:
        overlay = OverlayFileLoader().load(overlay_path)
    except OverlayNotFoundError:
        if require_overlay:
            raise OverlayNotFoundError(
                f"Overlay file required but not found at {overlay_path}. "
                "ONEX_REQUIRE_OVERLAY=true enforces overlay-only boot. "
                "Run onboarding to generate an overlay file: `onex onboard`"
            )
        if _legacy_env_vars_present():
            logger.warning(
                "No overlay file found at %s but legacy env vars detected. "
                "Running in env-var compatibility mode. "
                "Migrate by running: `onex onboard --generate-overlay`. "
                "This fallback will be removed in a future release.",
                overlay_path,
            )
            return None
        raise OverlayNotFoundError(
            f"No overlay file found at {overlay_path} and no legacy env vars detected. "
            "This appears to be a fresh install. "
            "Run onboarding to configure the runtime: `onex onboard`"
        )

    requirements = ContractConfigExtractor().extract_from_paths([contracts_dir])
    return OverlayConfigResolver().resolve(overlay, requirements)
