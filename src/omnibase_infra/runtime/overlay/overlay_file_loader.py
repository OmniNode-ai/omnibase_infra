# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OverlayFileLoader: loads and validates overlay YAML files."""

from __future__ import annotations

import logging
import stat
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from omnibase_infra.runtime.overlay.errors import (
    OverlayNotFoundError,
    OverlayPermissionError,
    OverlaySchemaInvalidError,
)

logger = logging.getLogger(__name__)

_ONBOARDING_MSG = (
    "Overlay file not found at '{path}'. "
    "Run onboarding to generate your overlay file: `onex-infra onboard`."
)

# group-read, group-write, other-read, other-write
_OPEN_PERMISSION_MASK = stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH


class OverlayFileLoader:
    """Loads and validates overlay YAML files into typed ModelOverlayFile instances.

    All overlay parsing goes through this loader — no direct yaml.safe_load elsewhere.
    """

    def __init__(self, *, require_restricted_permissions: bool = False) -> None:
        self._require_restricted_permissions = require_restricted_permissions

    def load(self, overlay_path: Path) -> object:
        """Load and validate an overlay YAML file.

        Args:
            overlay_path: Path to the overlay YAML file.

        Returns:
            Validated ModelOverlayFile instance.

        Raises:
            OverlayNotFoundError: File does not exist.
            OverlayPermissionError: File permissions too open (when require_restricted_permissions=True).
            OverlaySchemaInvalidError: YAML parse error or Pydantic validation failure.
        """
        # Import here to allow pytest.importorskip to handle missing core models
        try:
            from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
        except ImportError as exc:
            raise ImportError(
                "omnibase_core.models.overlay.model_overlay_file is not available. "
                "Ensure omnibase_core PR #1082 has been merged and installed."
            ) from exc

        if not overlay_path.exists():
            raise OverlayNotFoundError(_ONBOARDING_MSG.format(path=overlay_path))

        self._check_permissions(overlay_path)

        try:
            raw = yaml.safe_load(overlay_path.read_text())
        except yaml.YAMLError as exc:
            raise OverlaySchemaInvalidError(
                f"Failed to parse overlay YAML at '{overlay_path}': {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise OverlaySchemaInvalidError(
                f"Overlay file at '{overlay_path}' must be a YAML mapping, got {type(raw).__name__}"
            )

        try:
            return ModelOverlayFile.model_validate(raw)
        except ValidationError as exc:
            raise OverlaySchemaInvalidError(
                f"Overlay schema validation failed for '{overlay_path}': {exc}"
            ) from exc

    def _check_permissions(self, path: Path) -> None:
        mode = path.stat().st_mode
        is_open = bool(mode & _OPEN_PERMISSION_MASK)
        if not is_open:
            return
        if self._require_restricted_permissions:
            raise OverlayPermissionError(
                f"Overlay file '{path}' has permissions too open for a secret-containing file. "
                f"Run: chmod 600 '{path}'"
            )
        logger.warning(
            "Overlay file '%s' has open permissions (mode=%s). "
            "Run `chmod 600 '%s'` to restrict access.",
            path,
            oct(mode & 0o777),
            path,
        )
