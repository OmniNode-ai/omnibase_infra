# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay file loader - parses and validates overlay YAML files."""

from __future__ import annotations

import logging
import stat
from pathlib import Path

import yaml
from pydantic import ValidationError

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_infra.runtime.overlay.errors import (
    OverlayNotFoundError,
    OverlayPermissionError,
    OverlaySchemaInvalidError,
)

logger = logging.getLogger(__name__)

# Legacy env vars that indicate the operator has a .env-style setup.
LEGACY_ENV_SENTINEL_KEYS: tuple[str, ...] = (
    "POSTGRES_HOST",
    "KAFKA_BOOTSTRAP_SERVERS",
    "VALKEY_HOST",
)

_ONBOARDING_MSG = (
    "Overlay file not found at {path}. "
    "Run onboarding to generate an overlay file: `onex onboard`"
)

# group-read, group-write, other-read, other-write
_OPEN_PERMISSION_MASK = stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH


class OverlayFileLoader:
    """Loads and validates overlay YAML files into typed ModelOverlayFile instances."""

    def __init__(self, *, require_restricted_permissions: bool = False) -> None:
        self._require_restricted_permissions = require_restricted_permissions

    def load(self, path: Path) -> ModelOverlayFile:
        if not path.exists():
            raise OverlayNotFoundError(_ONBOARDING_MSG.format(path=path))

        self._check_permissions(path)

        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise OverlaySchemaInvalidError(
                f"YAML parse error in {path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise OverlaySchemaInvalidError(
                f"Overlay file at {path} must be a YAML mapping, got {type(raw).__name__}"
            )

        try:
            return ModelOverlayFile.model_validate(raw)
        except ValidationError as exc:
            raise OverlaySchemaInvalidError(
                f"Schema validation failed for {path}: {exc}"
            ) from exc

    def _check_permissions(self, path: Path) -> None:
        mode = path.stat().st_mode
        if not mode & _OPEN_PERMISSION_MASK:
            return

        if self._require_restricted_permissions:
            raise OverlayPermissionError(
                f"Overlay file {path} has permissions too open for a secret-containing file. "
                f"Restrict with: chmod 600 {path}"
            )

        logger.warning(
            "Overlay file %s has open permissions (mode=%s). "
            "Run `chmod 600 %s` to restrict access.",
            path,
            oct(mode & 0o777),
            path,
        )
