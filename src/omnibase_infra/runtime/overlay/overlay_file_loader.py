# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay file loader — parses and validates overlay YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_infra.runtime.overlay.errors import (
    OverlayNotFoundError,
    OverlaySchemaInvalidError,
)

# Legacy env vars that indicate the operator has a .env-style setup.
LEGACY_ENV_SENTINEL_KEYS: tuple[str, ...] = (
    "POSTGRES_HOST",
    "KAFKA_BOOTSTRAP_SERVERS",
    "VALKEY_HOST",
)


class OverlayFileLoader:
    """Loads and validates overlay YAML files into typed ModelOverlayFile instances."""

    def __init__(self, *, require_restricted_permissions: bool = False) -> None:
        self._require_restricted_permissions = require_restricted_permissions

    def load(self, path: Path) -> ModelOverlayFile:
        if not path.exists():
            raise OverlayNotFoundError(
                f"Overlay file not found at {path}. "
                "Run onboarding to generate an overlay file: `onex onboard`"
            )
        if self._require_restricted_permissions:
            mode = path.stat().st_mode
            if mode & 0o044:
                from omnibase_infra.runtime.overlay.errors import OverlayPermissionError

                raise OverlayPermissionError(
                    f"Overlay file {path} has group/other read permissions. "
                    "Restrict with: chmod 600 {path}"
                )
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise OverlaySchemaInvalidError(
                f"YAML parse error in {path}: {exc}"
            ) from exc
        try:
            return ModelOverlayFile.model_validate(raw or {})
        except Exception as exc:
            raise OverlaySchemaInvalidError(
                f"Schema validation failed for {path}: {exc}"
            ) from exc
