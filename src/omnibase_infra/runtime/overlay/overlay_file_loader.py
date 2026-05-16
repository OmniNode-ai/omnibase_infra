# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import stat
from pathlib import Path

import yaml
from pydantic import ValidationError

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile

from .errors import (
    OverlayNotFoundError,
    OverlayPermissionError,
    OverlaySchemaInvalidError,
)

logger = logging.getLogger(__name__)


class OverlayFileLoader:
    def __init__(self, *, require_restricted_permissions: bool = True) -> None:
        self._require_restricted = require_restricted_permissions

    def load(self, path: Path) -> ModelOverlayFile:
        if not path.exists():
            raise OverlayNotFoundError(
                f"Overlay file not found: {path}. Run onboarding to generate one."
            )

        self._check_permissions(path)

        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise OverlaySchemaInvalidError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise OverlaySchemaInvalidError(
                f"Overlay must be a YAML mapping, got {type(raw).__name__}"
            )

        try:
            return ModelOverlayFile.model_validate(raw)
        except ValidationError as exc:
            raise OverlaySchemaInvalidError(
                f"Overlay schema validation failed for {path}: {exc}"
            ) from exc

    def _check_permissions(self, path: Path) -> None:
        file_stat = path.stat()
        mode = stat.S_IMODE(file_stat.st_mode)
        group_or_other = mode & (stat.S_IRWXG | stat.S_IRWXO)
        if group_or_other:
            msg = (
                f"Overlay file {path} is accessible by group/other "
                f"(mode={oct(mode)}). Expected 0600."
            )
            if self._require_restricted:
                raise OverlayPermissionError(msg)
            logger.warning(msg)
