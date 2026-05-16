# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Atomic overlay YAML writer with secure permissions and deterministic output."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from contextlib import suppress
from pathlib import Path

import yaml

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile

logger = logging.getLogger(__name__)

_SECRET_KEY_PATTERN = re.compile(
    r"(PASSWORD|SECRET|TOKEN|KEY|CREDENTIAL)", re.IGNORECASE
)


def _contains_secret_keys(overlay: ModelOverlayFile) -> bool:
    key_sections = (
        overlay.secrets.keys(),
        *(vals.keys() for vals in overlay.transports.values()),
        *(vals.keys() for vals in overlay.services.values()),
        *(vals.keys() for vals in overlay.llm.values()),
    )
    return any(
        _SECRET_KEY_PATTERN.search(key) for section in key_sections for key in section
    )


class OverlayWriter:
    """Writes ModelOverlayFile instances to YAML with atomic write and chmod 600."""

    def write(self, overlay: ModelOverlayFile, target_path: Path) -> None:
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if _contains_secret_keys(overlay):
            logger.warning(
                "Writing overlay to %s which contains keys matching secret patterns "
                "(PASSWORD/SECRET/TOKEN/KEY/CREDENTIAL). File will be chmod 600.",
                target_path,
            )

        data = overlay.model_dump(mode="json")
        content = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)

        fd, tmp_path_str = tempfile.mkstemp(
            dir=target_path.parent, prefix=".overlay_tmp_", suffix=".yaml"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(content)
            tmp_path.chmod(0o600)
            tmp_path.replace(target_path)
        except Exception:
            with suppress(OSError):
                tmp_path.unlink()
            raise


__all__ = ["OverlayWriter"]
