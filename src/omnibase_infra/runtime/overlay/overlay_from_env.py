# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile

from .overlay_writer import OverlayWriter

_DEFAULT_OVERLAY_OUTPUT = Path.home() / ".omnibase" / "overlay.yaml"


def overlay_from_env_dict(
    env_dict: dict[str, str],
    *,
    output_path: Path | None = None,
    environment: str = "local",
    scope: str = "env",
) -> Path:
    """Generate an overlay YAML file from an env var dict.

    Used by onboarding to produce the initial overlay file from discovered
    or user-supplied configuration values.

    Production/default onboarding targets ~/.omnibase/overlay.yaml.
    Tests and automation MUST pass explicit temp output paths to avoid
    mutating real state.
    """
    if output_path is None:
        output_path = _DEFAULT_OVERLAY_OUTPUT

    overlay = ModelOverlayFile.model_validate(
        {
            "overlay_version": "1.0.0",
            "environment": environment,
            "scope": scope,
            "transports": {"custom": env_dict},
        }
    )
    OverlayWriter().write(overlay, output_path)
    return output_path
