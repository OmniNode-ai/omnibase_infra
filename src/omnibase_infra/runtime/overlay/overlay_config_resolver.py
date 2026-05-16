# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile

from .errors import RequiredConfigMissingError
from .model_overlay_resolution_result import ModelOverlayResolutionResult

logger = logging.getLogger(__name__)


class OverlayConfigResolver:
    """Pure resolver: computes resolution without mutating os.environ.

    Uses contract dependency declarations to determine which overlay keys
    are required, which are unused, and which are already satisfied by
    existing env vars.
    """

    def __init__(self, *, contracts_dir: Path) -> None:
        self._contracts_dir = contracts_dir

    def resolve(self, overlay: ModelOverlayFile) -> ModelOverlayResolutionResult:
        required_keys = self._extract_required_keys()
        all_overlay_pairs = overlay.all_env_pairs()

        resolved: dict[str, str] = {}
        skipped: set[str] = set()
        missing: set[str] = set()

        for key in required_keys:
            if key in os.environ:
                skipped.add(key)
            elif key in all_overlay_pairs:
                resolved[key] = all_overlay_pairs[key]
            else:
                missing.add(key)

        if missing:
            raise RequiredConfigMissingError(
                f"Contracts require config keys not provided by overlay or environment: "
                f"{sorted(missing)}. Add these to your overlay YAML or set them as env vars."
            )

        unused = (
            frozenset(all_overlay_pairs.keys()) - frozenset(resolved.keys()) - skipped
        )
        if unused:
            logger.info(
                "Overlay contains %d keys not required by any contract: %s",
                len(unused),
                sorted(unused),
            )

        return ModelOverlayResolutionResult(
            resolved_pairs=resolved,
            skipped_existing_keys=frozenset(skipped),
            unused_overlay_keys=unused,
            required_keys=frozenset(required_keys),
        )

    def _extract_required_keys(self) -> set[str]:
        """Extract required env keys from contract YAML dependency declarations."""
        required: set[str] = set()
        for contract_path in self._contracts_dir.rglob("contract.yaml"):
            try:
                raw = yaml.safe_load(contract_path.read_text())
            except (yaml.YAMLError, OSError):
                continue
            if not isinstance(raw, dict):
                continue
            deps = raw.get("dependencies", [])
            if not isinstance(deps, list):
                continue
            for dep in deps:
                if isinstance(dep, dict) and dep.get("type") == "environment":
                    key = dep.get("key")
                    if key:
                        required.add(key)
        return required
