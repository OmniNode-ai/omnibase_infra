# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Overlay config resolver — resolves overlay values against contract requirements."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_core.models.overlay.model_overlay_resolution_manifest import (
    ModelOverlayResolutionManifest,
)
from omnibase_infra.runtime.overlay.model_overlay_resolution_result import (
    ModelOverlayResolutionResult,
)


class OverlayConfigResolver:
    """Resolves a loaded ModelOverlayFile against contract requirements.

    When the real Tasks 3/4 implementations merge, this class is replaced by
    omnibase_infra.runtime.overlay.overlay_config_resolver from that PR.
    Until then, this stub resolves all overlay key/value pairs without
    requirement-level filtering.
    """

    def resolve(
        self,
        overlay: ModelOverlayFile,
        requirements: object | None = None,
    ) -> ModelOverlayResolutionResult:
        all_pairs = overlay.all_env_pairs()
        resolved_hash = f"sha256:{hashlib.sha256(json.dumps(all_pairs, sort_keys=True).encode()).hexdigest()}"
        manifest = ModelOverlayResolutionManifest(
            overlay_file_hash=overlay.content_hash(),
            overlay_version=str(overlay.overlay_version),
            overlay_scope_stack=(overlay.scope,),
            contract_requirements_hash=_requirements_hash(requirements),
            resolved_config_hash=resolved_hash,
            resolved_transports=tuple(overlay.transports.keys()),
            required_transports=tuple(overlay.transports.keys()),
            runtime_version="stub",
            timestamp=datetime.now(tz=UTC),
            config_source="overlay",
        )
        return ModelOverlayResolutionResult(
            resolved=all_pairs,
            missing=(),
            manifest=manifest,
        )


def _requirements_hash(requirements: object | None) -> str:
    if requirements is None:
        return "sha256:no-requirements"
    try:
        canonical = json.dumps({"contracts_dir": str(requirements)}, sort_keys=True)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
    except (TypeError, ValueError):
        return "sha256:unserializable-requirements"
