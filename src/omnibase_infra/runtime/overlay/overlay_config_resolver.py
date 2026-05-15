# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OverlayConfigResolver - resolves overlay files against contract requirements."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from omnibase_core.models.overlay.model_overlay_file import ModelOverlayFile
from omnibase_core.models.overlay.model_overlay_resolution_manifest import (
    ModelOverlayResolutionManifest,
)
from omnibase_infra.runtime.config_discovery.models.model_config_requirements import (
    ModelConfigRequirements,
)
from omnibase_infra.runtime.overlay.errors import RequiredConfigMissingError
from omnibase_infra.runtime.overlay.model_overlay_resolution_result import (
    ModelOverlayResolutionResult,
)

_UNKNOWN_VERSION = "unknown"


def _runtime_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("omnibase-infra")
    except Exception:  # noqa: BLE001  # best-effort version probe
        return _UNKNOWN_VERSION


def _contract_requirements_hash(requirements: ModelConfigRequirements) -> str:
    sorted_keys = sorted(r.key for r in requirements.requirements)
    sorted_transports = sorted(t.value for t in requirements.transport_types)
    sorted_paths = sorted(str(p) for p in requirements.contract_paths)
    payload = json.dumps(
        {
            "contract_paths": sorted_paths,
            "keys": sorted_keys,
            "transports": sorted_transports,
        },
        sort_keys=True,
    )
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _resolved_config_hash(resolved: dict[str, str]) -> str:
    payload = json.dumps({k: resolved[k] for k in sorted(resolved)}, sort_keys=True)
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


class OverlayConfigResolver:
    """Resolves a loaded ModelOverlayFile against contract requirements."""

    def resolve(
        self,
        overlay: ModelOverlayFile,
        requirements: ModelConfigRequirements,
    ) -> ModelOverlayResolutionResult:
        env_pairs = overlay.all_env_pairs()

        resolved: dict[str, str] = {}
        missing_optional: list[str] = []
        missing_required: list[str] = []
        resolved_transport_values: set[str] = set()

        for req in requirements.requirements:
            value = env_pairs.get(req.key)
            if value is not None:
                resolved[req.key] = value
                resolved_transport_values.add(req.transport_type.value)
            elif req.required:
                missing_required.append(req.key)
            else:
                missing_optional.append(req.key)

        if missing_required:
            raise RequiredConfigMissingError(
                f"Required config keys missing from overlay: {sorted(missing_required)}"
            )

        manifest = ModelOverlayResolutionManifest(
            overlay_file_hash=overlay.content_hash(),
            overlay_version=overlay.overlay_version,
            overlay_scope_stack=(overlay.scope.value,),
            contract_requirements_hash=_contract_requirements_hash(requirements),
            resolved_config_hash=_resolved_config_hash(resolved),
            resolved_transports=tuple(sorted(resolved_transport_values)),
            required_transports=tuple(
                sorted(t.value for t in requirements.transport_types)
            ),
            runtime_version=_runtime_version(),
            timestamp=datetime.now(tz=UTC),
            config_source="overlay",
        )

        return ModelOverlayResolutionResult(
            resolved=resolved,
            missing=tuple(sorted(missing_optional)),
            manifest=manifest,
        )


__all__ = ["ModelOverlayResolutionResult", "OverlayConfigResolver"]
