# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from .errors import OverlayNotFoundError
from .model_overlay_resolution_result import ModelOverlayResolutionResult
from .overlay_config_resolver import OverlayConfigResolver
from .overlay_file_loader import OverlayFileLoader

logger = logging.getLogger(__name__)

_DEFAULT_OVERLAY_PATH = Path.home() / ".omnibase" / "overlay.yaml"

_TRANSPORT_INDICATORS = frozenset(
    [
        "KAFKA_BOOTSTRAP_SERVERS",
        "POSTGRES_HOST",
        "POSTGRES_PASSWORD",
        "VALKEY_URL",
        "INFISICAL_ADDR",
    ]
)


def load_overlay_config(
    *,
    contracts_dir: Path,
    overlay_path: Path | None = None,
    manifest_path: Path | None = None,
) -> ModelOverlayResolutionResult | None:
    """Load overlay, resolve against contracts, inject into os.environ, write manifest.

    This function is the SOLE authority for os.environ mutation during overlay bootstrap.
    After this returns, injected values are immutable for the process lifetime.

    Cold-start branching:
    - Overlay present → load, resolve, inject, write manifest
    - No overlay + ONEX_REQUIRE_OVERLAY=true → OverlayNotFoundError (overrides migration mode)
    - No overlay + env vars present → None (migration compat mode, deprecation warning)
    - No overlay + no env vars → OverlayNotFoundError with onboarding instructions
    """
    if overlay_path is None:
        overlay_path = Path(
            os.environ.get("ONEX_OVERLAY_PATH", str(_DEFAULT_OVERLAY_PATH))
        )
    if manifest_path is None:
        manifest_path_str = os.environ.get("ONEX_OVERLAY_MANIFEST_PATH")
        manifest_path = Path(manifest_path_str) if manifest_path_str else None

    require_overlay = os.environ.get("ONEX_REQUIRE_OVERLAY", "").lower() == "true"

    if not overlay_path.exists():
        if require_overlay:
            raise OverlayNotFoundError(
                f"ONEX_REQUIRE_OVERLAY=true but overlay not found at {overlay_path}. "
                "Environment variables alone are insufficient when overlay enforcement is enabled. "
                "Run onboarding to generate one."
            )
        if _has_transport_env_vars():
            logger.warning(
                "No overlay file at %s but transport env vars are present. "
                "Running in migration compatibility mode. "
                "Generate an overlay via onboarding to silence this warning.",
                overlay_path,
            )
            return None
        raise OverlayNotFoundError(
            f"No overlay file at {overlay_path} and no transport env vars detected. "
            "Run onboarding to generate an overlay file."
        )

    # Load and resolve (pure — no env mutation yet)
    loader = OverlayFileLoader()
    overlay = loader.load(overlay_path)

    resolver = OverlayConfigResolver(contracts_dir=contracts_dir)
    result = resolver.resolve(overlay)

    # ENV MUTATION BOUNDARY — this is the only place env gets written
    _inject_resolved_pairs(result.resolved_pairs)

    # Write manifest as durable evidence artifact
    if manifest_path:
        _write_manifest(
            manifest_path=manifest_path,
            overlay_path=overlay_path,
            overlay=overlay,
            result=result,
        )

    logger.info(
        "Overlay config loaded: injected=%d keys, skipped=%d existing, unused=%d",
        len(result.resolved_pairs),
        len(result.skipped_existing_keys),
        len(result.unused_overlay_keys),
    )
    return result


def _inject_resolved_pairs(pairs: dict[str, str]) -> None:
    """Inject resolved pairs into os.environ. Called exactly once during bootstrap."""
    for key, value in sorted(pairs.items()):
        os.environ[key] = value


def _write_manifest(
    *,
    manifest_path: Path,
    overlay_path: Path,
    overlay: object,
    result: ModelOverlayResolutionResult,
) -> None:
    """Write resolution manifest as JSON evidence artifact."""
    manifest_data = {
        "overlay_path": str(overlay_path),
        "environment": getattr(overlay, "environment", "unknown"),
        "scope": getattr(overlay, "scope", "unknown"),
        "content_hash": getattr(overlay, "content_hash", lambda: "")(),
        "resolved_pairs_hash": result.resolved_pairs_hash,
        "injected_keys": sorted(result.resolved_pairs.keys()),
        "skipped_existing_keys": sorted(result.skipped_existing_keys),
        "unused_overlay_keys": sorted(result.unused_overlay_keys),
        "required_keys": sorted(result.required_keys),
        "stable_identity_hash": hashlib.sha256(
            json.dumps(
                {
                    "content_hash": getattr(overlay, "content_hash", lambda: "")(),
                    "resolved_pairs_hash": result.resolved_pairs_hash,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True))
    logger.info("Overlay resolution manifest written to %s", manifest_path)


def _has_transport_env_vars() -> bool:
    """Migration-compatibility detection only. NOT authoritative config validation.

    Checks whether common transport env vars exist to distinguish "migrating from
    env-var-based config" from "greenfield with no config at all." Future versions
    should derive required config from contracts, not heuristics.
    """
    return any(key in os.environ for key in _TRANSPORT_INDICATORS)
