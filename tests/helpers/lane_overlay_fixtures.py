# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Neutral lane-overlay fixtures for tests (OMN-20932).

The lab's lane overlays are private deployment config and are not in this
repository (operator RULING 2026-10-10T21:57:38Z). Tests that need a realistic
overlay build one here, with placeholder endpoints that resolve nowhere, and
every lane compose file mounts
``${ONEX_LANE_OVERLAY_DIR}/${COMPOSE_PROJECT_NAME}.bifrost.yaml`` at one
lane-neutral path, spelled once below so every compose test asserts the same bind.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

#: The ``:?`` message every lane compose file attaches to its overlay mount.
LANE_OVERLAY_DIR_MESSAGE = (
    "ONEX_LANE_OVERLAY_DIR must name the directory holding your "
    "COMPOSE_PROJECT_NAME.bifrost.yaml "
    "(config/lab/lane-overlays under ONEX_WORKSPACE_CONFIG_ROOT); "
    "this repository ships no lab lane overlay"
)

#: The one lane-neutral container path every lane's runtime reads its overlay at.
LANE_OVERLAY_TARGET = "/app/config/delegation/lane.bifrost.yaml"

#: The bind every lane compose file declares once (``x-lane-overlay-mount``) and
#: lists in each runtime service's volumes, as YAML loads it.
LANE_OVERLAY_MOUNT: dict[str, Any] = {
    "type": "bind",
    "source": (
        f"${{ONEX_LANE_OVERLAY_DIR:?{LANE_OVERLAY_DIR_MESSAGE}}}"
        "/${COMPOSE_PROJECT_NAME}.bifrost.yaml"
    ),
    "target": LANE_OVERLAY_TARGET,
    "read_only": True,
    "bind": {"create_host_path": False},
}

_COMPOSE_TAG = re.compile(r"(?<=\s)!(override|reset)(?=\s)")


def load_compose(path: Path) -> dict[str, Any]:
    """Load a compose file with YAML anchors resolved and merge tags dropped."""
    text = _COMPOSE_TAG.sub("", path.read_text(encoding="utf-8"))
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict)
    return loaded


def lane_overlay_mounts(path: Path) -> dict[str, list[Any]]:
    """Each service's volume entries that bind a Bifrost lane overlay."""
    services = load_compose(path).get("services") or {}
    found: dict[str, list[Any]] = {}
    for name, service in services.items():
        mounts = [
            volume
            for volume in (service or {}).get("volumes") or []
            if (isinstance(volume, dict) and "bifrost" in str(volume.get("target", "")))
            or (isinstance(volume, str) and ".bifrost.yaml" in volume)
        ]
        if mounts:
            found[name] = mounts
    return found


#: A placeholder chat-completions endpoint (RFC 2606 ``.invalid``).
NEUTRAL_CHAT_ENDPOINT = "http://model.example.invalid:8000/v1/chat/completions"
#: A placeholder embeddings endpoint on the same placeholder host.
NEUTRAL_EMBEDDINGS_ENDPOINT = "http://model.example.invalid:8002/v1/embeddings"
#: The served id the neutral template (local.bifrost.example.yaml) declares.
NEUTRAL_SERVED_MODEL_ID = "Qwen3.8-27B"


def lab_backend(
    backend_id: str,
    *,
    endpoint_url: str = NEUTRAL_CHAT_ENDPOINT,
    served_model_id: str = NEUTRAL_SERVED_MODEL_ID,
    parameter_count: str = "27B",
    context_window: int = 131072,
    max_tokens: int = 65536,
    timeout_ms: int = 300000,
) -> dict[str, Any]:
    """One local-rung binding in the ``bifrost_lane_overlay.v3`` shape."""
    return {
        "backend_id": backend_id,
        "endpoint_url": endpoint_url,
        "served_model_id": served_model_id,
        "parameter_count": parameter_count,
        "context_window": context_window,
        "max_tokens": max_tokens,
        "timeout_ms": timeout_ms,
    }


def lab_overlay(
    *, lane: str = "example-lab", backends: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """A ``locale: lab`` overlay binding both routed local rungs by default."""
    return {
        "schema_version": "bifrost_lane_overlay.v3",
        "lane": lane,
        "locale": "lab",
        "backends": (
            backends
            if backends is not None
            else [lab_backend("local-coder"), lab_backend("local-heavy-reasoning")]
        ),
    }


def cloud_overlay(*, lane: str = "example-cloud") -> dict[str, Any]:
    """A ``locale: cloud`` overlay that binds no backend."""
    return {
        "schema_version": "bifrost_lane_overlay.v3",
        "lane": lane,
        "locale": "cloud",
        "backends": [],
    }


def write_overlay(path: Path, overlay: dict[str, Any]) -> Path:
    """Write ``overlay`` as YAML to ``path`` and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(overlay, sort_keys=False), encoding="utf-8")
    return path
