# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22756: a lane overlay read from a YAML file on disk resolves a media capability.

The overlay is written to a file as an operator would author it, read back and
validated through the overlay model, then a node asks for each typed capability
and receives the serving binding and URL without naming a host or port.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.enums.enum_bifrost_lane_media_capability import (
    EnumBifrostLaneMediaCapability,
)
from omnibase_infra.models.delegation.model_bifrost_lane_media_capability_error import (
    BifrostLaneMediaCapabilityNotServingError,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.integration

# RFC 5737 documentation-range address, never a lab host.
_HOST = "http://192.0.2.30"
_PATHS = {
    "tts": "/v1/audio/speech",
    "stt": "/v1/audio/transcriptions",
    "embed": "/v1/embeddings",
    "decide": "/v1/chat/completions",
    "video": "/v1/video/generations",
}


def _backend(capability: str, port: int) -> dict[str, object]:
    return {
        "backend_id": f"media-{capability}",
        "endpoint_url": f"{_HOST}:{port}{_PATHS[capability]}",
        "served_model_id": f"{capability}-model",
        "parameter_count": "1B",
        "context_window": 8192,
        "max_tokens": 1024,
        "timeout_ms": 60_000,
        "provider": "local",
        "tier": "local",
        "credential": {"kind": "none"},
        "media_capability": capability,
    }


def _write_overlay(path: Path, capabilities: list[str]) -> None:
    document = {
        "schema_version": "bifrost_lane_overlay.v3",
        "lane": "media-test",
        "locale": "lab",
        "backends": [_backend(c, 9001 + i) for i, c in enumerate(capabilities)],
    }
    path.write_text(yaml.safe_dump(document), encoding="utf-8")


def _load(path: Path) -> ModelBifrostLaneOverlay:
    return ModelBifrostLaneOverlay.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def test_overlay_file_resolves_every_capability(tmp_path: Path) -> None:
    capabilities = [c.value for c in EnumBifrostLaneMediaCapability]
    overlay_file = tmp_path / "overlay.yaml"
    _write_overlay(overlay_file, capabilities)

    overlay = _load(overlay_file)

    for capability in capabilities:
        binding = overlay.resolve_media_capability(capability)
        assert binding.backend_key == f"media-{capability}"
        assert binding.endpoint_url.endswith(_PATHS[capability])


def test_overlay_file_without_capability_names_it_not_serving(
    tmp_path: Path,
) -> None:
    overlay_file = tmp_path / "overlay.yaml"
    _write_overlay(overlay_file, ["tts"])

    overlay = _load(overlay_file)

    with pytest.raises(BifrostLaneMediaCapabilityNotServingError):
        overlay.resolve_media_capability("stt")
