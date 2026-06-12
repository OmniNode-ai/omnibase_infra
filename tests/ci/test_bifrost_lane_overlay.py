# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI gate: Bifrost lane overlay env sidecar is in sync with the YAML source.

OMN-12864: docker/lane-overlays/dev.bifrost.env must be the rendered form of
docker/lane-overlays/dev.bifrost.yaml. This test fails when the sidecar is
stale, preventing a deploy that would silently use wrong endpoint bindings.

Re-render with:
    uv run python scripts/render_bifrost_lane_overlay_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_YAML = ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
OVERLAY_ENV = ROOT / "docker" / "lane-overlays" / "dev.bifrost.env"
RENDER_SCRIPT = ROOT / "scripts" / "render_bifrost_lane_overlay_env.py"

pytestmark = pytest.mark.unit

# Insert src/ into path so imports work without an installed package.
sys.path.insert(0, str(ROOT / "src"))

from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)


def _load_overlay() -> ModelBifrostLaneOverlay:
    raw = yaml.safe_load(OVERLAY_YAML.read_text(encoding="utf-8"))
    return ModelBifrostLaneOverlay.model_validate(raw)


def _parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def test_overlay_yaml_loads_and_validates() -> None:
    """The dev.bifrost.yaml is valid and parses without errors."""
    overlay = _load_overlay()
    assert overlay.lane == "dev"
    assert overlay.coder_endpoint_url.endswith("/v1/chat/completions")
    assert overlay.reasoner_endpoint_url.endswith("/v1/chat/completions")
    assert overlay.embedding_endpoint_url.endswith("/v1/chat/completions")
    assert overlay.ds4_flash_endpoint_url.endswith("/v1/chat/completions")


def test_overlay_env_sidecar_exists() -> None:
    """The rendered dev.bifrost.env sidecar must be committed alongside the YAML."""
    assert OVERLAY_ENV.exists(), (
        f"Missing rendered sidecar: {OVERLAY_ENV}\n"
        "Re-render with: uv run python scripts/render_bifrost_lane_overlay_env.py"
    )


def test_overlay_env_sidecar_in_sync_with_yaml() -> None:
    """The dev.bifrost.env sidecar must match what the YAML would render.

    Fails when the YAML source was edited without re-rendering the sidecar.
    Fix: uv run python scripts/render_bifrost_lane_overlay_env.py
    """
    overlay = _load_overlay()
    expected_env = overlay.as_env_dict()

    actual_env = _parse_env_file(OVERLAY_ENV)

    for key, expected_value in expected_env.items():
        assert key in actual_env, (
            f"Missing key {key!r} in {OVERLAY_ENV}. "
            "Re-render: uv run python scripts/render_bifrost_lane_overlay_env.py"
        )
        assert actual_env[key] == expected_value, (
            f"{key}: env sidecar has {actual_env[key]!r}, YAML source wants {expected_value!r}. "
            "Re-render: uv run python scripts/render_bifrost_lane_overlay_env.py"
        )


def test_overlay_env_has_all_four_bifrost_keys() -> None:
    """The sidecar must declare all four BIFROST_LOCAL_* keys."""
    env = _parse_env_file(OVERLAY_ENV)
    required_keys = {
        "BIFROST_LOCAL_CODER_ENDPOINT_URL",
        "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
        "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
        "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
    }
    missing = required_keys - set(env.keys())
    assert not missing, (
        f"Missing keys in {OVERLAY_ENV}: {sorted(missing)}. "
        "Re-render: uv run python scripts/render_bifrost_lane_overlay_env.py"
    )


def test_render_script_exists() -> None:
    """The render script must exist so operators can regenerate the sidecar."""
    assert RENDER_SCRIPT.exists(), f"Missing render script: {RENDER_SCRIPT}"


COMPOSE_INFRA = ROOT / "docker" / "docker-compose.infra.yml"
_RUNTIME_SERVICES_NEEDING_BIFROST = frozenset(
    {"omninode-runtime", "runtime-effects", "runtime-worker"}
)


def test_compose_env_file_is_service_level_not_top_level() -> None:
    """OMN-12864: dev.bifrost.env must be wired as a service-level env_file, not top-level.

    Docker Compose v2 schema rejects 'env_file' as an additional property at the
    top level (outside of 'services'). The committed sidecar must be wired
    per-service so compose config renders cleanly in CI.
    """
    import yaml

    text = COMPOSE_INFRA.read_text(encoding="utf-8")
    data = yaml.safe_load(text)

    # Top-level env_file is schema-invalid in compose v2 — must be absent.
    assert "env_file" not in data, (
        "docker-compose.infra.yml has a top-level 'env_file' key which is schema-invalid "
        "in Docker Compose v2. Move it to service-level env_file blocks."
    )


def test_runtime_services_have_bifrost_env_file() -> None:
    """OMN-12864: Runtime services that render Bifrost must declare the committed env_file.

    This ensures BIFROST_LOCAL_* vars flow into the three runtime containers from
    the committed overlay, not from ephemeral shell exports.
    """
    import yaml

    data = yaml.safe_load(COMPOSE_INFRA.read_text(encoding="utf-8"))
    services = data.get("services", {})

    for svc in _RUNTIME_SERVICES_NEEDING_BIFROST:
        assert svc in services, f"Service {svc!r} not found in compose file"
        svc_cfg = services[svc]
        env_files = svc_cfg.get("env_file", [])
        assert isinstance(env_files, list) and env_files, (
            f"Service {svc!r} has no env_file block — "
            "BIFROST_LOCAL_* overlay won't reach this container"
        )
        paths = [(e["path"] if isinstance(e, dict) else e) for e in env_files]
        assert any("dev.bifrost.env" in p for p in paths), (
            f"Service {svc!r} env_file does not include dev.bifrost.env. Found: {paths}"
        )
