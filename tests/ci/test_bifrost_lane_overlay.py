# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI gates for the schema-v2, overlay-only Bifrost dev lane."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
OVERLAY_YAML = ROOT / "docker" / "lane-overlays" / "dev.bifrost.yaml"
PARITY_FIXTURE = ROOT / "tests" / "fixtures" / "bifrost_lane_overlay_v2.yaml"
COMPOSE_INFRA = ROOT / "docker" / "docker-compose.infra.yml"
OVERLAY_ENV = ROOT / "docker" / "lane-overlays" / "dev.bifrost.env"
RENDER_SCRIPT = ROOT / "scripts" / "render_bifrost_lane_overlay_env.py"
RECEIPT_MODE = ROOT / "src" / "omnibase_infra" / "cli" / "receipt_mode.py"
RUNTIME_HOST = ROOT / "src" / "omnibase_infra" / "runtime" / "runtime_host_process.py"
RENDERER = (
    ROOT
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "render_bifrost_delegation_contract.py"
)
LEGACY_CONFIG_LOADER = (
    ROOT
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_llm_inference_effect"
    / "handlers"
    / "bifrost"
    / "config_loader_bifrost.py"
)
_RUNTIME_SERVICES = frozenset({"omninode-runtime", "runtime-effects", "runtime-worker"})

pytestmark = pytest.mark.unit
sys.path.insert(0, str(ROOT / "src"))

from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)


def _load(path: Path) -> ModelBifrostLaneOverlay:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelBifrostLaneOverlay.model_validate(raw)


def test_dev_overlay_matches_cross_repo_v2_parity_fixture() -> None:
    """Infra's real binding and the shared parser fixture cannot drift."""
    overlay = _load(OVERLAY_YAML)
    fixture = _load(PARITY_FIXTURE)

    assert overlay.model_dump(mode="json") == fixture.model_dump(mode="json")

    # OMN-16833: the lane serves more than one local rung, so these are pinned
    # per-backend rather than as single-valued sets.  Live readback 2026-08-28:
    # .201:8000 -> "qwen3.8" (max_model_len 122880); .200:8101 -> "deepseek-v4-flash"
    # (context_length 131072).
    by_id = {binding.backend_key: binding for binding in overlay.backends}
    assert set(by_id) == {"local-coder", "local-heavy-reasoning", "local-ds-v4-flash"}

    for backend_key in ("local-coder", "local-heavy-reasoning"):
        binding = by_id[backend_key]
        assert binding.endpoint_url == "http://192.168.86.201:8000/v1/chat/completions"
        assert binding.advertised_model == "qwen3.8"
        assert binding.parameter_count == "27B"
        assert binding.context_window == 122_880

    ds_v4 = by_id["local-ds-v4-flash"]
    assert ds_v4.endpoint_url == "http://192.168.86.200:8101/v1/chat/completions"
    assert ds_v4.advertised_model == "deepseek-v4-flash"
    assert ds_v4.parameter_count == "284B"
    assert ds_v4.context_window == 131_072


def test_bifrost_lane_has_no_dotenv_sidecar_or_renderer() -> None:
    assert not OVERLAY_ENV.exists()
    assert not RENDER_SCRIPT.exists()
    assert not LEGACY_CONFIG_LOADER.exists()


def test_compose_mounts_typed_overlay_without_model_endpoint_env_wiring() -> None:
    text = COMPOSE_INFRA.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    assert "dev.bifrost.env" not in text
    assert "BIFROST_LOCAL_" not in text
    assert "BIFROST_SOURCE_CONTRACT_PATH" not in text

    services = data["services"]
    for service_name in _RUNTIME_SERVICES:
        service = services[service_name]
        assert "env_file" not in service
        assert (
            "./lane-overlays/dev.bifrost.yaml:/app/config/delegation/dev.bifrost.yaml:ro"
            in service["volumes"]
        )


def test_receipt_mode_has_no_legacy_home_dotenv_loader() -> None:
    """`onex delegate --bus inmemory` must not gain an env-file fallback."""
    receipt_source = RECEIPT_MODE.read_text(encoding="utf-8")
    runtime_source = RUNTIME_HOST.read_text(encoding="utf-8")

    assert "_load_omnibase_env_file" not in receipt_source
    assert "_load_omnibase_env_file" not in runtime_source
    assert "OMNIBASE_ENV_FILE" not in receipt_source


def test_renderer_has_no_legacy_source_or_reseed_compatibility() -> None:
    renderer_source = RENDERER.read_text(encoding="utf-8")

    assert (
        "/app/src/omnibase_infra/configs/bifrost_delegation.yaml" not in renderer_source
    )
    assert "_LEGACY_SOURCE_PATH" not in renderer_source
    assert "reseed" not in renderer_source


# ---------------------------------------------------------------------------
# OMN-17150: lane-resolved overlay pins — no lane may fall through to another
# lane's overlay. The renderer resolves BIFROST_LANE_OVERLAY_PATH per lane and
# fails fast without it; these gates keep every committed lane recipe honest.
# ---------------------------------------------------------------------------

LANE_OVERLAYS_DIR = ROOT / "docker" / "lane-overlays"
COMPOSE_STABILITY = ROOT / "docker" / "docker-compose.stability-test.yml"
_DEV_OVERLAY_PIN = "/app/config/delegation/dev.bifrost.yaml"
#: Standalone single-lane compose files that render the Bifrost contract
#: (their runtime env sets a non-empty BIFROST_CONTRACT_PATH). Each must pin
#: and mount ITS OWN overlay. docker-compose.prod.yml and dev-lane.yml layer
#: docker-compose.infra.yml and inherit the dev pin + mount from its anchor.
_STANDALONE_RENDERING_LANE_FILES = {
    "judge": ROOT / "docker" / "docker-compose.judge.yml",
    "lakshman": ROOT / "docker" / "docker-compose.lakshman.yml",
}


def test_renderer_has_no_default_lane_overlay_path() -> None:
    """The dev-named default was the OMN-17150 defect — it must not return."""
    renderer_source = RENDERER.read_text(encoding="utf-8")

    assert "_DEFAULT_LANE_OVERLAY_PATH" not in renderer_source
    assert "dev.bifrost.yaml" not in renderer_source
    assert "BIFROST_LANE_OVERLAY_PATH" in renderer_source


def test_every_lane_overlay_is_typed_and_named_for_its_lane() -> None:
    """Each overlay file parses against the v2 schema and its ``lane`` field
    matches its filename, so a lane can never mount a file that attributes its
    rendered contract to a different lane."""
    overlays = sorted(LANE_OVERLAYS_DIR.glob("*.bifrost.yaml"))
    assert {path.name for path in overlays} >= {
        "dev.bifrost.yaml",
        "judge.bifrost.yaml",
        "lakshman.bifrost.yaml",
    }
    for path in overlays:
        overlay = _load(path)
        expected_lane = path.name.removesuffix(".bifrost.yaml")
        assert overlay.lane == expected_lane, (
            f"{path.name} declares lane {overlay.lane!r}; the filename says "
            f"{expected_lane!r} — a mis-attributed overlay is exactly the "
            "cross-lane confusion OMN-17150 removed"
        )


def test_dev_runtime_anchor_pins_the_overlay_it_mounts() -> None:
    data = yaml.safe_load(COMPOSE_INFRA.read_text(encoding="utf-8"))
    anchor = data["x-runtime-env"]
    assert anchor["BIFROST_LANE_OVERLAY_PATH"] == _DEV_OVERLAY_PIN


def test_stability_lane_mounts_the_dev_overlay_at_the_pinned_path() -> None:
    """stability-test layers infra.yml, inheriting the dev pin — sharing the
    dev binding is a deliberate, legible decision, so the mount must provide
    the file at exactly the pinned path for all three runtime services."""
    text = COMPOSE_STABILITY.read_text(encoding="utf-8")
    mount = f"./lane-overlays/dev.bifrost.yaml:{_DEV_OVERLAY_PIN}:ro"
    assert text.count(mount) == 3


def test_standalone_lane_files_pin_and_mount_their_own_overlay() -> None:
    """A standalone lane that renders must pin and mount ITS lane's overlay —
    never reach for another lane's file (the OMN-17150 class)."""
    for lane, path in _STANDALONE_RENDERING_LANE_FILES.items():
        text = path.read_text(encoding="utf-8")
        container_path = f"/app/config/delegation/{lane}.bifrost.yaml"

        assert f"BIFROST_LANE_OVERLAY_PATH: {container_path}" in text, (
            f"{path.name} must pin {container_path} in its runtime env anchor"
        )
        mount = f"./lane-overlays/{lane}.bifrost.yaml:{container_path}:ro"
        assert text.count(mount) == 2, (
            f"{path.name} must mount the {lane} overlay on both runtime "
            "services (main + effects)"
        )
        assert "dev.bifrost.yaml" not in text, (
            f"{path.name} references the dev lane's overlay — a standalone "
            "lane must carry its own"
        )
        assert (LANE_OVERLAYS_DIR / f"{lane}.bifrost.yaml").is_file()
