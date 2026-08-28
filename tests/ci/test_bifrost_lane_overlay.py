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
