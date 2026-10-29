# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI gates for typed Bifrost overlays and private compose mounts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
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
from tests.helpers.lane_overlay_fixtures import (
    LANE_OVERLAY_MOUNT,
    LANE_OVERLAY_TARGET,
    lane_overlay_mounts,
    load_compose,
)


def _load(path: Path) -> ModelBifrostLaneOverlay:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelBifrostLaneOverlay.model_validate(raw)


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
        assert service["environment"]["BIFROST_LANE_OVERLAY_PATH"] == (
            LANE_OVERLAY_TARGET
        )
    assert lane_overlay_mounts(COMPOSE_INFRA) == {
        service: [LANE_OVERLAY_MOUNT]
        for service in _RUNTIME_SERVICES
        | {"tenant-projection-writer", "projection-claude-hook-events-writer"}
    }


def test_every_service_that_renders_bifrost_mounts_the_overlay_it_pins() -> None:
    """A kernel service that pins the overlay path must also mount the file.

    OMN-21433: projection-claude-hook-events-writer inherited the pin from
    x-runtime-env but declared no mount, so its entrypoint preflight raised
    "Bifrost lane overlay not found" on every boot. The pin and the mount are
    one decision; this reads the pin from the rendered environment so a new
    kernel service cannot carry one without the other.
    """
    services = yaml.safe_load(COMPOSE_INFRA.read_text(encoding="utf-8"))["services"]
    pinned = {
        name
        for name, service in services.items()
        if (service.get("environment") or {}).get("BIFROST_LANE_OVERLAY_PATH")
        == LANE_OVERLAY_TARGET
    }

    # Positive control: the services the pin is known to reach are found.
    assert pinned >= _RUNTIME_SERVICES
    assert "projection-claude-hook-events-writer" in pinned
    assert pinned == set(lane_overlay_mounts(COMPOSE_INFRA))


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
#: Standalone recipes render the Bifrost contract and bind their project's
#: overlay at the shared target. Layered recipes inherit the base environment.
_STANDALONE_RENDERING_LANE_FILES = {
    "judge": ROOT / "docker" / "docker-compose.judge.yml",
    "lakshman": ROOT / "docker" / "docker-compose.lakshman.yml",
    # OMN-19339: an overlay on the dogfood file that replaces the runtime
    # services' volumes wholesale, so it pins and mounts its own file.
    "sim-202": ROOT / "docker" / "docker-compose.sim-202.yml",
}


def test_renderer_has_no_default_lane_overlay_path() -> None:
    """The dev-named default was the OMN-17150 defect — it must not return."""
    renderer_source = RENDERER.read_text(encoding="utf-8")

    assert "_DEFAULT_LANE_OVERLAY_PATH" not in renderer_source
    assert "dev.bifrost.yaml" not in renderer_source
    assert "BIFROST_LANE_OVERLAY_PATH" in renderer_source


def test_shipped_overlay_directory_contains_only_a_typed_local_template() -> None:
    """Deployment overlays are private; the shipped template names its local lane."""
    assert sorted(LANE_OVERLAYS_DIR.glob("*.bifrost.yaml")) == []
    overlay = _load(LANE_OVERLAYS_DIR / "local.bifrost.example.yaml")
    assert overlay.lane == "local"


def test_dev_runtime_anchor_pins_the_overlay_it_mounts() -> None:
    data = yaml.safe_load(COMPOSE_INFRA.read_text(encoding="utf-8"))
    anchor = data["x-runtime-env"]
    assert anchor["BIFROST_LANE_OVERLAY_PATH"] == LANE_OVERLAY_TARGET


def test_stability_lane_mounts_the_overlay_at_the_pinned_path() -> None:
    """All four runtime services, including the carrier, bind the shared path."""
    runtime_services = _RUNTIME_SERVICES | {"tenant-projection-writer"}
    assert lane_overlay_mounts(COMPOSE_STABILITY) == {
        service: [LANE_OVERLAY_MOUNT] for service in runtime_services
    }
    base_environment = load_compose(COMPOSE_INFRA)["x-runtime-env"]
    services = load_compose(COMPOSE_STABILITY)["services"]
    for service in runtime_services:
        environment = {**base_environment, **services[service]["environment"]}
        assert environment["BIFROST_LANE_OVERLAY_PATH"] == LANE_OVERLAY_TARGET


def test_standalone_lane_files_pin_and_mount_their_own_overlay() -> None:
    """Each standalone recipe binds its project's overlay at the shared path.

    Both runtimes pin the path, and every service that names a non-empty
    BIFROST_LANE_OVERLAY_PATH mounts it. The collaborator lane's delegation
    writer merges that lane's runtime env and renders from the path too
    (OMN-17427); a service that blanks it, such as projection-api, mounts nothing.
    """
    runtime_services = {"omninode-runtime", "runtime-effects"}
    for path in _STANDALONE_RENDERING_LANE_FILES.values():
        services = load_compose(path)["services"]
        carriers = {
            name
            for name, service in services.items()
            if ((service or {}).get("environment") or {}).get(
                "BIFROST_LANE_OVERLAY_PATH"
            )
        }
        assert runtime_services <= carriers
        assert lane_overlay_mounts(path) == {
            service: [LANE_OVERLAY_MOUNT] for service in carriers
        }
        for service in carriers:
            assert services[service]["environment"]["BIFROST_LANE_OVERLAY_PATH"] == (
                LANE_OVERLAY_TARGET
            )


def test_local_template_declares_an_explicit_locale() -> None:
    """The template states its locale instead of relying on a model default."""
    raw = yaml.safe_load(
        (LANE_OVERLAYS_DIR / "local.bifrost.example.yaml").read_text(encoding="utf-8")
    )
    assert "locale" in raw
    assert raw["locale"] == "lab"
