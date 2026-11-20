# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev runtime's overlay store mount holds the overlay store (OMN-24626).

Review finding HIGH 2 of omnibase_infra#6027: docker-compose.dev-lane.yml defaults the
store mount to ``../../overlays/overlays`` relative to its own ``docker/`` directory,
which was the registry root's overlays clone while the agent built from the canonical
clone, and is an empty path beside the agent's own clone. The agent now declares the
variable in the compose environment it renders.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml
from deploy_agent import executor as executor_mod

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
COMPOSE_FILES = ("docker-compose.dev-lane.yml", "docker-compose.dev-tenant-fold.yml")
STORE_TARGET = "/etc/onex/skill-overlays-store"
VOLUME = re.compile(r"^\$\{(?P<var>\w+)(?::-(?P<default>[^}]*))?\}:(?P<target>[^:]+)")


COMPOSE_MERGE_TAG = re.compile(r"!(?:override|reset)\b")


def _store_mount_sources(
    compose: Path, env: dict[str, str], project_dir: Path
) -> list[Path]:
    """Each store mount's host path as compose resolves it: the variable, else its default."""
    services = yaml.safe_load(COMPOSE_MERGE_TAG.sub("", compose.read_text("utf-8")))[
        "services"
    ]
    sources: list[Path] = []
    for service in services.values():
        for volume in service.get("volumes", []):
            match = VOLUME.match(volume) if isinstance(volume, str) else None
            if match is None or match["target"] != STORE_TARGET:
                continue
            raw = env.get(match["var"]) or match["default"] or ""
            sources.append(Path(os.path.normpath(project_dir / raw)))
    return sources


@pytest.fixture
def agent_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    registry = tmp_path / "registry"
    own_clone = tmp_path / "state" / "jobs-dev" / "deploy-source" / "omnibase_infra"
    (registry / "omnibase_infra").mkdir(parents=True)
    own_clone.mkdir(parents=True)
    for name, value in {
        "DEPLOY_AGENT_REPO_DIR": str(own_clone),
        "DEPLOY_AGENT_CANONICAL_CLONE": str(registry / "omnibase_infra"),
        "DEPLOY_AGENT_LAB_OVERLAY_SOURCE": str(registry / "overlay-source"),
        "DEPLOY_AGENT_TRACKING_REF": "dev",
        "DEPLOY_AGENT_ALLOWED_LANES": "dev",
        "ONEX_LANE_OVERLAY_DIR": str(tmp_path),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("DEV_LANE_OVERLAY_STORE_ROOTS_DIR", raising=False)
    executor_mod.deploy_source.cache_clear()
    monkeypatch.setattr(executor_mod, "operator_store_overlay", dict)
    monkeypatch.setattr(executor_mod, "_load_runtime_policy_env", dict)
    monkeypatch.setattr(executor_mod, "_lane_env_overlay", dict)
    monkeypatch.setattr(executor_mod, "_dev_broker_policy_env", lambda env: {})
    monkeypatch.setattr(
        executor_mod, "_llm_endpoint_env_overlay", lambda env: ({}, None)
    )
    yield {"registry": registry, "own_clone": own_clone}
    executor_mod.deploy_source.cache_clear()


@pytest.mark.parametrize("compose_name", COMPOSE_FILES)
def test_overlay_mount_source_is_the_populated_overlay_store(
    agent_world: dict[str, Path], compose_name: str
) -> None:
    store = agent_world["registry"] / "overlays" / "overlays"
    store.mkdir(parents=True)
    (store / "overlay.yaml").write_text("schema_version: '1'\n", "utf-8")

    env = executor_mod._compose_env()
    sources = _store_mount_sources(
        REPO_ROOT / "docker" / compose_name, env, agent_world["own_clone"] / "docker"
    )

    assert sources, f"{compose_name} mounts no overlay store"
    for source in sources:
        assert source == store
        assert any(source.iterdir()), f"{source} is empty"


def test_overlay_mount_default_beside_the_own_clone_is_the_empty_path_the_review_found(
    agent_world: dict[str, Path],
) -> None:
    """Control: with no declared variable the compose default lands beside the own clone."""
    sources = _store_mount_sources(
        REPO_ROOT / "docker" / COMPOSE_FILES[0], {}, agent_world["own_clone"] / "docker"
    )
    assert sources
    assert all(not source.exists() for source in sources)


def test_overlay_mount_an_absent_or_empty_store_is_refused(
    agent_world: dict[str, Path],
) -> None:
    handler = executor_mod.load_dev_boot_guard("handler_deploy_source_clone")
    with pytest.raises(handler.DeploySourceCloneError, match="absent or empty"):
        executor_mod._compose_env()
    (agent_world["registry"] / "overlays" / "overlays").mkdir(parents=True)
    with pytest.raises(handler.DeploySourceCloneError, match="absent or empty"):
        executor_mod._compose_env()


def test_overlay_mount_an_operator_declared_store_root_is_kept(
    agent_world: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rendered = tmp_path / "rendered"
    rendered.mkdir()
    monkeypatch.setenv("DEV_LANE_OVERLAY_STORE_ROOTS_DIR", str(rendered))
    assert executor_mod._compose_env()["DEV_LANE_OVERLAY_STORE_ROOTS_DIR"] == str(
        rendered
    )
