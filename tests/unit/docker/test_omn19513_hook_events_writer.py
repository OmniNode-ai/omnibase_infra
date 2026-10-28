# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19513: the claude hook-events projection has its own dev-lane writer.

``node_projection_claude_hook_events`` was auto-wired inside ``omninode-runtime``,
whose boot joins about 560 consumer groups one at a time. Each restart left the
projection unsubscribed for about 19 minutes (measured 2026-10-28). The fix is a
dedicated container that boots over a manifest of that one contract, plus an
exclusion on the main runtime so the node has exactly one consumer-group owner.

The consumer group id carries the process's ``KAFKA_INSTANCE_ID``
(``...consume.1.0.0.__i.runtime-main.__t.<topic>``), so the writer must share the
main runtime's instance id for the committed offsets to carry over.

These tests render YAML and start no container.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver, ResolvedStack
from omnibase_infra.topology import validate_docker_catalog_parity

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER = REPO_ROOT / "docker"

NODE = "projection_claude_hook_events"
WRITER = "projection-claude-hook-events-writer"
MAIN = "omninode-runtime"
INERT_PROFILE = "profile-carrier-optin"


def _load(name: str) -> dict[str, Any]:
    text = (DOCKER / name).read_text(encoding="utf-8")
    loaded = yaml.safe_load(text.replace("!override", ""))
    assert isinstance(loaded, dict)
    return loaded


def _service(compose: str, service: str) -> dict[str, Any]:
    services = _load(compose)["services"]
    assert service in services, f"{compose} declares no {service}"
    return dict(services[service])


def _env(compose: str, service: str) -> dict[str, str]:
    return {
        str(key): str(value)
        for key, value in _service(compose, service).get("environment", {}).items()
    }


def test_base_declares_the_writer_inert() -> None:
    writer = _service("docker-compose.infra.yml", WRITER)

    assert writer["profiles"] == [INERT_PROFILE]
    assert writer["container_name"] == "omnimarket-projection-claude-hook-events-writer"


def test_writer_loads_exactly_the_one_node() -> None:
    env = _env("docker-compose.infra.yml", WRITER)

    assert env["ONEX_RUNTIME_NODE_SELECTION"] == NODE
    assert env["RUNTIME_PROFILE"] == "main"


def test_writer_shares_the_main_runtime_instance_id_so_offsets_carry_over() -> None:
    writer = _env("docker-compose.infra.yml", WRITER)
    main = _env("docker-compose.infra.yml", MAIN)

    assert writer["KAFKA_INSTANCE_ID"] == main["KAFKA_INSTANCE_ID"] == "runtime-main"


def test_writer_keeps_its_own_kernel_group_and_publishes_no_introspection() -> None:
    writer = _env("docker-compose.infra.yml", WRITER)
    main = _env("docker-compose.infra.yml", MAIN)

    assert writer["ONEX_GROUP_ID"] != main["ONEX_GROUP_ID"]
    assert writer["OMNIINTELLIGENCE_PUBLISH_INTROSPECTION"] == "false"


def test_dev_lane_starts_the_writer_and_main_excludes_the_same_node() -> None:
    writer = _service("docker-compose.dev-lane.yml", WRITER)
    main_env = _env("docker-compose.dev-lane.yml", MAIN)

    assert writer["profiles"] == ["runtime", "full"]
    assert main_env["ONEX_RUNTIME_NODE_EXCLUSION"] == NODE
    assert (
        main_env["ONEX_RUNTIME_NODE_EXCLUSION"]
        == _env("docker-compose.infra.yml", WRITER)["ONEX_RUNTIME_NODE_SELECTION"]
    )


def test_dev_lane_writer_names_the_lane_the_node_is_scoped_to() -> None:
    env = _env("docker-compose.dev-lane.yml", WRITER)
    main_env = _env("docker-compose.dev-lane.yml", MAIN)

    assert env["ONEX_RUNTIME_LANE"] == main_env["ONEX_RUNTIME_LANE"]
    # Placement identity: the lane's one health speaker stays omninode-runtime.
    assert env["ONEX_RUNTIME_LANE_HEALTH_SPEAKER"] == "false"
    assert "ONEX_RUNTIME_LANE_HEALTH_SPEAKER" not in main_env
    assert env["ONEX_WIRING_STRICT_MODE"] == "1"


def test_catalog_writer_preserves_dev_lane_placement_and_broker_policy() -> None:
    resolver = CatalogResolver(catalog_dir=str(DOCKER / "catalog"))
    manifest = resolver._manifests[WRITER]
    services = generate_compose(
        ResolvedStack(manifests={WRITER: manifest}, required_env=set(), injected_env={})
    )["services"]
    assert isinstance(services, dict)
    rendered = services[WRITER]
    env = rendered["environment"]
    overlay_env = _env("docker-compose.dev-lane.yml", WRITER)

    for key in (
        "ONEX_RUNTIME_LANE_HEALTH_SPEAKER",
        "ONEX_WIRING_STRICT_MODE",
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_MAX_POLL_INTERVAL_MS",
    ):
        assert env.get(key) == overlay_env[key], key
    # Public catalogs require the deployment's registered lane as input.
    assert "ONEX_RUNTIME_LANE" in manifest.required_env
    assert env["ONEX_RUNTIME_LANE"].startswith("${ONEX_RUNTIME_LANE:?")
    for key in ("KAFKA_SASL_USERNAME", "KAFKA_SASL_PASSWORD"):
        # Bind the same operator secret; required-var messages may differ.
        assert env[key].split(":?", 1)[0] == overlay_env[key].split(":?", 1)[0]
        assert env[key].startswith("${DEV_KAFKA_SASL_")
    assert "${ROLE_OMNIDASH_PASSWORD:?" in env["OMNIDASH_ANALYTICS_DB_URL"]
    assert "OMNINODE_INTERNAL_DB_URL" in manifest.required_env
    assert env["OMNINODE_INTERNAL_DB_URL"].startswith("${OMNINODE_INTERNAL_DB_URL:?")
    assert rendered["ulimits"]["nofile"] == {"soft": 65535, "hard": 65535}


def test_catalog_writer_declares_its_application_database_bindings() -> None:
    catalog = yaml.safe_load((DOCKER / "catalog/database-consumers.yaml").read_text())
    assert set(catalog["consumers"][WRITER]["bindings"]) == {
        "omninode_runtime_service",
        "app_dashboard",
    }
    validate_docker_catalog_parity()


def _derived_dev_overlays() -> list[str]:
    """The overlays layered on the dev lane for other hosts and the pre-PR slots."""
    names = sorted(path.name for path in DOCKER.glob("docker-compose.dev-[0-9]*.yml"))
    return [*names, "docker-compose.prepr.yml"]


def _other_overlays() -> list[str]:
    """Every compose file that is neither the base, the dev lane nor derived from it."""
    skip = {
        "docker-compose.infra.yml",
        "docker-compose.dev-lane.yml",
        "docker-compose.dev-tenant-fold.yml",
        *_derived_dev_overlays(),
    }
    return sorted(
        path.name
        for path in DOCKER.glob("docker-compose*.yml")
        if path.name not in skip
    )


def test_the_overlay_sets_are_not_empty() -> None:
    """Positive control for the two parametrised tests below."""
    assert len(_derived_dev_overlays()) >= 2
    assert len(_other_overlays()) >= 2


@pytest.mark.parametrize("overlay", _derived_dev_overlays())
def test_derived_dev_lanes_keep_the_writer_inert(overlay: str) -> None:
    """Their lanes are outside the node's `runtime_lanes` scope: a writer there wires zero."""
    assert _service(overlay, WRITER)["profiles"] == [INERT_PROFILE]


@pytest.mark.parametrize("overlay", _other_overlays())
def test_other_lanes_neither_start_the_writer_nor_exclude_the_node(
    overlay: str,
) -> None:
    services = _load(overlay).get("services") or {}

    assert "profiles" not in dict(services.get(WRITER) or {})
    for service in services.values():
        env = (service or {}).get("environment") or {}
        if isinstance(env, dict):
            assert "ONEX_RUNTIME_NODE_EXCLUSION" not in env


def test_the_writer_is_in_the_dev_lane_restart_scope() -> None:
    refresh = (REPO_ROOT / "scripts/runtime_build/refresh_dev_lane.sh").read_text(
        encoding="utf-8"
    )
    deploy = (
        REPO_ROOT / "src/omnibase_infra/handlers/handler_runtime_deploy.sh"
    ).read_text(encoding="utf-8")
    agent = (REPO_ROOT / "scripts/deploy-agent/deploy_agent/events.py").read_text(
        encoding="utf-8"
    )

    assert WRITER in refresh
    assert WRITER in deploy
    assert f'"{WRITER}"' in agent
