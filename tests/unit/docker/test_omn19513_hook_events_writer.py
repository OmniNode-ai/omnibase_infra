# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rollback carrier bindings and lane isolation remain valid after the dev fold.

The base compose and catalog retain the opt-in writer for rollback. These checks
keep its node selection, offset identity, database and socket wiring covered,
and keep derived lanes inert while primary dev runs the node inside main.
"""

from __future__ import annotations

import re
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


@pytest.mark.parametrize("source", ["compose", "catalog"])
def test_writer_mounts_the_ingress_socket_directory_for_its_runtime_user(
    source: str,
) -> None:
    """Bind the mount to the kernel contract and the image's privilege drop."""
    if source == "compose":
        writer = _service("docker-compose.infra.yml", WRITER)
    else:
        resolver = CatalogResolver(catalog_dir=str(DOCKER / "catalog"))
        manifest = resolver._manifests[WRITER]
        services = generate_compose(
            ResolvedStack(
                manifests={WRITER: manifest}, required_env=set(), injected_env={}
            )
        )["services"]
        assert isinstance(services, dict)
        writer = services[WRITER]

    config = yaml.safe_load(
        (REPO_ROOT / "contracts/runtime/runtime_config.yaml").read_text()
    )
    ingress = config["local_ingress"]
    assert ingress["enabled"] is True
    assert "main" in ingress["enabled_profiles"]
    socket_dir = str(Path(ingress["socket_path"]).parent)

    dockerfile = (DOCKER / "Dockerfile.runtime").read_text()
    group = re.search(r"groupadd --gid (\d+) (\w+)", dockerfile)
    user = re.search(
        r"useradd --uid (\d+) --gid (\w+).*--create-home (\w+)", dockerfile
    )
    assert group is not None and user is not None
    assert user[2] == group[2]
    entrypoint = (DOCKER / "entrypoint-runtime.sh").read_text()
    assert f'exec gosu {user[3]} "$0" "$@"' in entrypoint
    assert "user" not in writer and not writer.get("entrypoint")

    mounts = [
        mount.partition(":")[2]
        for mount in writer["tmpfs"]
        if mount.partition(":")[0] == socket_dir
    ]
    assert len(mounts) == 1, f"expected one writable mount for {socket_dir}"
    options = dict(option.split("=", 1) for option in mounts[0].split(","))
    assert options["uid"] == user[1]
    assert options["gid"] == group[1]
    assert int(options["mode"], 8) == 0o770


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
