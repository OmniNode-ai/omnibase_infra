# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21279: dev-lane Bifrost kernels and market commands need the composed image."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_DOCKER = _REPO / "docker"
_COMPOSE_TAG = re.compile(r"(?m)(:\s+|-\s+)!(?:override|reset)\b[ \t]*")
_BASE_CONTRACT_ENV = re.compile(r"(?m)^\s*ENV\s+BIFROST_BASE_CONTRACT_PATH(?:=|\s)")


def _load(path: Path) -> dict[str, Any]:
    """Load a compose file, reading its merge tags as plain data."""
    loaded = yaml.safe_load(_COMPOSE_TAG.sub(r"\1", path.read_text(encoding="utf-8")))
    return loaded if isinstance(loaded, dict) else {}


def load_compose(path: Path) -> dict[str, dict[str, Any]]:
    """The ``services`` mapping of one compose file."""
    return dict(_load(path).get("services") or {})


def _env(service: Mapping[str, Any]) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return dict(str(e).split("=", 1) for e in env if "=" in str(e))
    return {str(k): str(v) for k, v in env.items()}


def dev_stack_services() -> dict[str, dict[str, Any]]:
    """The dev stack: the base infra services with the dev-lane overlay merged on."""
    merged = load_compose(_DOCKER / "docker-compose.infra.yml")
    for name, overlay in load_compose(_DOCKER / "docker-compose.dev-lane.yml").items():
        base = dict(merged.get(name) or {})
        env = {**_env(base), **_env(overlay)}
        base.update(overlay)
        if env:
            base["environment"] = env
        merged[name] = base
    return merged


def find_services_without_market_image(
    services: Mapping[str, Mapping[str, Any]], infra_dockerfile_text: str
) -> dict[str, str]:
    """Return infra-only services that require market code or its base contract."""
    refused: dict[str, str] = {}
    for name, service in services.items():
        build = service.get("build")
        if not isinstance(build, Mapping):
            continue
        if not str(build.get("dockerfile", "")).endswith("docker/Dockerfile.runtime"):
            continue
        if service.get("scale") == 0:
            continue
        env = _env(service)
        if (
            env.get("BIFROST_CONTRACT_PATH", "").strip()
            and not env.get("BIFROST_BASE_CONTRACT_PATH", "").strip()
            and not _BASE_CONTRACT_ENV.search(infra_dockerfile_text)
        ):
            refused[name] = "renders Bifrost without BIFROST_BASE_CONTRACT_PATH"
        command = service.get("command") or []
        command_text = (
            command if isinstance(command, str) else " ".join(map(str, command))
        )
        if "omnimarket" in command_text:
            refused[name] = "runs an omnimarket command without the composed image"
    return refused


def test_fixture_infra_only_kernels_and_writers_are_refused() -> None:
    services = {
        "kernel": {
            "build": {"context": "..", "dockerfile": "docker/Dockerfile.runtime"},
            "environment": {"BIFROST_CONTRACT_PATH": "/app/data/bifrost.yaml"},
        },
        "writer": {
            "build": {"context": "..", "dockerfile": "docker/Dockerfile.runtime"},
            "command": ["python", "-m", "omnimarket.nodes.x"],
        },
    }
    assert set(find_services_without_market_image(services, "")) == {"kernel", "writer"}


def test_fixture_composed_services_and_inert_infra_builds_are_clean() -> None:
    composed = {
        "context": "../workspace/sibling-repos/omnimarket",
        "dockerfile": "Dockerfile.runtime",
    }
    infra = {"context": "..", "dockerfile": "docker/Dockerfile.runtime"}
    services = {
        "kernel": {
            "build": composed,
            "environment": {"BIFROST_CONTRACT_PATH": "/app/data/bifrost.yaml"},
        },
        "writer": {
            "build": composed,
            "command": ["python", "-m", "omnimarket.nodes.x"],
        },
        "build-only": {
            "build": infra,
            "scale": 0,
            "environment": {"BIFROST_CONTRACT_PATH": "/app/data/bifrost.yaml"},
            "command": "omnimarket-projection-api",
        },
        "render-disabled": {
            "build": infra,
            "environment": {"BIFROST_CONTRACT_PATH": ""},
        },
    }
    assert find_services_without_market_image(services, "") == {}


@pytest.mark.parametrize(
    "command", [["python", "-m", "omnimarket.nodes.x"], "omnimarket-projection-api"]
)
def test_fixture_market_commands_require_composition(command: list[str] | str) -> None:
    services = {
        "market": {
            "build": {"dockerfile": "docker/Dockerfile.runtime"},
            "command": command,
        }
    }
    assert "market" in find_services_without_market_image(services, "")


@pytest.mark.parametrize("base_from_image", [False, True])
def test_fixture_explicit_base_contract_allows_infra_kernel(
    base_from_image: bool,
) -> None:
    env = {"BIFROST_CONTRACT_PATH": "/app/data/bifrost.yaml"}
    dockerfile = ""
    if base_from_image:
        dockerfile = "ENV BIFROST_BASE_CONTRACT_PATH=/app/base.yaml\n"
    else:
        env["BIFROST_BASE_CONTRACT_PATH"] = "/app/base.yaml"
    services = {
        "kernel": {
            "build": {"dockerfile": "docker/Dockerfile.runtime"},
            "environment": env,
        }
    }
    assert find_services_without_market_image(services, dockerfile) == {}


def test_real_dev_stack_has_no_market_workloads_on_the_infra_only_image() -> None:
    assert (
        find_services_without_market_image(
            dev_stack_services(),
            (_DOCKER / "Dockerfile.runtime").read_text(encoding="utf-8"),
        )
        == {}
    )


def test_infra_image_does_not_supply_the_market_base_contract() -> None:
    assert not _BASE_CONTRACT_ENV.search(
        (_DOCKER / "Dockerfile.runtime").read_text(encoding="utf-8")
    )


def test_composed_builds_use_the_staged_market_recipe_and_build_only_infra() -> None:
    services = dev_stack_services()
    composed = {
        name: service["build"]
        for name, service in services.items()
        if isinstance(service.get("build"), Mapping)
        and str(service["build"].get("context", "")).endswith(
            "sibling-repos/omnimarket"
        )
    }
    assert composed, "no composed runtime services found"
    for name, build in composed.items():
        assert build["dockerfile"] == "Dockerfile.runtime", name
        assert build["args"]["INFRA_RUNTIME_IMAGE"] == "infra-runtime", name
        assert build["additional_contexts"]["infra-runtime"] == (
            "service:dev-lane-infra-runtime-image"
        ), name
    infra = services["dev-lane-infra-runtime-image"]
    assert infra["build"]["dockerfile"] == "docker/Dockerfile.runtime"
    assert infra["scale"] == 0
    assert "runtime" in infra["profiles"]
