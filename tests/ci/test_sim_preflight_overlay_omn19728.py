# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Static isolation assertions for the disposable OMN-19728 sim overlay."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.ci

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "docker" / "docker-compose.sim-preflight.yml"
RENDERER = ROOT / "scripts" / "runtime_build" / "render_sim_preflight_compose.sh"

_RENDER_INPUTS = {
    "DOGFOOD_REDPANDA_ADVERTISED_HOST": "127.0.0.1",
    "DOGFOOD_RUNTIME_EFFECTS_BIFROST_VERIFY_ENDPOINTS": "false",
    "DOGFOOD_RUNTIME_EFFECTS_CAPABILITIES": "[]",
    "DOGFOOD_RUNTIME_EFFECTS_OMNIMEMORY_ENABLED": "false",
    "DOGFOOD_RUNTIME_EFFECTS_OMNIMEMORY_MEMGRAPH_HOST": "memgraph",
    "DOGFOOD_RUNTIME_EFFECTS_PORT": "8085",
    "DOGFOOD_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON": "{}",
    "DOGFOOD_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_PATH": "/var/empty/config",
    "DOGFOOD_RUNTIME_MAIN_BIFROST_VERIFY_ENDPOINTS": "false",
    "DOGFOOD_RUNTIME_MAIN_CAPABILITIES": "[]",
    "DOGFOOD_RUNTIME_MAIN_OMNIMEMORY_ENABLED": "false",
    "DOGFOOD_RUNTIME_MAIN_OMNIMEMORY_MEMGRAPH_HOST": "memgraph",
    "DOGFOOD_RUNTIME_MAIN_PORT": "8085",
    "DOGFOOD_RUNTIME_MAIN_PUBLISH_INTROSPECTION": "false",
    "DOGFOOD_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON": "{}",
    "DOGFOOD_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH": "/var/empty/config",
    "DOGFOOD_TOPIC_PROVISIONER_MAX_PARTITIONS": "1",
    "LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST": "example.invalid",
    "OMNICLAUDE_SKILLS_DIR": "/var/empty",
    "OMNIMEMORY_MEMGRAPH_PORT": "7687",
    "OMNINODE_RUNTIME_PASSWORD": "render-only-password",
    "ONEX_ACTIVE_RUNTIME_PACKAGES": "omnibase_infra",
    "POSTGRES_PASSWORD": "render-only-password",
    "SIM_202_RUNTIME_IMAGE": "example.invalid/sim@sha256:" + "a" * 64,
    "TENANT_PROJECTION_WRITER_PASSWORD": "render-only-password",
    "VALKEY_PASSWORD": "render-only-password",
}


def test_sim_preflight_overlay_has_no_shared_identifiers_or_ambient_credentials() -> (
    None
):
    """The rendered-chain overlay must remain a separate, credential-blank lane."""
    raw = OVERLAY.read_text(encoding="utf-8")
    assert "name: omnibase-infra-sim-preflight" in raw
    assert "name: omnibase-infra-sim-202" not in raw
    assert "-p omnibase-infra-sim-202` is forbidden" in raw
    assert "omnibase-infra-dogfood-network" in raw
    assert "name: omnibase-infra-sim-preflight-network" in raw
    for name in (
        'GEMINI_API_KEY: ""',
        'OPENROUTER_API_KEY: ""',
        'LLM_GLM_API_KEY: ""',
        'LOCAL_LLM_SHARED_SECRET: ""',
        'GITHUB_TOKEN: ""',
        'LINEAR_API_KEY: ""',
    ):
        assert name in raw
    assert 'profiles: !override ["sim-preflight-faults-disabled"]' in raw
    assert "127.0.0.1:65092:19092" in raw
    assert "127.0.0.1:65444:9644" in raw
    assert "127.0.0.1:65036:5432" in raw


def test_renderer_pins_only_the_disposable_project_and_has_no_lifecycle_mode() -> None:
    raw = RENDERER.read_text(encoding="utf-8")
    assert 'readonly PROJECT="omnibase-infra-sim-preflight"' in raw
    assert 'COMPOSE_PROJECT_NAME="${PROJECT}"' in raw
    assert 'docker compose -p "${PROJECT}"' in raw
    assert " config --format json" in raw
    assert " up " not in raw
    assert " down " not in raw


@pytest.mark.skipif(shutil.which("docker") is None, reason="compose renderer absent")
def test_synthetic_render_has_no_fault_services_or_ambient_credentials() -> None:
    """Parse the effective multi-file config; render-only values are never printed."""
    env = {"HOME": os.environ.get("HOME", ""), "PATH": os.environ.get("PATH", "")}
    env.update(_RENDER_INPUTS)
    result = subprocess.run(
        [str(RENDERER)], cwd=ROOT, check=True, capture_output=True, env=env, text=True
    )
    rendered = cast("dict[str, Any]", json.loads(result.stdout))
    services = cast("dict[str, dict[str, Any]]", rendered["services"])
    assert not {"dogfood-delegation-fault-429", "dogfood-delegation-fault-503"} & set(
        services
    )
    containers = [service.get("container_name") for service in services.values()]
    assert len(containers) == len(set(containers))
    for service in services.values():
        for port in cast("list[dict[str, Any]]", service.get("ports", [])):
            assert port["host_ip"] == "127.0.0.1"
    for service_name in ("omninode-runtime", "runtime-effects", "projection-api"):
        environment = cast("dict[str, str]", services[service_name]["environment"])
        for key in (
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "LLM_GLM_API_KEY",
            "LOCAL_LLM_SHARED_SECRET",
            "GITHUB_TOKEN",
            "GH_TOKEN",
            "LINEAR_API_KEY",
        ):
            assert environment[key] == ""
