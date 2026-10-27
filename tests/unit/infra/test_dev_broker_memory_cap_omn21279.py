# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev broker obeys OPERATOR-CONSENT 2026-10-27T06:37:48Z (OMN-20275)."""

from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from dotenv import dotenv_values

from tests.ci._compose_render_omn21099 import DEV_LANE_STACK, load_compose, render_stack

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def _docker_env() -> dict[str, str]:
    return {
        key: os.environ[key]
        for key in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONFIG", "XDG_RUNTIME_DIR")
        if key in os.environ
    }


@pytest.fixture(scope="module")
def docker_compose() -> None:
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is required for preflight Compose rendering")
    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=_docker_env(),
    )
    if result.returncode:
        pytest.skip("docker compose plugin is required for preflight Compose rendering")


def _checker() -> ModuleType:
    path = ROOT / "scripts/preflight_required_compose_env.py"
    spec = importlib.util.spec_from_file_location("broker_cap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_checker_accepts_the_real_dev_broker_configuration() -> None:
    # The overlay replaces the base command and fixes both bounds as literals.
    services = load_compose("docker-compose.dev-lane.yml")["services"]
    broker = services["redpanda"]
    command = " ".join(broker["command"])
    assert "--memory 12G" in command
    assert broker["mem_limit"] == "16g"
    _checker()._check_broker_memory(broker)
    assert (
        services["redpanda-partition-cap"]["environment"]["REDPANDA_PROFILE_MEMORY"]
        == "12G"
    )


@pytest.mark.parametrize(
    ("memory", "limit"),
    [
        ("1G", "8g"),
        ("12G", "8g"),
        ("7G", "8g"),
        ("6145M", "8g"),
        ("3G", "4g"),
        ("6G", "8g"),
        ("13G", "16g"),
        ("12G", "17g"),
        ("3G", 0),
        ("3G", "9g"),
        ("3G", "8193m"),
        ("3G", "1g"),
        ("6G", "5g"),
        ("4097M", "4g"),
        ("oops", "8g"),
    ],
)
def test_checker_refuses_unruled_or_missing_memory(memory: str, limit: object) -> None:
    with pytest.raises(ValueError):
        _checker()._check_broker_memory(
            {"command": ["redpanda", "start", f"--memory {memory}"], "mem_limit": limit}
        )


@pytest.mark.parametrize(
    ("memory", "limit"),
    [
        ("12G", "16g"),
        ("12G", 16 * 1024**3),
        ("12288M", "16384m"),
    ],
)
def test_checker_accepts_the_ruled_boundaries(memory: str, limit: object) -> None:
    _checker()._check_broker_memory(
        {"command": ["redpanda", "start", "--memory", memory], "mem_limit": limit}
    )


@pytest.mark.parametrize("command", [[], ["--memory"], ["--memory 3G", "--memory=12G"]])
def test_checker_refuses_missing_or_ambiguous_flag(command: list[str]) -> None:
    with pytest.raises(ValueError):
        _checker()._check_broker_memory({"command": command, "mem_limit": "8g"})


@pytest.mark.parametrize(
    ("memory", "limit", "expected"),
    [
        ("12G", "8g", 1),
        ("3G", "0", 1),
        ("6G", "8g", 1),
        ("3G", "4g", 1),
        ("12G", "16g", 0),
    ],
)
@pytest.mark.usefixtures("docker_compose")
def test_deploy_preflight_checks_the_rendered_broker(
    tmp_path: Path, memory: str, limit: str, expected: int
) -> None:
    compose = tmp_path / "docker-compose.dev-lane.yml"
    compose.write_text(
        f"services:\n  redpanda:\n    image: redpandadata/redpanda:v24.3.18\n"
        f"    command: [redpanda, start, '--memory {memory}']\n    mem_limit: {limit}\n"
    )
    policy = tmp_path / "policy.env"
    policy.write_text("")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/preflight_required_compose_env.py"),
            "--lane",
            "dev",
            "--compose-file",
            str(compose),
            "--runtime-policy-env",
            str(policy),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=_docker_env(),
    )
    assert result.returncode == expected, result.stderr
    if expected:
        assert "DEV_BROKER_MEMORY_CAP_REFUSED" in result.stderr


@pytest.mark.parametrize("advertise_host", ["198.51.100.50", None])
@pytest.mark.usefixtures("docker_compose")
def test_deploy_preflight_checks_the_real_dev_stack(advertise_host: str | None) -> None:
    # Same file order as resolve_compose_file_args() and the deploy agent.
    env = _docker_env()
    for name in DEV_LANE_STACK:
        contents = (ROOT / "docker" / name).read_text(encoding="utf-8")
        for key in re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?", contents):
            env[key] = "render-only"
    env.update(
        {
            key: value
            for key, value in dotenv_values(ROOT / "docker/runtime-policy.env").items()
            if value is not None
        }
    )
    env.update(
        OMNI_HOME=str(ROOT),
        OMNICLAUDE_SKILLS_DIR=str(ROOT / "tests"),
        ONEX_DEPLOY_AGENT_ENV_FILE=str(ROOT / ".env.example"),
        ONEX_LANE_OVERLAY_DIR=str(ROOT / "tests"),
        ONEX_LAB_TENANT_STATE_DIR=str(ROOT / "tests"),
        OMNINODE_RUNTIME_PASSWORD="a" * 64,
        TENANT_PROJECTION_WRITER_PASSWORD="b" * 64,
        DEV_REDPANDA_MEMORY="12G",
    )
    env.pop("DEV_REDPANDA_ADVERTISE_HOST", None)
    if advertise_host is not None:
        env["DEV_REDPANDA_ADVERTISE_HOST"] = advertise_host
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/preflight_required_compose_env.py"),
            "--lane",
            "dev",
            "--compose-file",
            str(ROOT / "docker/docker-compose.infra.yml"),
            "--compose-file",
            str(ROOT / "docker/docker-compose.dev-lane.yml"),
            "--runtime-policy-env",
            str(ROOT / "docker/runtime-policy.env"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )
    if advertise_host is None:
        assert result.returncode == 1
        assert "REQUIRED_COMPOSE_ENV_MISSING — 1 of " in result.stderr
        assert "DEV_REDPANDA_ADVERTISE_HOST" in result.stderr
        assert "DEV_BROKER_MEMORY_CAP_REFUSED" not in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "OK: all" in result.stdout


@pytest.mark.parametrize("memory", ["3G", "6G", "12G", "24G"])
@pytest.mark.usefixtures("docker_compose")
def test_rendered_dev_broker_keeps_consented_12g_16g(memory: str) -> None:
    services = render_stack(DEV_LANE_STACK, {"DEV_REDPANDA_MEMORY": memory})
    broker = services["redpanda"]
    assert "--memory 12G" in " ".join(broker["command"])
    assert int(broker["mem_limit"]) == 16 * 1024**3
    assert (
        services["redpanda-partition-cap"]["environment"]["REDPANDA_PROFILE_MEMORY"]
        == "12G"
    )


def test_deploy_agent_env_declares_consented_dev_memory() -> None:
    policy = dotenv_values(ROOT / "docker/runtime-policy.env")
    assert policy["DEV_REDPANDA_MEMORY"] == "12G"
    assert policy["DEV_REDPANDA_MEM_LIMIT"] == "16g"
