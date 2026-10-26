# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev broker obeys the 2026-10-25T17:53:44Z operator consent."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tests.unit.infra.test_broker_profile_omn19419 import _render

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.unit


def _checker() -> ModuleType:
    path = ROOT / "scripts/preflight_required_compose_env.py"
    spec = importlib.util.spec_from_file_location("broker_cap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dev_broker_cap_survives_the_stale_operator_environment() -> None:
    services = _render("dev", {"DEV_REDPANDA_MEMORY": "12G"})["services"]
    broker = services["redpanda"]
    command = " ".join(broker["command"])
    assert "--memory 3G" in command
    assert int(broker["mem_limit"]) == 4 * 1024**3
    assert (
        services["redpanda-partition-cap"]["environment"]["REDPANDA_PROFILE_MEMORY"]
        == "3G"
    )


@pytest.mark.parametrize(
    ("memory", "limit"),
    [
        ("1G", "4g"),
        ("12G", "4g"),
        ("3G", 0),
        ("3G", "5g"),
        ("3G", "1g"),
        ("4G", "3g"),
        ("oops", "4g"),
    ],
)
def test_checker_refuses_unruled_or_missing_memory(memory: str, limit: object) -> None:
    with pytest.raises(ValueError):
        _checker()._check_broker_memory(
            {"command": ["redpanda", "start", f"--memory {memory}"], "mem_limit": limit}
        )


@pytest.mark.parametrize(
    ("memory", "limit"),
    [("2G", "4g"), ("3G", 4 * 1024**3), ("4G", "4g"), ("2048M", "4096m")],
)
def test_checker_accepts_the_ruled_boundaries(memory: str, limit: object) -> None:
    _checker()._check_broker_memory(
        {"command": ["redpanda", "start", "--memory", memory], "mem_limit": limit}
    )


@pytest.mark.parametrize("command", [[], ["--memory"], ["--memory 3G", "--memory=12G"]])
def test_checker_refuses_missing_or_ambiguous_flag(command: list[str]) -> None:
    with pytest.raises(ValueError):
        _checker()._check_broker_memory({"command": command, "mem_limit": "4g"})


@pytest.mark.parametrize(
    ("memory", "limit", "expected"), [("12G", "4g", 1), ("3G", "0", 1), ("3G", "4g", 0)]
)
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
        env={"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]},
    )
    assert result.returncode == expected, result.stderr
    if expected:
        assert "DEV_BROKER_MEMORY_CAP_REFUSED" in result.stderr
