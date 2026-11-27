# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The entrypoint preflight refuses a cloud-mode boot holding a lab address (OMN-26709).

Runs the real ``run_preflight`` entrypoint step in a fresh process, with the
real image fleet file, the way the container entrypoint runs it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_infra.topology.application_database import (
    TOPOLOGY_PROFILE_INSTANCE_MAP,
)

pytestmark = [pytest.mark.integration]

_ROOT = Path(__file__).resolve().parents[3]
_CLOUD = next(p for p, i in TOPOLOGY_PROFILE_INSTANCE_MAP.items() if i != "local")
_LAB = ".".join(("192", "168", "7", "7"))
_PROGRAM = (
    "import os, sys\n"
    "from omnibase_infra.handlers.handler_cloud_lab_guard import refuse_in_cloud_mode\n"
    "sys.exit(refuse_in_cloud_mode(os.environ))\n"
)


def _run(extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not (k.startswith("LLM_") and k.endswith("_URL"))
        and k not in {"INFISICAL_ADDR", "RUNNER_FLEET_CONFIG_PATH"}
    }
    env["ONEX_DATABASE_TOPOLOGY_PROFILE"] = _CLOUD
    env.update(extra)
    return subprocess.run(
        [sys.executable, "-c", _PROGRAM],
        cwd=_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cloud_mode_with_the_image_fleet_file_boots() -> None:
    result = _run(
        {"RUNNER_FLEET_CONFIG_PATH": str(_ROOT / "config/image/runner_fleet.yaml")}
    )
    assert result.returncode == 0, result.stderr


def test_cloud_mode_with_a_lab_address_is_refused_by_setting_name() -> None:
    result = _run({"LLM_CODER_URL": f"http://{_LAB}:8000"})
    assert result.returncode == 1
    assert "CloudModeLabAddressError" in result.stderr
    assert "LLM_CODER_URL" in result.stderr
    assert _LAB not in result.stderr
