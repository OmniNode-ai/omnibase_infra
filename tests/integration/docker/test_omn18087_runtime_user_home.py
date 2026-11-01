# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose renders credential and workspace paths for the runtime account."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("username", [None, "readback"])
@pytest.mark.parametrize("source", ["infra", "main-catalog", "effects-catalog"])
def test_runtime_account_paths_render_together(
    source: str, username: str | None
) -> None:
    if source == "infra":
        config = yaml.safe_load((ROOT / "docker/docker-compose.infra.yml").read_text())
        runtime = config["services"]["runtime-effects"]
        environment = runtime["environment"]
    else:
        name = "omninode-runtime" if source == "main-catalog" else "runtime-effects"
        runtime = yaml.safe_load(
            (ROOT / f"docker/catalog/services/{name}.yaml").read_text()
        )
        environment = runtime["hardcoded_env"]
    # Render only the runtime's account paths; config never starts a container.
    service = {
        "image": "runtime:latest",
        "environment": {
            k: environment[k]
            for k in ("CODING_AGENT_CRED_HOME", "ONEX_CODING_AGENT_WORKSPACE_ROOT")
            if k in environment
        },
        "volumes": [v for v in runtime["volumes"] if "/home/" in v],
    }
    env = dict(os.environ)
    env.pop("RUNTIME_USER_NAME", None)
    if username is not None:
        env["RUNTIME_USER_NAME"] = username
    env["COMPOSE_PROJECT_NAME"] = "omn18087-home-readback"
    result = subprocess.run(
        ["docker", "compose", "-f", "-", "config", "--format", "json"],
        input=json.dumps(
            {
                "services": {"runtime": service},
                "volumes": {"delegation_credentials": {}},
            }
        ),
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)["services"]["runtime"]
    home = f"/home/{username or 'omniinfra'}"
    for volume in rendered["volumes"]:
        assert volume["target"].startswith(home + "/")
    if source != "main-catalog":
        assert rendered["environment"]["CODING_AGENT_CRED_HOME"] == home
        assert (
            rendered["environment"]["ONEX_CODING_AGENT_WORKSPACE_ROOT"]
            == home + "/agent-workspaces"
        )
    assert rendered["volumes"]
