# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev lane's onex-api carries the cost estimator's Fernet key (OMN-20388).

onex-api encrypts every estimator upload, prompt and lead under
``ESTIMATOR_FERNET_KEY`` and refuses an upload it could not encrypt. The k8s
manifests carry the key from Infisical; the compose dev lane spells
onex-api's environment explicitly, so without this entry the lane's estimator
answers 503 whatever the operator env file holds.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml
from cryptography.fernet import Fernet

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_LANE = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
ENV_RENDER = (
    REPO_ROOT / "scripts" / "runtime_build" / "render_dev_lane_tenant_path_env.sh"
)


def _onex_api_env() -> dict[str, object]:
    text = DEV_LANE.read_text().replace("!!merge ", "").replace("!override", "")
    service = yaml.safe_load(text)["services"]["onex-api"]
    env = service["environment"]
    assert isinstance(env, dict)
    return env


@pytest.mark.unit
def test_onex_api_reads_the_estimator_key_from_the_operator_env() -> None:
    # `:-`: unset leaves only /v1/estimator/* at 503 rather than failing the
    # whole lane's compose render on a host that has not re-run the script.
    assert _onex_api_env().get("ESTIMATOR_FERNET_KEY") == "${ESTIMATOR_FERNET_KEY:-}"


@pytest.mark.unit
def test_the_render_script_provisions_a_valid_fernet_key(tmp_path: Path) -> None:
    env_file = tmp_path / "operator.env"
    env_file.write_text("")
    subprocess.run(
        ["bash", str(ENV_RENDER)],
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "HOME": str(tmp_path),
            "OMNIBASE_OPERATOR_ENV_FILE": str(env_file),
        },
    )
    values = dict(
        line.split("=", 1) for line in env_file.read_text().splitlines() if "=" in line
    )
    key = values["ESTIMATOR_FERNET_KEY"]
    fernet = Fernet(key.encode())
    assert fernet.decrypt(fernet.encrypt(b"probe")) == b"probe"
