# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev broker memory policy agrees from contract to deploy preflight (OMN-20275).

OPERATOR-CONSENT 2026-10-27T06:37:48Z authorizes only the dev broker at a 12G
arena inside a 16g container cap. This joins the runtime policy contract, its
typed model, the rendered and committed policy env, the dev-lane overlay and
the deploy preflight bound, without Docker, credentials or a running broker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from scripts.preflight_required_compose_env import _check_broker_memory
from scripts.render_runtime_policy_env import format_dotenv, render_env
from tests.ci._compose_render_omn21099 import load_compose

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = REPO_ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV_PATH = REPO_ROOT / "docker" / "runtime-policy.env"

pytestmark = pytest.mark.integration


def _raw_contract() -> dict[str, Any]:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", maxsplit=1)
        env[key] = value
    return env


def test_contract_renders_the_consented_dev_broker_memory() -> None:
    contract = ModelRuntimePolicyContract.model_validate(_raw_contract())
    rendered = render_env(contract)

    assert rendered["DEV_REDPANDA_MEMORY"] == "12G"
    assert rendered["DEV_REDPANDA_MEM_LIMIT"] == "16g"
    assert POLICY_ENV_PATH.read_text(encoding="utf-8") == format_dotenv(rendered)
    committed = _load_dotenv(POLICY_ENV_PATH)
    assert committed["DEV_REDPANDA_MEMORY"] == rendered["DEV_REDPANDA_MEMORY"]
    assert committed["DEV_REDPANDA_MEM_LIMIT"] == rendered["DEV_REDPANDA_MEM_LIMIT"]


def test_rendered_policy_passes_preflight_and_matches_the_overlay() -> None:
    rendered = render_env(ModelRuntimePolicyContract.model_validate(_raw_contract()))
    _check_broker_memory(
        {
            "command": ["--memory", rendered["DEV_REDPANDA_MEMORY"]],
            "mem_limit": rendered["DEV_REDPANDA_MEM_LIMIT"],
        }
    )

    broker = load_compose("docker-compose.dev-lane.yml")["services"]["redpanda"]
    assert f"--memory {rendered['DEV_REDPANDA_MEMORY']}" in broker["command"]
    assert broker["mem_limit"] == rendered["DEV_REDPANDA_MEM_LIMIT"]
    _check_broker_memory(broker)


@pytest.mark.parametrize(
    ("memory", "limit"),
    [("3G", "4g"), ("6G", "8g"), ("12G", "8g"), ("13G", "16g"), ("12G", "17g")],
)
def test_contract_model_refuses_unconsented_dev_broker_memory(
    memory: str, limit: str
) -> None:
    raw = _raw_contract()
    raw["dev_redpanda_memory"] = memory
    raw["dev_redpanda_mem_limit"] = limit

    with pytest.raises(ValidationError):
        ModelRuntimePolicyContract.model_validate(raw)
    with pytest.raises(ValueError):
        _check_broker_memory({"command": ["--memory", memory], "mem_limit": limit})
