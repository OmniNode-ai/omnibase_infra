# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime policy contract and compose boundary checks."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "services" / "runtime_policy.contract.yaml"
POLICY_ENV_PATH = ROOT / "docker" / "runtime-policy.env"
COMPOSE_PATH = ROOT / "docker" / "docker-compose.infra.yml"
STABILITY_COMPOSE_PATH = ROOT / "docker" / "docker-compose.stability-test.yml"
PROD_COMPOSE_PATH = ROOT / "docker" / "docker-compose.prod.yml"

pytestmark = pytest.mark.unit


def _load_contract() -> ModelRuntimePolicyContract:
    raw = yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    return ModelRuntimePolicyContract.model_validate(raw)


def _load_dotenv(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def test_runtime_policy_contract_declares_runtime_lanes() -> None:
    contract = _load_contract()

    assert set(contract.profiles) == {"dev", "stability-test", "judge", "prod"}
    assert contract.profiles["dev"].main_port == 8085
    assert contract.profiles["dev"].effects_port == 8086
    assert contract.profiles["stability-test"].main_port == 18085
    assert contract.profiles["stability-test"].effects_port == 18086
    assert contract.profiles["stability-test"].topic_provisioner_max_partitions == 1
    assert contract.profiles["judge"].main_port == 48085
    assert contract.profiles["judge"].effects_port == 48086
    assert contract.profiles["judge"].topic_provisioner_max_partitions == 1
    assert contract.profiles["prod"].main_port == 28085
    assert contract.profiles["prod"].effects_port == 28086


def test_runtime_policy_env_matches_contract_renderer() -> None:
    from scripts.render_runtime_policy_env import format_dotenv, render_env

    contract = _load_contract()
    expected = format_dotenv(render_env(contract))
    observed = POLICY_ENV_PATH.read_text(encoding="utf-8")

    assert observed == expected


def test_runtime_policy_env_shell_source_preserves_json() -> None:
    keys = " ".join(
        [
            "DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON",
            "DEV_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON",
            "DEV_RUNTIME_WORKER_SECRET_RESOLVER_CONFIG_JSON",
            "STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON",
            "STABILITY_TEST_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON",
            "STABILITY_TEST_RUNTIME_WORKER_SECRET_RESOLVER_CONFIG_JSON",
            "JUDGE_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON",
            "JUDGE_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON",
            "JUDGE_RUNTIME_WORKER_SECRET_RESOLVER_CONFIG_JSON",
        ]
    )
    command = f"""
set -euo pipefail
set -a
source {shlex.quote(str(POLICY_ENV_PATH))}
set +a
{shlex.quote(sys.executable)} - {keys} <<'PY'
import json
import os
import sys

for key in sys.argv[1:]:
    payload = json.loads(os.environ[key])
    assert payload["enable_convention_fallback"] is False
    assert payload["mappings"]
PY
"""

    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_policy_env_has_expected_lane_values() -> None:
    env = _load_dotenv(POLICY_ENV_PATH)

    assert env["AUXILIARY_SERVICES_OMNIMEMORY_ENABLED"] == "false"
    assert env["ONEX_ACTIVE_RUNTIME_PACKAGES"] == "omnibase_infra,omnimarket"
    assert (
        env["LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST"]
        == "generativelanguage.googleapis.com,api.z.ai"
    )
    assert env["DEV_RUNTIME_MAIN_PORT"] == "8085"
    assert env["STABILITY_TEST_RUNTIME_MAIN_PORT"] == "18085"
    assert env["STABILITY_TEST_TOPIC_PROVISIONER_MAX_PARTITIONS"] == "1"
    assert env["JUDGE_RUNTIME_MAIN_PORT"] == "48085"
    assert env["JUDGE_RUNTIME_EFFECTS_PORT"] == "48086"
    assert env["JUDGE_TOPIC_PROVISIONER_MAX_PARTITIONS"] == "1"
    assert env["PROD_RUNTIME_MAIN_PORT"] == "28085"
    assert (
        env["STABILITY_TEST_RUNTIME_MAIN_CAPABILITIES"]
        == "market.skill-proof,workflow.orchestration,runtime.main"
    )
    assert (
        env["DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH"]
        == "/app/data/delegation/secret_resolver.yaml"
    )
    assert (
        env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH"]
        == "/app/data/delegation/secret_resolver.yaml"
    )
    assert (
        env["JUDGE_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH"]
        == "/app/data/delegation/secret_resolver.yaml"
    )
    assert (
        "llm.openrouter.api_key" in env["DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
    assert (
        '"source_type":"infisical"'
        in env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
    assert (
        "OPENROUTER_API_KEY"
        in env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
    assert (
        "OPEN_ROUTER_API_KEY"
        not in env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
    assert (
        "llm.vertex.access_token"
        not in env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )


def test_runtime_policy_llm_secret_refs_resolve_from_infisical_only() -> None:
    contract = _load_contract()

    for profile_name, profile in contract.profiles.items():
        for mapping in profile.secret_resolver_mappings:
            assert mapping.source.source_type == "infisical", profile_name
            assert mapping.logical_name != "llm.vertex.access_token", profile_name


def test_compose_consumes_policy_env_instead_of_hardcoded_policy_values() -> None:
    compose_text = COMPOSE_PATH.read_text(encoding="utf-8")
    stability_text = STABILITY_COMPOSE_PATH.read_text(encoding="utf-8")
    prod_text = PROD_COMPOSE_PATH.read_text(encoding="utf-8")

    assert (
        "ONEX_ACTIVE_RUNTIME_PACKAGES: ${ONEX_ACTIVE_RUNTIME_PACKAGES:-"
        not in compose_text
    )
    assert "OMNIMEMORY_ENABLED: ${OMNIMEMORY_ENABLED:-" not in compose_text
    assert 'OMNIINTELLIGENCE_PUBLISH_INTROSPECTION: "true"' not in compose_text
    assert "ONEX_SECRET_RESOLVER_CONFIG_PATH: /app/data" not in compose_text
    assert "llm.openrouter.api_key" not in compose_text

    for text in (stability_text, prod_text):
        assert 'OMNIMEMORY_ENABLED: ""' not in text
        assert 'OMNIMEMORY_MEMGRAPH_HOST: ""' not in text
        assert 'BIFROST_VERIFY_ENDPOINTS: "0"' not in text
        assert "ONEX_RUNTIME_CAPABILITIES: market.skill-proof" not in text
        assert "ONEX_RUNTIME_CAPABILITIES: effects.consumer" not in text
        assert "ONEX_RUNTIME_CAPABILITIES: workflow.dispatch" not in text


def test_worker_replicas_pinned_in_contract_for_every_lane() -> None:
    """OMN-12990: each lane's worker process declares an explicit replica pin.

    The base compose default is ``${WORKER_REPLICAS:-0}`` (silent drop). The
    contract worker process must carry a contract-declared replica count >= 1 so
    the rendered policy env preserves the worker on a lane recreate.
    """
    contract = _load_contract()

    for profile_name, profile in contract.profiles.items():
        worker = profile.processes["worker"]
        assert worker.replicas >= 1, profile_name


def test_worker_replicas_rendered_into_policy_env_for_every_lane() -> None:
    """OMN-12990: the renderer emits ``{PROFILE}_WORKER_REPLICAS`` per lane.

    These are the ledgered config values the lane overrides reference fail-fast.
    """
    env = _load_dotenv(POLICY_ENV_PATH)

    assert env["DEV_WORKER_REPLICAS"] == "1"
    assert env["STABILITY_TEST_WORKER_REPLICAS"] == "1"
    assert env["JUDGE_WORKER_REPLICAS"] == "1"
    assert env["PROD_WORKER_REPLICAS"] == "1"


def test_runtime_worker_replicas_are_fail_fast_not_silent_default() -> None:
    """OMN-12990: lane overrides reference the policy value with fail-fast ``:?``.

    A soft ``:-1`` / ``:-0`` default would silently re-introduce the silent-drop
    hole on any recreate that omitted the policy env. The base compose keeps its
    ``${WORKER_REPLICAS:-0}`` default (that is the gap the overrides close), so a
    plain recreate against the base alone still scales to zero — by design the
    lane overrides MUST be applied, and they now abort loudly when the policy env
    is missing rather than dropping the worker with no signal.
    """
    stability_text = STABILITY_COMPOSE_PATH.read_text(encoding="utf-8")
    prod_text = PROD_COMPOSE_PATH.read_text(encoding="utf-8")

    assert "${STABILITY_TEST_WORKER_REPLICAS:?" in stability_text, (
        "stability worker replicas must be fail-fast on the ledgered policy value"
    )
    assert "${STABILITY_TEST_WORKER_REPLICAS:-" not in stability_text
    assert "${PROD_WORKER_REPLICAS:?" in prod_text, (
        "prod worker replicas must be fail-fast on the ledgered policy value"
    )
    assert "${PROD_WORKER_REPLICAS:-" not in prod_text
