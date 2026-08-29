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
        == "generativelanguage.googleapis.com,api.z.ai,aiplatform.googleapis.com"
    )
    assert (
        env["BIFROST_VERTEX_GEMINI_ENDPOINT_URL"]
        == "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/gen-lang-client-0084338881/locations/us-central1/endpoints/openapi/chat/completions"
    )
    assert env["GOOGLE_CLOUD_PROJECT"] == "gen-lang-client-0084338881"
    assert env["GOOGLE_CLOUD_LOCATION"] == "us-central1"
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
        "OPENROUTER_API_KEY"
        in env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )
    assert (
        "OPEN_ROUTER_API_KEY"
        not in env["STABILITY_TEST_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON"]
    )


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

    The contract worker process must carry a contract-declared replica count
    >= 1 so the rendered policy env preserves the worker on a lane recreate.
    Every lane's compose surface resolves that pin fail-closed (OMN-14968); a
    lane whose contract dropped below 1 would push a zero-container lane through
    a render that cannot detect it.
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
    """OMN-12990 / OMN-14968: EVERY lane surface resolves replicas fail-fast.

    A soft ``:-1`` / ``:-0`` default silently re-introduces the silent-drop hole
    on any recreate that omits the policy env.

    OMN-12990 converted the stability-test and prod overlays but left the base
    infra compose on a BARE ``${WORKER_REPLICAS:-0}``. That bare name is exported
    by no surface in this repo, so it always took the ``0`` branch — and the base
    file with no overlay IS the dev lane, so the dev lane rendered a
    zero-container worker, `up` created nothing, and the RT-6 deploy readback in
    ``scripts/deploy-runtime.sh`` (whose ``RUNTIME_SERVICES`` includes
    ``runtime-worker``) failed closed on every dev-lane deploy. OMN-14968 closed
    it with the lane-prefixed ``${DEV_WORKER_REPLICAS:?...}`` form used by the
    sibling ``DEV_RUNTIME_WORKER_*`` vars in the same service block.
    """
    base_text = COMPOSE_PATH.read_text(encoding="utf-8")
    stability_text = STABILITY_COMPOSE_PATH.read_text(encoding="utf-8")
    prod_text = PROD_COMPOSE_PATH.read_text(encoding="utf-8")

    assert "${DEV_WORKER_REPLICAS:?" in base_text, (
        "the base infra compose (the dev lane's own compose file) must resolve "
        "worker replicas fail-fast on the ledgered DEV_WORKER_REPLICAS"
    )
    assert "${DEV_WORKER_REPLICAS:-" not in base_text
    assert "replicas: ${WORKER_REPLICAS" not in base_text, (
        "the bare WORKER_REPLICAS name is exported by no surface in this repo; "
        "it always resolved to the silent 0 default (OMN-14968)"
    )
    assert "${STABILITY_TEST_WORKER_REPLICAS:?" in stability_text, (
        "stability worker replicas must be fail-fast on the ledgered policy value"
    )
    assert "${STABILITY_TEST_WORKER_REPLICAS:-" not in stability_text
    assert "${PROD_WORKER_REPLICAS:?" in prod_text, (
        "prod worker replicas must be fail-fast on the ledgered policy value"
    )
    assert "${PROD_WORKER_REPLICAS:-" not in prod_text


def test_boundary_dlq_enabled_declared_explicitly_for_every_lane() -> None:
    """OMN-14551: every lane declares an explicit boundary-DLQ stance.

    No-invisible-env-config doctrine -- the field is required (no default), so
    a lane cannot silently inherit an implicit off; each profile must name its
    position in the contract.
    """
    contract = _load_contract()

    # OMN-14551: flipped ON 2026-08-05 after a dedicated dev-lane live proof
    # (see the contract's boundary_dlq_enabled comment on the dev profile).
    assert contract.profiles["dev"].boundary_dlq_enabled is True
    assert contract.profiles["stability-test"].boundary_dlq_enabled is True
    assert contract.profiles["judge"].boundary_dlq_enabled is False
    assert contract.profiles["prod"].boundary_dlq_enabled is False


def test_boundary_dlq_enabled_rendered_into_policy_env_for_every_lane() -> None:
    """OMN-14551: the renderer emits ``{PROFILE}_BOUNDARY_DLQ_ENABLED`` per lane."""
    env = _load_dotenv(POLICY_ENV_PATH)

    assert env["DEV_BOUNDARY_DLQ_ENABLED"] == "true"
    assert env["STABILITY_TEST_BOUNDARY_DLQ_ENABLED"] == "true"
    assert env["JUDGE_BOUNDARY_DLQ_ENABLED"] == "false"
    assert env["PROD_BOUNDARY_DLQ_ENABLED"] == "false"


def test_stability_test_boundary_dlq_wired_fail_fast_prod_and_judge_untouched() -> None:
    """OMN-14551 G6: stability-test's runtime containers reference the ledgered
    ``STABILITY_TEST_BOUNDARY_DLQ_ENABLED`` value fail-fast (``:?``, no silent
    ``:-false`` default). Prod and judge compose files carry no
    ``ONEX_BOUNDARY_DLQ_ENABLED`` reference at all -- this PR's flip is scoped
    strictly to the stability-test lane.
    """
    stability_text = STABILITY_COMPOSE_PATH.read_text(encoding="utf-8")
    prod_text = PROD_COMPOSE_PATH.read_text(encoding="utf-8")
    judge_text = (ROOT / "docker" / "docker-compose.judge.yml").read_text(
        encoding="utf-8"
    )

    assert (
        stability_text.count(
            "ONEX_BOUNDARY_DLQ_ENABLED: ${STABILITY_TEST_BOUNDARY_DLQ_ENABLED:?"
        )
        == 3
    ), "expected the flag wired on main, effects, and worker"
    assert "${STABILITY_TEST_BOUNDARY_DLQ_ENABLED:-" not in stability_text
    assert "ONEX_BOUNDARY_DLQ_ENABLED" not in prod_text
    assert "ONEX_BOUNDARY_DLQ_ENABLED" not in judge_text


def test_dev_boundary_dlq_wired_fail_fast() -> None:
    """OMN-14551 dev-lane flip (2026-08-05): dev's runtime containers reference
    the ledgered ``DEV_BOUNDARY_DLQ_ENABLED`` value fail-fast (``:?``, no
    silent ``:-false`` default) via the shared ``x-runtime-env`` anchor.
    """
    dev_text = COMPOSE_PATH.read_text(encoding="utf-8")

    assert "ONEX_BOUNDARY_DLQ_ENABLED: ${DEV_BOUNDARY_DLQ_ENABLED:?" in dev_text, (
        "expected the flag wired fail-fast on the shared runtime-env anchor"
    )
    assert "${DEV_BOUNDARY_DLQ_ENABLED:-" not in dev_text


def test_secret_resolver_mappings_satisfy_gateway_boot_check(tmp_path: Path) -> None:
    """OMN-16110: every profile that sets ``secret_resolver_config_path`` must
    declare every ``gateway.attach.keycloak.*`` mapping that
    ``service_kernel.py::_build_runtime_handler_dependencies`` hard-requires at
    boot whenever a secret-resolver config path is configured (raises
    ``ProtocolConfigurationError`` otherwise -- introduced by OMN-15918 PR
    #2731, missed by this contract until OMN-16110). This drives the real
    boot-time function against each profile's contract-declared mappings, so
    the next time that function's required-ref set changes, CI fails here
    instead of a live lane crashing at container boot.
    """
    from omnibase_infra.runtime.service_kernel import (
        _build_runtime_handler_dependencies,
    )

    contract = _load_contract()
    exercised_profiles: list[str] = []
    for profile_name, profile in contract.profiles.items():
        if not profile.secret_resolver_config_path.strip():
            # No secret-resolver config path -> gateway_secret_resolver_config_path
            # resolves to None at boot and this check never runs for this profile
            # (e.g. prod today -- see runtime_profile.resolve_secret_resolver_config_path).
            continue
        exercised_profiles.append(profile_name)

        config_path = tmp_path / f"{profile_name}-secret-resolver.yaml"
        config_payload = {
            "enable_convention_fallback": False,
            "mappings": [
                mapping.model_dump(mode="json")
                for mapping in profile.secret_resolver_mappings
            ],
        }
        config_path.write_text(yaml.safe_dump(config_payload), encoding="utf-8")

        try:
            _build_runtime_handler_dependencies(
                postgres_pool=None,
                kafka_bootstrap_servers=None,
                gateway_secret_resolver_config_path=config_path,
            )
        except Exception as exc:  # noqa: BLE001 -- surface exactly what boot sees
            pytest.fail(
                f"profile {profile_name!r}'s secret_resolver_mappings do not "
                "satisfy the real service_kernel gateway-attach boot check "
                f"(this profile would crash at container boot): {exc}"
            )

    # Guard the guard: fail loudly if a future contract edit removes every
    # secret_resolver_config_path, silently turning this test into a no-op.
    assert exercised_profiles, (
        "expected at least one runtime profile to set secret_resolver_config_path"
    )
