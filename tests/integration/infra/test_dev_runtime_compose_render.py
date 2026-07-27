# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-mutating compose render checks for the dev-lane Redpanda advertise host.

OMN-15173: `docker-compose.infra.yml` previously defaulted the dev lane's
Redpanda advertise host to `localhost` via `${DEV_REDPANDA_ADVERTISE_HOST:-localhost}`
whenever the env var was unset — silently rendering an address unreachable by
any off-host client (CI runner, another machine). These tests prove the fix:
an unset var now fails the compose render loudly instead of the client failing
silently later, and an explicitly-set var is honored (never overridden with a
localhost fallback).

This module only ever invokes `docker compose config` (a non-mutating render)
— it never brings up, restarts, or otherwise mutates any lane.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.infra.yml"

# NOTE: docker-compose.infra.yml (bare, no overlay) is the dev lane's own
# compose file (scripts/deploy-runtime.sh: "Dev lane: infra.yml alone"). A
# `docker compose config` render interpolates every service's env block
# regardless of --profile, so every other :?-required var in the file must
# still be supplied here even though this suite only cares about
# DEV_REDPANDA_ADVERTISE_HOST. Kept in sync with the fixture in
# tests/integration/docker/test_docker_integration.py::test_compose_config_valid
# by the tests/ci/test_compose_required_env_coverage.py CI gate (every :?
# var in compose must appear in that fixture; this module mirrors it rather
# than importing it, matching the existing per-file convention used by
# test_prod_runtime_compose_render.py / test_stability_test_runtime_compose_render.py).
_PG_DSN = "postgresql://postgres:test@postgres:5432/omnibase_infra"
_INTEL_DSN = "postgresql://postgres:test@postgres:5432/omniintelligence"
_LOCAL_LAN_CIDR = ".".join(("192", "168", "86", "0")) + "/24"
_SECRET_RESOLVER_CONFIG_JSON = (
    '{"enable_convention_fallback":false,"mappings":['
    '{"logical_name":"llm.openrouter.api_key",'
    '"source":{"source_path":"OPEN_ROUTER_API_KEY","source_type":"env"}}]}'
)
_SECRET_RESOLVER_CONFIG_PATH = "/app/data/delegation/secret_resolver.yaml"

# Every :?-required var in docker-compose.infra.yml EXCEPT
# DEV_REDPANDA_ADVERTISE_HOST, which each test sets (or omits) explicitly.
BASE_REQUIRED_ENV: dict[str, str] = {
    "POSTGRES_PASSWORD": "test",
    "VALKEY_PASSWORD": "test",
    "INFISICAL_ENCRYPTION_KEY": "0" * 64,
    "INFISICAL_AUTH_SECRET": "test-auth-secret",
    "OMNIBASE_INFRA_DB_URL": _PG_DSN,
    "OMNIINTELLIGENCE_DB_URL": _INTEL_DSN,
    "INFISICAL_DB_CONNECTION_URI": "postgresql://postgres:test@postgres:5432/infisical_db",
    "INFISICAL_REDIS_URL": "redis://:test@valkey:6379",
    "OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN": _PG_DSN,
    "OMNIBASE_INFRA_SKILL_LIFECYCLE_POSTGRES_DSN": _PG_DSN,
    "OMNIBASE_INFRA_CONTEXT_AUDIT_POSTGRES_DSN": _PG_DSN,
    "KAFKA_BOOTSTRAP_SERVERS": "localhost:19092",  # kafka-fallback-ok — test fixture
    "ARCH_GRAPH_BOLT_URI": "bolt://omnibase-infra-memgraph:7687",
    "ONEX_REGISTRATION_AUTO_ACK": "true",
    "ONEX_SERVICE_CLIENT_SECRET": "test-service-secret",
    "LINEAR_API_KEY": "test-linear-api-key",
    "GITHUB_TOKEN": "test-github-token",
    "LLM_CODER_URL": "http://llm-coder.test:8000",
    "LLM_CODER_FAST_URL": "http://llm-coder-fast.test:8001",
    "LLM_EMBEDDING_URL": "http://llm-embed.test:8100",
    "LLM_DEEPSEEK_R1_URL": "http://llm-r1.test:8101",
    "BIFROST_LOCAL_CODER_ENDPOINT_URL": "http://llm-coder.test:8000/v1/chat/completions",
    "BIFROST_LOCAL_REASONER_ENDPOINT_URL": (
        "http://llm-coder-fast.test:8001/v1/chat/completions"
    ),
    "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL": (
        "http://llm-embed.test:8100/v1/chat/completions"
    ),
    "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL": "http://llm-r1.test:8101/v1/chat/completions",
    "LLM_GLM_URL": "http://llm-glm.test:8102",
    "LLM_GLM_MODEL_NAME": "glm-4.5",
    "LLM_GLM_API_KEY": "render-only-glm-api-key",
    "GEMINI_API_KEY": "render-only-gemini-api-key",
    "GOOGLE_API_KEY": "render-only-google-api-key",
    "BIFROST_VERTEX_GEMINI_ENDPOINT_URL": (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/"
        "gen-lang-client-0084338881/locations/us-central1/endpoints/openapi/chat/completions"
    ),
    "GOOGLE_CLOUD_PROJECT": "gen-lang-client-0084338881",
    "GOOGLE_CLOUD_LOCATION": "us-central1",
    "LOCAL_LLM_SHARED_SECRET": "render-only-local-llm-secret",
    "LLM_ENDPOINT_CIDR_ALLOWLIST": _LOCAL_LAN_CIDR,
    "LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST": "generativelanguage.googleapis.com,api.z.ai",
    "AUXILIARY_SERVICES_OMNIMEMORY_ENABLED": "false",
    "BIFROST_VERIFY_ENDPOINTS": "1",
    "DEV_RUNTIME_EFFECTS_CAPABILITIES": "effects.consumer,market.skill-proof,runtime.effects",
    "DEV_RUNTIME_EFFECTS_PORT": "8086",
    "DEV_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_JSON": _SECRET_RESOLVER_CONFIG_JSON,
    "DEV_RUNTIME_EFFECTS_SECRET_RESOLVER_CONFIG_PATH": _SECRET_RESOLVER_CONFIG_PATH,
    "DEV_RUNTIME_MAIN_CAPABILITIES": "market.skill-proof,workflow.orchestration,runtime.main",
    "DEV_RUNTIME_MAIN_PORT": "8085",
    "DEV_RUNTIME_MAIN_PUBLISH_INTROSPECTION": "true",
    "DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_JSON": _SECRET_RESOLVER_CONFIG_JSON,
    "DEV_RUNTIME_MAIN_SECRET_RESOLVER_CONFIG_PATH": _SECRET_RESOLVER_CONFIG_PATH,
    "DEV_RUNTIME_WORKER_CAPABILITIES": "workflow.dispatch,contract.update,runtime.worker",
    "DEV_RUNTIME_WORKER_SECRET_RESOLVER_CONFIG_JSON": _SECRET_RESOLVER_CONFIG_JSON,
    "DEV_RUNTIME_WORKER_SECRET_RESOLVER_CONFIG_PATH": _SECRET_RESOLVER_CONFIG_PATH,
    "OMNIMEMORY_ENABLED": "false",
    "OMNIMEMORY_MEMGRAPH_PORT": "7687",
    "ONEX_ACTIVE_RUNTIME_PACKAGES": "omnibase_infra,omnimarket",
}

# RFC 5737 TEST-NET-2 documentation address — never a real host, avoids
# asserting against any live LAN/Tailscale identity.
_OFF_HOST_ADVERTISE_HOST = "198.51.100.50"


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _render_env(**overrides: str) -> dict[str, str]:
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "USER": os.environ.get("USER", ""),
        **BASE_REQUIRED_ENV,
    }
    env.update(overrides)
    return env


def _run_compose_config(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "docker/runtime-policy.env",
            "-f",
            str(COMPOSE_FILE),
            "config",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=60,
    )


pytestmark = pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose is required for non-mutating compose render validation",
)


@pytest.mark.integration
def test_dev_redpanda_advertise_host_fails_fast_when_unset() -> None:
    """Unset DEV_REDPANDA_ADVERTISE_HOST must fail the compose render, never
    silently render a localhost advertise address."""
    env = _render_env()
    assert "DEV_REDPANDA_ADVERTISE_HOST" not in env

    result = _run_compose_config(env)

    assert result.returncode != 0, (
        "docker compose config unexpectedly succeeded with "
        "DEV_REDPANDA_ADVERTISE_HOST unset:\n" + result.stdout
    )
    assert "DEV_REDPANDA_ADVERTISE_HOST" in result.stderr


@pytest.mark.integration
def test_dev_redpanda_advertise_host_uses_explicit_value_when_set() -> None:
    """An explicitly-set DEV_REDPANDA_ADVERTISE_HOST is honored verbatim —
    never silently overridden with a localhost fallback."""
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    assert f"{_OFF_HOST_ADVERTISE_HOST}:19092" in result.stdout
    assert f"{_OFF_HOST_ADVERTISE_HOST}:18082" in result.stdout
    assert "localhost:19092" not in result.stdout
