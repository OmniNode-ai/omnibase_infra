# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-mutating compose render checks for the dev lane's silent-default holes.

Two fixes of the same class are proven here — a base-compose var with a soft
`${VAR:-default}` that failed OPEN into a wrong-but-quiet render, replaced by
the lane-prefixed fail-closed `${DEV_...:?}` form:

OMN-15173 (`DEV_REDPANDA_ADVERTISE_HOST`): the dev lane defaulted its Redpanda
advertise host to `localhost`, silently rendering an address unreachable by any
off-host client (CI runner, another machine).

OMN-14968 (`DEV_WORKER_REPLICAS`): the `runtime-worker` deploy block resolved a
BARE `${WORKER_REPLICAS:-0}` that no surface exported, so the dev lane rendered
`replicas: 0`. `docker compose up -d --no-deps runtime-worker` then exited 0
creating NOTHING, while `deploy-runtime.sh`'s `RUNTIME_SERVICES` / RT-6 deploy
readback requires a running container — so every dev-lane deploy aborted at the
readback and auto-restored. The lane-prefixed value is the ledgered policy
contract's (`DEV_WORKER_REPLICAS=1`, rendered from
`contracts/services/runtime_policy.contract.yaml`), matching what OMN-12988 /
OMN-12990 already did for the stability-test and prod overlays.

This module only ever invokes `docker compose config` (a non-mutating render)
— it never brings up, restarts, or otherwise mutates any lane.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.infra.yml"
# OMN-17448: the dev lane's SECOND `-f` file. `resolve_compose_file_args()` in
# scripts/deploy-runtime.sh appends this for the bare `omnibase-infra` project
# and never for a lane with its own overlay, so a service declared here reaches
# the dev lane and provably no other.
DEV_LANE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.dev-lane.yml"
_DEFAULT_POLICY_ENV_FILE = "docker/runtime-policy.env"
POLICY_ENV_PATH = REPO_ROOT / "docker" / "runtime-policy.env"

# NOTE: docker-compose.infra.yml (bare, no overlay) is the dev lane's own
# compose file (scripts/deploy-runtime.sh: "Dev lane: infra.yml alone"). A
# `docker compose config` render interpolates every service's env block
# regardless of --profile, so every other :?-required var in the file must
# still be supplied here even though this suite only cares about
# DEV_REDPANDA_ADVERTISE_HOST. Kept in sync by the
# tests/ci/test_compose_required_env_coverage.py CI gate, which since OMN-15263
# checks EVERY registered compose-render fixture (not just the one in
# tests/integration/docker/test_docker_integration.py). This module mirrors that
# fixture rather than importing it, matching the existing per-file convention
# used by test_prod_runtime_compose_render.py /
# test_stability_test_runtime_compose_render.py.
#
# DEV_REDPANDA_ADVERTISE_HOST is deliberately absent below and is registered in
# that gate as `intentionally_unset` for this module: supplying it here would
# make test_dev_redpanda_advertise_host_fails_fast_when_unset vacuous.
_PG_DSN = "postgresql://postgres:test@postgres:5432/omnibase_infra"
_INTEL_DSN = "postgresql://postgres:test@postgres:5432/omniintelligence"
_LOCAL_LAN_CIDR = ".".join(("192", "168", "86", "0")) + "/24"
_SECRET_RESOLVER_CONFIG_JSON = (
    '{"enable_convention_fallback":false,"mappings":['
    '{"logical_name":"llm.openrouter.api_key",'
    '"source":{"source_path":"OPENROUTER_API_KEY","source_type":"env"}}]}'
)
_SECRET_RESOLVER_CONFIG_PATH = "/app/data/delegation/secret_resolver.yaml"

# One shared synthetic value for every `${VAR:?}` name the tenant-path block
# adds (OMN-17530). A named constant rather than eleven string literals: the
# render only needs each name to be NON-EMPTY, and eleven distinct
# credential-shaped literals in a test file are eleven things a future reader
# has to confirm are not real.
_RENDER_ONLY = "render-only"

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
    "GATEWAY_ATTACH_KEYCLOAK_INTROSPECTION_URL": (
        "http://keycloak:8080/realms/omninode/protocol/openid-connect/token/introspect"
    ),
    "GATEWAY_ATTACH_KEYCLOAK_JWKS_URL": (
        "http://keycloak:8080/realms/omninode/protocol/openid-connect/certs"
    ),
    "OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN": _PG_DSN,
    "OMNIBASE_INFRA_SKILL_LIFECYCLE_POSTGRES_DSN": _PG_DSN,
    "OMNIBASE_INFRA_CONTEXT_AUDIT_POSTGRES_DSN": _PG_DSN,
    "KAFKA_BOOTSTRAP_SERVERS": "localhost:19092",  # kafka-fallback-ok — test fixture
    "ARCH_GRAPH_BOLT_URI": "bolt://omnibase-infra-memgraph:7687",
    # OMN-18012: the dev-lane overlay's redpanda-scram-user service takes both
    # of these in the fail-closed ${VAR:?} form, so the LAYERED render aborts
    # without them -- which is how CI caught their absence here. Render-only
    # synthetic values; the real synthetic principal lives in the operator env
    # file on the lane host and never appears in this repo.
    "DEV_KAFKA_SASL_USERNAME": "render-only",
    "DEV_KAFKA_SASL_PASSWORD": "render-only",
    "ONEX_REGISTRATION_AUTO_ACK": "true",
    "ONEX_SERVICE_CLIENT_SECRET": "test-service-secret",
    # OMN-16843: x-runtime-env builds OMNINODE_INTERNAL_DB_URL from this with
    # the fail-closed ${VAR:?} form, so the layered render aborts without it.
    # Render-only, never a real credential.
    "OMNINODE_RUNTIME_PASSWORD": "render-only-omninode-runtime-password",
    # OMN-15425: TENANT-domain counterpart, same `:?` seam in x-runtime-env.
    "TENANT_PROJECTION_WRITER_PASSWORD": "render-only-tenant-projection-writer-password",
    "LINEAR_API_KEY": "test-linear-api-key",
    "GITHUB_TOKEN": "test-github-token",
    "DEPLOY_AGENT_HMAC_SECRET": "render-only-deploy-agent-hmac-secret",
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
    # `:?`-required by docker-compose.dev-lane.yml (OMN-15363), not by the base
    # file. Supplied here so the overlay-layered renders below can run; harmless
    # to the base-only renders, which never read it.
    "ROLE_OMNIDASH_PASSWORD": "test",
    # OMN-17530: the tenant-scoped control plane in the same overlay. Every one
    # of these takes the fail-closed `${VAR:?}` form there, so the LAYERED
    # render aborts without them -- which is the point of that form and is how
    # this fixture learns about a new one. Render-only synthetic values; the
    # lane's real lane-local sentinels are generated on the lane host by
    # scripts/runtime_build/render_dev_lane_tenant_path_env.sh and never appear
    # in this repo.
    "ONEX_API_IMAGE": "onex-api:render-only",
    # A path under the repo, not under the system temp directory: `docker
    # compose config` only has to INTERPOLATE this, never open it, and a
    # temp-directory literal here is a real lint finding rather than a false
    # positive -- a world-writable path is the wrong shape for a variable whose
    # production value holds a lane credential.
    "ONEX_LAB_TENANT_STATE_DIR": str(REPO_ROOT / ".render-only-tenant-state"),
    "ONEX_CLOUD_MIGRATE_IMAGE": "omninode-cloud-migrate:render-only",
    "ROLE_OMNINODE_PASSWORD": _RENDER_ONLY,
    "KEYCLOAK_ADMIN_CLIENT_SECRET": _RENDER_ONLY,
    "TENANT_BOOTSTRAP_ADMIN_SECRET": _RENDER_ONLY,
    "TENANT_TOPICS_ADMIN_SECRET": _RENDER_ONLY,
    "TENANT_CLIENTS_ADMIN_SECRET": _RENDER_ONLY,
    "TENANT_OFFBOARD_ADMIN_SECRET": _RENDER_ONLY,
    "ALPHA_INVITE_ADMIN_SECRET": _RENDER_ONLY,
    "STRIPE_API_KEY": _RENDER_ONLY,
    "STRIPE_WEBHOOK_SECRET": _RENDER_ONLY,
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


def _run_compose_config(
    env: dict[str, str],
    *,
    policy_env_file: str = _DEFAULT_POLICY_ENV_FILE,
    profile: str = "",
    with_dev_lane_overlay: bool = False,
    borrower_overlay: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    # NOTE: the default arm keeps the literal "--env-file",
    # "docker/runtime-policy.env" pair on the command line, because
    # tests/ci/test_compose_required_env_coverage.py discovers this fixture's
    # env-file coverage by regex over that literal pair. Do not collapse the two
    # arms into a single interpolated path.
    command = ["docker", "compose"]
    if policy_env_file == _DEFAULT_POLICY_ENV_FILE:
        command += [
            "--env-file",
            "docker/runtime-policy.env",
        ]
    else:
        command += ["--env-file", policy_env_file]
    command += [
        "-f",
        str(COMPOSE_FILE),
    ]
    if with_dev_lane_overlay:
        command += ["-f", str(DEV_LANE_OVERLAY)]
    if borrower_overlay is not None:
        # A lane that layers its own overlay THIRD, over this base and the
        # dev-lane overlay, as the deploy agent renders it.
        command += ["-f", str(borrower_overlay)]
    if profile:
        command += ["--profile", profile]
    command.append("config")
    return subprocess.run(
        command,
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


@pytest.mark.integration
def test_dev_lane_renders_one_runtime_worker_replica() -> None:
    """OMN-14968: the dev lane must render `runtime-worker` with replicas == 1.

    The value is the ledgered policy contract's `DEV_WORKER_REPLICAS`, supplied
    by `docker/runtime-policy.env`. A render of 0 reproduces the defect: compose
    creates no container, `up` exits 0 with no output, and the RT-6 deploy
    readback in `scripts/deploy-runtime.sh` then fails closed on an in-scope
    service it can never resolve.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime")

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    rendered = yaml.safe_load(result.stdout)
    worker = rendered["services"]["runtime-worker"]
    assert worker["deploy"]["replicas"] == 1, (
        "dev-lane runtime-worker must render deploy.replicas == 1 (the ledgered "
        f"DEV_WORKER_REPLICAS); got {worker['deploy']['replicas']!r}"
    )


@pytest.mark.integration
def test_dev_lane_delegation_routing_tiers_path_binding() -> None:
    """OMN-15645: DELEGATION_ROUTING_TIERS_PATH must be bound on every runtime
    service in the dev lane, to a fixed, non-version-embedded in-image path.

    omnimarket#2000 (OMN-15628) removed the packaged-default fallback for this
    key in the delegation routing reducer's ``_get_config()`` singleton
    (``resolve_required_path_config("DELEGATION_ROUTING_TIERS_PATH")`` —
    omnimarket ``handler_delegation_routing.py:392-393``); an unbound key now
    raises ``ProtocolConfigurationError`` at first config read instead of
    silently defaulting. The bound value must never be a literal
    ``python3.X`` site-packages path (a base-image Python version bump would
    silently invalidate it) — ``docker/Dockerfile.runtime`` bakes the packaged
    omnimarket ``routing_tiers.yaml`` into this exact fixed location at build
    time via a glob-derived COPY, so the compose-declared value here is always
    backed by a real file regardless of the interpreter minor version.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime")

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    rendered = yaml.safe_load(result.stdout)
    services = rendered["services"]

    expected_path = "/app/config/delegation/routing_tiers.yaml"
    for service_name in ("omninode-runtime", "runtime-effects", "runtime-worker"):
        environment = services[service_name]["environment"]
        assert environment.get("DELEGATION_ROUTING_TIERS_PATH") == expected_path, (
            f"Service '{service_name}' must bind DELEGATION_ROUTING_TIERS_PATH="
            f"{expected_path!r}; got "
            f"{environment.get('DELEGATION_ROUTING_TIERS_PATH')!r}"
        )
        assert "python3." not in environment.get("DELEGATION_ROUTING_TIERS_PATH", ""), (
            f"Service '{service_name}' binds a version-embedded python3.X literal "
            "for DELEGATION_ROUTING_TIERS_PATH — the exact trap OMN-15628's "
            "runtime self-heal exists to correct for a *stale* pin; the compose "
            "default must be a stable, version-independent path instead."
        )

    # Services with no delegation-routing surface deliberately opt out (mirrors
    # the BIFROST_CONTRACT_PATH opt-out pattern for the same two services).
    for service_name in ("projection-api", "omninode-contract-resolver"):
        environment = services[service_name]["environment"]
        assert environment.get("DELEGATION_ROUTING_TIERS_PATH", "") == "", (
            f"Service '{service_name}' deliberately has no delegation-routing "
            "surface and must not bind DELEGATION_ROUTING_TIERS_PATH; got "
            f"{environment.get('DELEGATION_ROUTING_TIERS_PATH')!r}"
        )


@pytest.mark.integration
def test_dev_worker_replicas_fails_closed_when_policy_value_unset(
    tmp_path: Path,
) -> None:
    """OMN-14968 counter-test: an unset DEV_WORKER_REPLICAS must FAIL the render.

    This is the RED half of the fix. The old bare `${WORKER_REPLICAS:-0}` had no
    exporter anywhere in the repo, so it always took the silent `0` branch and
    the lane lost its worker with zero signal. The lane-prefixed `:?` form must
    abort the render instead — never fall back to a replica count.
    """
    policy_without_worker_replicas = tmp_path / "runtime-policy-no-worker.env"
    policy_without_worker_replicas.write_text(
        "\n".join(
            line
            for line in POLICY_ENV_PATH.read_text(encoding="utf-8").splitlines()
            if not line.startswith("DEV_WORKER_REPLICAS=")
        )
        + "\n",
        encoding="utf-8",
    )
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)
    assert "DEV_WORKER_REPLICAS" not in env

    result = _run_compose_config(
        env,
        policy_env_file=str(policy_without_worker_replicas),
        profile="runtime",
    )

    assert result.returncode != 0, (
        "docker compose config unexpectedly succeeded with DEV_WORKER_REPLICAS "
        "unset — the silent-zero hole is back:\n" + result.stdout
    )
    assert "DEV_WORKER_REPLICAS" in result.stderr
    assert "replicas: 0" not in result.stdout


# =============================================================================
# OMN-17448 — standalone projection writers exist on the dev lane, and ONLY there
# =============================================================================
#
# The defect these assertions close: every `*ProjectionRunner` node on the .201
# compose dev lane was a no-op. The shared kernel subscribes their topics and
# its dispatch callback returns `None` before any handler runs (deliberate,
# OMN-15905 / OMN-16874 — a runner owns its own pool and its own consume loop,
# so the sanctioned way to run it is a dedicated process). OMN-15905 shipped
# that dedicated process for onex-dev k8s as five writer Deployments; nothing
# mirrored it onto compose, so `.201` ran ZERO standalone writers and every
# such projection consumed to LAG 0 and wrote nothing, silently.
#
# Measured live 2026-09-01: a well-formed TENANT_CREATED at offset 37 on
# `onex.tenant.events` advanced the consumer group to LAG 0 and left
# `tenant_registry_mirror` at 0 rows, with HWM 0 on both the DLQ and the
# terminal-event topic.

# OMN-17562 widened this from the two beta-critical writers OMN-17448 landed to
# the full ADOPT set: the six projections that have a checked-in onex-dev writer
# Deployment on `omninode_infra` origin/dev and therefore a proven runner to
# mirror. `omninode_infra#1147` ("revert(OMN-17519): remove rejected projection
# writer rollout", merged 2026-09-02T19:28:06Z) removed the
# pattern-learning and routing-decision Deployments as a prohibited rollout
# direction, so those two are deliberately NOT here — they are OMN-17557 /
# OMN-17556 store-resolved-credential work, not writer-mirroring work.
#
# Service name -> the runner module its `__main__` block starts. One mapping,
# not a tuple beside a dict: a writer whose name is asserted but whose module is
# not is exactly the half-checked service this file exists to prevent.
_WRITER_MODULES: dict[str, str] = {
    "projection-tenant-registry-writer": (
        "omnimarket.nodes.node_projection_tenant_registry.handlers"
        ".handler_tenant_registry_projection"
    ),
    "projection-delegation-writer": (
        "omnimarket.nodes.node_projection_delegation.handlers.handler_delegation"
    ),
    "projection-registration-writer": (
        "omnimarket.nodes.node_projection_registration.handlers.handler_registration"
    ),
    "projection-savings-writer": (
        "omnimarket.nodes.node_projection_savings.handlers.handler_savings"
    ),
    "projection-tenant-credentials-writer": (
        "omnimarket.nodes.node_projection_tenant_credentials.handlers"
        ".handler_tenant_credentials_projection"
    ),
    "projection-live-events-writer": (
        "omnimarket.nodes.node_projection_live_events.handlers.handler_live_events"
    ),
}
_WRITER_SERVICES: tuple[str, ...] = tuple(_WRITER_MODULES)


@pytest.mark.integration
def test_dev_lane_renders_the_standalone_projection_writers() -> None:
    """OMN-17448 AC2: the dev lane has a real write path for these two nodes."""
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]
    for name in _WRITER_SERVICES:
        assert name in services, (
            f"dev lane must declare {name!r}: without it the shared kernel "
            "subscribes this projection's topics, commits every offset, and "
            "writes nothing (OMN-17448)"
        )
    for name in (
        "projection-delegation-writer",
        "projection-savings-writer",
        "projection-tenant-credentials-writer",
    ):
        writer_env = services[name]["environment"]
        assert writer_env["ONEX_DATABASE_TOPOLOGY_PROFILE"] == "local", name
        assert writer_env["ONEX_TENANT_DB_URL"].startswith(
            "postgresql://tenant_projection_writer:"
        ), name
        assert writer_env["OMNINODE_INTERNAL_DB_URL"].startswith(
            "postgresql://omninode_runtime:"
        ), name


@pytest.mark.integration
def test_dev_lane_renders_the_tenant_projection_carrier() -> None:
    """OMN-18114: the dev lane STARTS a process for the `tenant-projection` profile.

    Eight omnimarket contracts declare ``runtime_profiles: [tenant-projection]``
    and are therefore dropped from ``main`` and ``effects`` by
    ``filter_manifest_for_runtime_profile``. Before this service existed, no
    process on any compose lane bound that profile, so all eight were discovered
    and subscribed by nothing -- measured on the .201 dev lane 2026-09-10 as
    eight Empty consumer groups, with
    ``node_projection_delegation_inference_response`` at LAG 196 and rising.

    Rendering under ``--profile runtime`` is the assertion that matters: the
    service is declared in the base under a compose profile no lane requests, so
    it appears here only because this lane's overlay opted it in.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]
    assert "tenant-projection-writer" in services, (
        "the dev lane must START the tenant-projection carrier: without it the "
        "eight contracts pinned to that profile are discovered and consumed by "
        "nothing, with no error on any process (OMN-18114)"
    )

    carrier = services["tenant-projection-writer"]
    assert carrier["environment"]["RUNTIME_PROFILE"] == "tenant-projection"

    # It must be the KERNEL, not a runner. A `command:` override would start
    # something else entirely and reproduce the defect while looking fixed.
    assert "command" not in carrier or not carrier["command"], (
        "the carrier is the runtime kernel under a different RUNTIME_PROFILE, "
        "with no command override -- that is what gives it the real "
        "topology-resolved projection arm a BaseProjectionRunner does not get"
    )

    # Distinct instance id, or its consumer groups collide with main's. The
    # group name embeds KAFKA_INSTANCE_ID as `...__i.<instance>...`.
    shared_instance_ids = {
        services[name]["environment"]["KAFKA_INSTANCE_ID"]
        for name in ("omninode-runtime", "runtime-effects", "runtime-worker")
    }
    assert carrier["environment"]["KAFKA_INSTANCE_ID"] not in shared_instance_ids, (
        "the carrier must not share a KAFKA_INSTANCE_ID with a shared kernel; "
        "the consumer group name embeds it, so a collision would put two "
        "processes in one group"
    )


@pytest.mark.integration
def test_writers_invoke_the_runner_module_entrypoint() -> None:
    """The command must be the handler module's own ``__main__``.

    This is the whole point of a standalone writer: it runs the runner class
    OUTSIDE the kernel. A command that started the kernel instead would
    reproduce the defect exactly — the process would come up healthy, join the
    group, and dispatch nothing.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]
    for name, module in _WRITER_MODULES.items():
        command = services[name]["command"]
        assert command[:3] == ["python", "-m", module], (
            f"{name} must run {module} as a module entrypoint; got {command!r}"
        )


@pytest.mark.integration
def test_each_writer_holds_its_own_consumer_group() -> None:
    """Two writers sharing a group would split partitions and lose half the rows.

    A shared group is worse than no writer at all: the topic's partitions would
    be divided between two processes that project DIFFERENT relations, so each
    would silently drop whatever the other was assigned — and it would look like
    it was working.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]
    groups = [
        services[name]["environment"]["KAFKA_CONSUMER_GROUP"]
        for name in _WRITER_SERVICES
    ]
    assert len(set(groups)) == len(groups), (
        f"each standalone writer needs its own consumer group; got {groups!r}"
    )
    assert all(g for g in groups), (
        "an unset KAFKA_CONSUMER_GROUP falls back to BaseProjectionRunner's "
        "DEFAULT_GROUP_ID, which every writer would then share"
    )


@pytest.mark.integration
def test_each_writer_healthcheck_probes_its_own_readiness_port() -> None:
    """A healthcheck pointing at a sibling's port is an autoheal restart loop.

    ``BaseProjectionRunner`` serves readiness on ``PROJECTION_RUNNER_HEALTH_PORT``
    and nothing else listens inside the container, so a copy-pasted healthcheck
    URL that kept the previous writer's port would curl a closed port forever.
    These services carry ``autoheal=true``, so that does not read as one
    unhealthy container — it restarts the process every 30s indefinitely, and
    the writer that looks deployed writes nothing between restarts. The exact
    silent-loss shape this whole writer set exists to end.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]
    for name in _WRITER_SERVICES:
        port = services[name]["environment"]["PROJECTION_RUNNER_HEALTH_PORT"]
        probe = " ".join(str(part) for part in services[name]["healthcheck"]["test"])
        assert f"localhost:{port}/ready" in probe, (
            f"{name} serves readiness on port {port!r} but its healthcheck probes "
            f"{probe!r}. A mismatched port is permanently unhealthy, and with "
            "autoheal=true that is a restart loop, not a visible failure."
        )


@pytest.mark.integration
def test_writers_are_absent_from_the_base_file_every_other_lane_merges() -> None:
    """Fail-closed containment: prod and judge must not inherit these.

    The base file is merged by EVERY lane; only the dev lane layers this
    overlay (``resolve_compose_file_args()``). Declaring the writers in the base
    would add them to prod and to any lane created later by someone who has
    never read this file — the same fail-open shape the migration-lane
    indicator at the top of the overlay exists to prevent.

    OMN-17562 gave the stability-test lane the same six writers, and that is
    exactly why this stays a base-file check rather than becoming a
    "nowhere but dev" one: the proof lane declares them EXPLICITLY in
    ``docker-compose.stability-test.yml``, with its own container names, its own
    lane-scoped consumer groups and its own DSN spelling. Inheriting them
    silently from the base would have given it the dev lane's identities
    instead, which is the failure this assertion is shaped against.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=False)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]
    for name in _WRITER_SERVICES:
        assert name not in services, (
            f"{name!r} leaked into docker-compose.infra.yml — every non-dev "
            "lane merges that file and would inherit this service"
        )


# =============================================================================
# OMN-17562 ruling item (4) — the dev lane's runtime probe becomes semantic and
# autoheal is disarmed in the same change
# =============================================================================
#
# The base compose probe is ``curl -sf http://localhost:8085/health``, which
# asserts exactly one property: HTTP status < 400. ``/health`` returns 200 for a
# running-but-DEGRADED runtime BY DESIGN (a degraded container stays in rotation
# rather than triggering cascading restarts), so the base probe is a liveness
# check wearing a health check's name. OMN-15217 replaced it on the stability
# lane after the mask was read live there — ``Up 4 hours (healthy)`` on all three
# runtime containers while their own monitors logged ``status=DEGRADED``.
#
# The dev lane kept the shallow probe, and that is precisely how it carried the
# OMN-17448 silent-loss defect green: `tests/ci/test_lane_projection_writer_
# coverage_omn17562.py` names the shallow probe as the reason the defect was
# only ever caught on the lane that had already been made honest. This block is
# the other half — the dev lane now runs the same strict check.
#
# The two changes are ONE change. ``autoheal`` watches Docker health, and
# semantic degradation is typically restart-immune (four contracts that fail to
# import will fail to import again). Flipping the probe while leaving
# ``autoheal=true`` armed would convert an honest unhealthy signal into a
# restart of all three runtime containers every 30s. Compose APPENDS label
# sequences, so the base service's ``autoheal=true`` survives a plain ``labels:``
# block — only ``labels: !override`` disarms it, and that is a merge behaviour
# the overlay file alone cannot prove, which is why these read the RENDER.
_STRICT_PROBE: list[str] = [
    "CMD",
    "python",
    "/usr/local/bin/onex-container-healthcheck",
    "--degraded-policy",
    "fail",
]
_SHALLOW_PROBE: list[str] = ["CMD", "curl", "-sf", "http://localhost:8085/health"]
_KERNEL_RUNTIME_SERVICES: tuple[str, ...] = (
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
)


def _label_value(service_config: dict[str, Any], key: str) -> str | None:
    """Read one label off a rendered service, whichever shape compose emits."""
    labels = service_config.get("labels", {})
    if isinstance(labels, dict):
        value = labels.get(key)
        return str(value) if value is not None else None
    if isinstance(labels, list):
        prefix = f"{key}="
        for label in labels:
            if isinstance(label, str) and label.startswith(prefix):
                return label.removeprefix(prefix)
    return None


@pytest.mark.integration
def test_dev_lane_runtime_probe_is_the_strict_semantic_check() -> None:
    """OMN-17562(4): the rendered dev lane runs the semantic probe, not curl.

    Mirror of ``tests/unit/infra/test_stability_test_runtime_lane.py``'s
    ``test_stability_lane_runtime_healthchecks_are_semantic_not_shallow``, read
    off the RENDER rather than the overlay file: compose replaces ``healthcheck``
    wholesale, so a mis-authored override surfaces here as the inherited
    ``curl -sf`` probe rather than as a file diff.

    The flap budget is asserted against the BASE service's own resolved values
    rather than against literals, so a future change to the base window cannot
    leave this lane silently tighter than the process it is probing (the
    monitor's first verdict lands ~one ``RUNTIME_HEALTH_CHECK_INTERVAL`` (300s)
    after boot, and an absent verdict passes, so startup cannot flap on it).
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    lane = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)
    assert lane.returncode == 0, f"docker compose config failed:\n{lane.stderr}"
    base = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=False)
    assert base.returncode == 0, f"docker compose config failed:\n{base.stderr}"

    lane_services = yaml.safe_load(lane.stdout)["services"]
    base_services = yaml.safe_load(base.stdout)["services"]

    for service_name in _KERNEL_RUNTIME_SERVICES:
        healthcheck = lane_services[service_name]["healthcheck"]

        # Exact list, not a substring check: the probe is what Docker executes,
        # so a partial match would accept a shallow fallback appended beside it.
        assert healthcheck["test"] == _STRICT_PROBE, (
            f"{service_name}: the dev lane must run the strict semantic check; "
            "the shallow curl probe reports healthy for a runtime whose own "
            f"monitor says DEGRADED. Got {healthcheck['test']!r}"
        )
        assert "curl" not in healthcheck["test"], (
            f"{service_name}: shallow curl probe survived the strict override"
        )
        assert healthcheck["test"][-2:] == ["--degraded-policy", "fail"], (
            f"{service_name}: strict policy flag missing — without it the check "
            "degrades to the same pass-on-DEGRADED semantics as curl -sf"
        )

        base_healthcheck = base_services[service_name]["healthcheck"]
        for window_key in ("interval", "timeout", "retries", "start_period"):
            assert healthcheck[window_key] == base_healthcheck[window_key], (
                f"{service_name}: strict probe must keep the base service's "
                f"{window_key} budget ({base_healthcheck[window_key]!r}); got "
                f"{healthcheck[window_key]!r}. A tighter window flaps on the "
                "boot interval where no verdict has been published yet."
            )


@pytest.mark.integration
def test_dev_lane_runtime_services_do_not_carry_autoheal() -> None:
    """OMN-17562(4): strict health and armed autoheal must never coexist here.

    ``labels`` are APPENDED by compose, so the base service's ``autoheal=true``
    survives a plain ``labels:`` block and only ``labels: !override`` removes
    it. The overlay file cannot prove that on its own — both spellings parse
    identically — so this reads the resolved render.

    With the strict probe above, "unhealthy" now means "semantically degraded",
    and semantic degradation is usually restart-immune. An armed autoheal would
    therefore restart all three dev runtime containers every 30s forever and
    destroy the forensic state, instead of surfacing one honest unhealthy
    container.

    The identity labels are asserted too: ``!override`` replaces the whole
    sequence, so an override that forgot to restate them would silently strip
    the service/layer identity the lane census and every ``docker ps`` filter
    read.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]

    expected_service_label = {
        "omninode-runtime": "runtime-main",
        "runtime-effects": "runtime-effects",
        "runtime-worker": "runtime-worker",
    }
    for service_name in _KERNEL_RUNTIME_SERVICES:
        service = services[service_name]

        assert _label_value(service, "autoheal") is None, (
            f"{service_name}: autoheal survived into the rendered dev lane — "
            "compose appends label sequences, so `labels:` must be "
            "`labels: !override`. Strict health plus armed autoheal restart-"
            "loops a restart-immune defect every 30s."
        )
        assert (
            _label_value(service, "com.omninode.service")
            == (expected_service_label[service_name])
        ), (
            f"{service_name}: the `!override` label block dropped the service "
            "identity label the lane census reads"
        )
        assert _label_value(service, "com.omninode.layer") == "runtime", (
            f"{service_name}: the `!override` label block dropped the layer label"
        )


@pytest.mark.integration
def test_the_base_file_probe_and_autoheal_are_unchanged_for_every_other_lane() -> None:
    """RED-guard: prod, judge and lakshman are provably untouched by this change.

    This is why the strict probe lands in the dev-lane OVERLAY and not at
    ``docker-compose.infra.yml`` lines ~992/1067/1235, which is where ruling
    item (4) literally pointed. ``docker-compose.prod.yml`` declares NO
    healthcheck override for these three services and ``autoheal=true`` is a
    base-file label, so editing the base would flip the PROD probe to the strict
    semantic check and disarm prod's autoheal — two live-blast-radius changes
    this ticket has no mandate over, made silently as a side effect.

    Rendering the base with no overlay is exactly what those lanes inherit, so
    an edit that leaked into the base fails here.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=False)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]

    for service_name in _KERNEL_RUNTIME_SERVICES:
        service = services[service_name]
        assert service["healthcheck"]["test"] == _SHALLOW_PROBE, (
            f"{service_name}: the base probe changed. Every non-dev lane merges "
            "docker-compose.infra.yml, and prod declares no healthcheck override "
            "for this service — so this edit moved PROD's probe. Put the strict "
            "check in docker/docker-compose.dev-lane.yml instead."
        )
        assert _label_value(service, "autoheal") == "true", (
            f"{service_name}: autoheal=true was removed from the base file. "
            "prod inherits that label; dropping it here disarms prod's "
            "self-recovery. Disarm it in the dev-lane overlay instead."
        )


# ---------------------------------------------------------------------------
# OMN-18789: the broker's own probe reads a partition, not `leaderless_count`
# ---------------------------------------------------------------------------

#: The broker healthcheck this change replaces, verbatim from the base file.
_LEADERLESS_BROKER_PROBE: list[str] = [
    "CMD-SHELL",
    "rpk cluster health | grep -q 'Healthy:.*true' || exit 1",
]

#: What the dev lane runs instead.
_PARTITION_READ_BROKER_PROBE: list[str] = [
    "CMD",
    "/usr/bin/bash",
    "/usr/local/bin/onex-broker-readiness-probe",
]


@pytest.mark.integration
def test_dev_lane_broker_probe_reads_a_partition_not_leaderless_count() -> None:
    """OMN-18789 AC3/AC4: the rendered dev lane runs the new probe.

    The static half of this claim is in
    tests/unit/docker/test_omn18789_broker_readiness_lane_scope.py. This is the
    half that proves compose actually MERGES the override into the lane -- a
    service key in an overlay that compose silently drops would satisfy the
    YAML assertion and change nothing on the host.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    redpanda = yaml.safe_load(result.stdout)["services"]["redpanda"]

    assert redpanda["healthcheck"]["test"] == _PARTITION_READ_BROKER_PROBE, (
        "the dev lane's broker is back on an admin-API liveness check. That "
        "surface read `healthy` through a 97-minute total data-plane outage on "
        "2026-09-18 (5921 not_leader_for_partition errors, "
        "`leaderless_count: 0` throughout)."
    )

    mount_targets = {
        entry["target"]
        for entry in redpanda["volumes"]
        if isinstance(entry, dict) and "target" in entry
    }
    assert "/usr/local/bin/onex-broker-readiness-probe" in mount_targets
    assert "/etc/onex/broker_readiness_declaration.conf" in mount_targets, (
        "the probe fails closed without its declaration (Operating Rule 8: no "
        "environment fallback for the window), so an unmounted declaration is "
        "a lane that never reports healthy"
    )
    assert "redpanda_data" in {
        entry.get("source") for entry in redpanda["volumes"] if isinstance(entry, dict)
    }, "the override appended its mounts instead of replacing the data volume"


@pytest.mark.integration
def test_the_base_broker_probe_is_unchanged_for_every_other_lane() -> None:
    """OMN-18789 AC4 RED-guard: stability-test and prod are provably untouched.

    Both merge `docker-compose.infra.yml` and declare no broker healthcheck of
    their own, so the base render IS their probe. Moving the check here rather
    than in the overlay would have changed the STABILITY lane -- the surface
    the compose path's `stability-proven` premise is resolved from.
    """
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)

    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=False)

    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    redpanda = yaml.safe_load(result.stdout)["services"]["redpanda"]

    assert redpanda["healthcheck"]["test"] == _LEADERLESS_BROKER_PROBE, (
        "the BASE broker probe changed. stability-test and prod inherit it and "
        "declare no override, so this edit moved their probe too. Put the "
        "change in docker/docker-compose.dev-lane.yml instead."
    )
    for entry in redpanda["volumes"]:
        target = entry.get("target") if isinstance(entry, dict) else str(entry)
        assert "onex-broker-readiness-probe" not in str(target), (
            "the OMN-18789 probe leaked into the base every other lane merges"
        )


# OMN-20159 (Amendment 4 step 3). node_projection_read_effect runs in
# runtime-effects and reads through its OWN binding variable (omnimarket#3364).
# The runtime binding variable stays off runtime-effects: it also selects
# node_delegate_skill_orchestrator's claim store and evidence store, and with
# role_omnidash as their login every bus delegation failed with UndefinedTable
# on delegate_skill_command_claims (OMN-17427, chain-canary 37043007992).
_RUNTIME_BINDING_ENV = "OMNIMARKET_PROJECTION_RUNTIME_BINDING_OVERLAY"
_RUNTIME_BINDING_TARGET = "/etc/onex/projection-runtime-binding.yaml"
_READ_BINDING_ENV = "OMNIMARKET_PROJECTION_READ_BINDING_OVERLAY"
_READ_BINDING_TARGET = "/etc/onex/projection-read-binding.yaml"
_READ_BINDING_FILE = (
    REPO_ROOT / "docker" / "projection-runtime-binding" / "runtime-read.yaml"
)
_READ_BINDING_GROUP = "local.omnimarket-projections.runtime-read.consume.v1"

# Every lane that layers docker-compose.dev-lane.yml under its own overlay, with
# the profile its runtime-effects renders under. Named, not discovered:
# test_every_lane_layering_the_dev_overlay_is_listed fails when a new overlay
# claims the dev-lane layering and is missing here.
_BORROWER_OVERLAYS: dict[str, str] = {
    "docker-compose.dev-105.yml": "runtime",
    "docker-compose.dev-200.yml": "runtime",
    "docker-compose.dev-202.yml": "runtime",
    "docker-compose.prepr.yml": "prepr",
}
# The pre-PR slot overlay's own `:?` inputs. Render-only stand-ins: the real
# values come from the slot allocator on the lane host.
_PREPR_SLOT_RENDER_ENV: dict[str, str] = {
    "KAFKA_ENVIRONMENT": "prepr-render-only",
    "KAFKA_TOPIC_NAMESPACE": "prepr-render-only",
    "ONEX_DB_SLOT": "render-only",
    "ONEX_PREPR_SLOT": "render-only",
    "ONEX_PREPR_TENANT_STATE_DIR": str(REPO_ROOT / ".render-only-prepr-state"),
    "PREPR_GATEWAY_PORT": "18001",
    "PREPR_PROJECTION_API_PORT": "18002",
    "PREPR_RUNTIME_EFFECTS_PORT": "18003",
    "PREPR_RUNTIME_MAIN_PORT": "18004",
    "PREPR_VALKEY_DB_INDEX": "9",
    "ROLE_OMNIBASE_PASSWORD": _RENDER_ONLY,
    "ROLE_OMNIINTELLIGENCE_PASSWORD": _RENDER_ONLY,
    "ROLE_OMNIMEMORY_PASSWORD": _RENDER_ONLY,
}


def _mounts_at(service: dict[str, Any], target: str) -> list[dict[str, Any]]:
    return [
        mount
        for mount in service.get("volumes", [])
        if isinstance(mount, dict) and mount.get("target") == target
    ]


def _binding_mount_source(service: dict[str, Any]) -> Path | None:
    for mount in _mounts_at(service, _RUNTIME_BINDING_TARGET):
        return Path(mount["source"])
    return None


def _render_dev_lane_services() -> dict[str, Any]:
    env = _render_env(DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST)
    result = _run_compose_config(env, profile="runtime", with_dev_lane_overlay=True)
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    services = yaml.safe_load(result.stdout)["services"]
    assert isinstance(services, dict)
    return services


@pytest.mark.integration
def test_dev_lane_runtime_effects_carries_the_projection_read_binding() -> None:
    """With no read binding the /skill edge refuses every read
    ``projection_binding_unconfigured``. The variable must name exactly the
    mount target, or omnimarket raises ProjectionReadBindingOverlayError and
    every read answers ``projection_binding_invalid``. The mount is read-only,
    so the container cannot rewrite its own database identity.
    """
    effects = _render_dev_lane_services()["runtime-effects"]
    assert effects["environment"].get(_READ_BINDING_ENV) == _READ_BINDING_TARGET
    mounts = _mounts_at(effects, _READ_BINDING_TARGET)
    assert len(mounts) == 1, f"expected one read-binding mount, got {mounts!r}"
    (mount,) = mounts
    assert mount.get("type") == "bind"
    assert mount.get("read_only") is True
    assert Path(mount["source"]).resolve() == _READ_BINDING_FILE.resolve()


@pytest.mark.integration
def test_dev_lane_runtime_effects_carries_no_runtime_binding() -> None:
    """OMN-17427, kept from omnibase_infra#4481: the runtime binding selects the
    delegation claim store, and role_omnidash lacks USAGE on omninode_internal
    and cannot write delegate_skill_command_claims, so Postgres skips the pinned
    search_path schema and every delegation fails with UndefinedTable.
    #4481 said to change this test once the claim store resolves a write
    principal apart from the /skill read binding; omnimarket#3364 did that with
    the read variable above. The runtime variable stays off runtime-effects, so
    the claim store and evidence store keep their container-local SQLite files.
    """
    services = _render_dev_lane_services()
    effects = services["runtime-effects"]
    assert _RUNTIME_BINDING_ENV not in effects["environment"]
    assert _mounts_at(effects, _RUNTIME_BINDING_TARGET) == []

    # Positive control: the helper finds a writer's runtime binding at the target.
    assert any(
        _binding_mount_source(service) is not None
        for name, service in services.items()
        if name != "runtime-effects"
    ), "no other dev-lane binding found; the helper check proves nothing"


@pytest.mark.integration
def test_dev_lane_read_binding_reads_as_the_projection_api_does() -> None:
    """The edge and :3002 must answer the same rows, so the read binding names
    OMNIDASH_ANALYTICS_DB_URL by reference, and runtime-effects renders that
    variable exactly as projection-api does (role_omnidash, non-BYPASSRLS,
    OMN-15363). A URL in the file would be a committed credential.
    """
    services = _render_dev_lane_services()
    binding = yaml.safe_load(_READ_BINDING_FILE.read_text(encoding="utf-8"))
    assert binding["database_url_secret_ref"] == "env:OMNIDASH_ANALYTICS_DB_URL"
    assert "://" not in _READ_BINDING_FILE.read_text(encoding="utf-8")

    effects_dsn = services["runtime-effects"]["environment"][
        "OMNIDASH_ANALYTICS_DB_URL"
    ]
    api_dsn = services["projection-api"]["environment"]["OMNIDASH_ANALYTICS_DB_URL"]
    assert effects_dsn == api_dsn
    assert urlsplit(effects_dsn).username == "role_omnidash"


@pytest.mark.integration
def test_dev_lane_read_binding_consumer_group_is_its_own() -> None:
    """A group shared with a writer would split that writer's partitions."""
    services = _render_dev_lane_services()
    binding = yaml.safe_load(_READ_BINDING_FILE.read_text(encoding="utf-8"))
    assert binding["kafka_consumer_group"] == _READ_BINDING_GROUP

    # Every group another dev-lane service consumes under: the ones its own
    # binding file declares and the ones its environment declares (each
    # standalone writer sets KAFKA_CONSUMER_GROUP; runtimes set ONEX_GROUP_ID).
    binding_groups = {
        yaml.safe_load(other.read_text(encoding="utf-8"))["kafka_consumer_group"]
        for name, service in services.items()
        if name != "runtime-effects"
        and (other := _binding_mount_source(service)) is not None
    }
    env_groups = {
        str(value)
        for name, service in services.items()
        if name != "runtime-effects"
        for key, value in (service.get("environment") or {}).items()
        if key in {"KAFKA_CONSUMER_GROUP", "ONEX_GROUP_ID"} and value
    }
    assert binding_groups, (
        "no other dev-lane binding found; the comparison proves nothing"
    )
    assert "local.omnimarket-projections.delegation-writer.consume.v1" in env_groups, (
        "the delegation writer's own group is missing; the env comparison proves nothing"
    )
    assert _READ_BINDING_GROUP not in binding_groups | env_groups


@pytest.mark.integration
def test_only_runtime_effects_carries_the_read_binding() -> None:
    """The read node runs in runtime-effects. Runtime main and every writer keep
    their shape, and main carries no binding of either kind."""
    services = _render_dev_lane_services()
    carriers = {
        name
        for name, service in services.items()
        if _READ_BINDING_ENV in (service.get("environment") or {})
        or _mounts_at(service, _READ_BINDING_TARGET)
    }
    assert carriers == {"runtime-effects"}
    runtime = services["omninode-runtime"]
    assert _RUNTIME_BINDING_ENV not in runtime["environment"]
    assert _mounts_at(runtime, _RUNTIME_BINDING_TARGET) == []


@pytest.mark.integration
@pytest.mark.parametrize("overlay", sorted(_BORROWER_OVERLAYS))
def test_lanes_layering_the_dev_overlay_carry_no_read_binding(overlay: str) -> None:
    """These lanes merge runtime-effects' environment from the dev-lane overlay
    but replace its volumes (`volumes: !override`), so they would inherit the
    read variable without the file and every read there would turn from
    ``projection_binding_unconfigured`` into ``projection_binding_invalid``.
    Each blanks the variable, which omnimarket reads as unset, and mounts
    nothing at the target: no behaviour change on those lanes.
    """
    env = _render_env(
        DEV_REDPANDA_ADVERTISE_HOST=_OFF_HOST_ADVERTISE_HOST, **_PREPR_SLOT_RENDER_ENV
    )
    result = _run_compose_config(
        env,
        profile=_BORROWER_OVERLAYS[overlay],
        with_dev_lane_overlay=True,
        borrower_overlay=REPO_ROOT / "docker" / overlay,
    )
    assert result.returncode == 0, f"docker compose config failed:\n{result.stderr}"
    effects = yaml.safe_load(result.stdout)["services"]["runtime-effects"]

    # Positive controls: the render layered the dev-lane overlay (its broker
    # auth is declared nowhere in the base file) and then this lane's overlay.
    assert effects["environment"].get("KAFKA_SASL_MECHANISM") == "SCRAM-SHA-256"
    assert effects["container_name"] != "omninode-runtime-effects"

    assert str(effects["environment"].get(_READ_BINDING_ENV, "")).strip() == ""
    assert _mounts_at(effects, _READ_BINDING_TARGET) == []
    assert _RUNTIME_BINDING_ENV not in effects["environment"]


@pytest.mark.integration
def test_every_lane_layering_the_dev_overlay_is_listed() -> None:
    docker_dir = REPO_ROOT / "docker"
    layering = sorted(
        path.name
        for path in docker_dir.glob("docker-compose*.yml")
        if path != DEV_LANE_OVERLAY
        and "\n  runtime-effects:\n" in (text := path.read_text(encoding="utf-8"))
        and (
            "-f docker/docker-compose.dev-lane.yml" in text
            or "over docker-compose.infra.yml and docker-compose.dev-lane.yml" in text
        )
    )
    assert layering == sorted(_BORROWER_OVERLAYS)
