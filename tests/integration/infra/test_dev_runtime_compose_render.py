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
