# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Env parity test: docker-compose and the onex-dev k8s manifests agree, BOTH ways.

Forward direction (OMN-4307): every variable in the docker-compose
``x-runtime-env`` anchor is accounted for in the k8s ConfigMap, a known Secret,
or the LOCAL_ONLY_KEYS allowlist.

Reverse direction (OMN-15628): every configuration key the k8s manifests bind —
ConfigMap ``data`` keys plus literal ``value:`` entries in the runtime
Deployments — is bound somewhere in ``docker-compose.infra.yml``, or is
explicitly classified as cluster-only / tracked parity debt.

The reverse direction exists because the forward-only gate was structurally
blind to the failure that shipped in OMN-15628: ``DELEGATION_ROUTING_TIERS_PATH``
was bound on all three onex-dev runtime Deployments and on ZERO compose files,
so every local/lab/stability/prod container booted healthy and then fail-closed
on the first delegation-routing request (the omnimarket routing reducer resolves
it through ``resolve_required_path_config``, which raises rather than defaulting).
A compose→k8s-only walk can never see a key that is missing on the compose side.

Run with: uv run pytest tests/ci/test_env_parity.py

Tickets: OMN-4307 (forward), OMN-15628 (reverse)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# COMPOSE_PATH: always relative to this file inside omnibase_infra
_REPO_ROOT = Path(__file__).parent.parent.parent

COMPOSE_PATH = _REPO_ROOT / "docker" / "docker-compose.infra.yml"

# CONFIGMAP_PATH: omninode_infra may live as a sibling in several layouts:
#   1. Local worktrees: /Volumes/.../omni_worktrees/<ticket>/omnibase_infra/ →
#      sibling at ../omninode_infra/
#   2. omni_home monorepo: /Volumes/.../omni_home/omnibase_infra/ →
#      sibling at ../omninode_infra/
#   3. CI with dual checkout: both repos checked out side-by-side
#   4. OMNINODE_INFRA_DIR env var override
_K8S_RUNTIME_SUBPATH = "k8s/onex-dev/runtime"
_CONFIGMAP_SUBPATH = f"{_K8S_RUNTIME_SUBPATH}/configmap.yaml"


def _resolve_k8s_runtime_dir() -> Path | None:
    """Resolve the onex-dev k8s runtime manifest directory in omninode_infra.

    Same candidate chain the ConfigMap resolution has always used; hoisted to
    the directory so the reverse-direction check (OMN-15628) can read the
    Deployment manifests alongside the ConfigMap from one resolved root.
    """
    # Env var override takes precedence
    override = os.environ.get("OMNINODE_INFRA_DIR", "").strip()
    if override:
        candidate = Path(override) / _K8S_RUNTIME_SUBPATH
        if (candidate / "configmap.yaml").exists():
            return candidate

    # Try sibling directories relative to the repo root
    candidates: list[Path] = [
        # Direct sibling (CI dual-checkout, or a ticket dir holding both repos)
        _REPO_ROOT.parent / "omninode_infra" / _K8S_RUNTIME_SUBPATH,
        # Two levels up (omni_home registry layout: omni_home/omnibase_infra)
        _REPO_ROOT.parent.parent / "omninode_infra" / _K8S_RUNTIME_SUBPATH,
        # Three levels up: the standard worktree layout
        # omni_home/omni_worktrees/<ticket>/omnibase_infra, whose canonical
        # omninode_infra clone sits at the omni_home root. Without this the
        # gate SKIPS on every local worktree run (including the pre-push hook),
        # which is a vacuous green — the pre-push run for OMN-15628 skipped all
        # three parity assertions for exactly this reason.
        _REPO_ROOT.parent.parent.parent / "omninode_infra" / _K8S_RUNTIME_SUBPATH,
    ]
    for candidate in candidates:
        if (candidate / "configmap.yaml").exists():
            return candidate

    return None


K8S_RUNTIME_DIR = _resolve_k8s_runtime_dir()
CONFIGMAP_PATH = (
    (K8S_RUNTIME_DIR / "configmap.yaml") if K8S_RUNTIME_DIR is not None else None
)

# Runtime-family Deployments: the k8s workloads whose compose counterparts merge
# the shared ``x-runtime-env`` anchor. These are the manifests that bind
# DELEGATION_ROUTING_TIERS_PATH.
K8S_RUNTIME_FAMILY_DEPLOYMENTS: tuple[str, ...] = (
    "deployment-omninode-runtime.yaml",
    "deployment-omninode-runtime-effects.yaml",
    "deployment-omninode-runtime-worker.yaml",
)

# Compose services that merge the ``x-runtime-env`` anchor and correspond
# one-to-one with K8S_RUNTIME_FAMILY_DEPLOYMENTS.
COMPOSE_RUNTIME_FAMILY_SERVICES: tuple[str, ...] = (
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
)

# ---------------------------------------------------------------------------
# Key classification
# ---------------------------------------------------------------------------

# Keys sourced from k8s Secrets or Infisical (not ConfigMap) — expected absent from ConfigMap.
# In the k8s cluster these are injected via InfisicalSecret or explicit Secret volumes.
SECRET_KEYS: frozenset[str] = frozenset(
    {
        # Bootstrap / Infisical identity — injected via InfisicalSecret (onex-runtime-infisical-secret.yaml)
        "POSTGRES_PASSWORD",
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
        "INFISICAL_ENVIRONMENT",
        "INFISICAL_ENCRYPTION_KEY",
        "INFISICAL_AUTH_SECRET",
        # Per-service database DSNs — contain credentials, sourced from Infisical at runtime
        "OMNIBASE_INFRA_DB_URL",  # k8s uses OMNIBASE_INFRA_DB_HOST + OMNIBASE_INFRA_DB_PORT + Secret
        "OMNIINTELLIGENCE_DB_URL",  # cross-service DSN with embedded credentials
        "OMNIDASH_ANALYTICS_DB_URL",  # analytics DSN with embedded credentials, sourced from Infisical
        "OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN",
        "OMNIBASE_INFRA_SKILL_LIFECYCLE_POSTGRES_DSN",
        # Valkey auth
        "VALKEY_PASSWORD",  # injected via Infisical at runtime
        # Keycloak secrets
        "KEYCLOAK_ADMIN_CLIENT_SECRET",
        "ONEX_SERVICE_CLIENT_SECRET",
        # Linear API — injected via Infisical at runtime
        "LINEAR_API_KEY",
        # GitHub API/CLI auth — injected via Infisical at runtime
        "GITHUB_TOKEN",
        "GH_TOKEN",
        # Deploy control-plane command authentication — injected via Infisical at runtime
        "DEPLOY_AGENT_HMAC_SECRET",
        # Qdrant vector store API key — credential, sourced from Infisical
        "QDRANT_API_KEY",
        # Cloud-tier LLM route secret ref — credential, sourced from Infisical/k8s secret.
        "LLM_GLM_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    }
)

# Keys that are bootstrap-only or local-docker-only — not propagated to k8s.
# These are either constructed differently in k8s or are irrelevant in a cluster context.
LOCAL_ONLY_KEYS: frozenset[str] = frozenset(
    {
        # Bus selection — k8s always uses the cluster bus; BUS_ID is in ConfigMap as "cluster"
        "KAFKA_LOCAL_BOOTSTRAP_SERVERS",
        "KAFKA_BROKER_ALLOWLIST",  # local Redpanda denylist bypass; k8s uses real DNS
        # Local postgres bootstrap
        "POSTGRES_USER",  # k8s uses Infisical-sourced DSN; local docker uses default "postgres"
        # Local filesystem paths — not meaningful in container images
        "OMNIBASE_INFRA_DIR",
        # OmniMemory crawl path — local server path, has no k8s equivalent
        "OMNIMEMORY_CRAWL_PATH_PREFIXES",
        # Local runtime surface and marketplace skill roots are Docker-runtime
        # ingress controls; k8s derives package/skill mounts separately.
        "ONEX_ACTIVE_RUNTIME_PACKAGES",
        "ONEX_MARKETPLACE_SKILLS_ROOT",
        # LLM endpoints — lab-local GPU servers (192.168.86.*); k8s onex-dev
        # routes to cluster-internal or public model endpoints via a different
        # mechanism (tracked: OMN-7979). In local docker these activate PluginLlm.
        "LLM_CODER_URL",
        "LLM_CODER_FAST_URL",
        "LLM_EMBEDDING_URL",
        "LLM_DEEPSEEK_R1_URL",
        "LLM_GLM_URL",
        "LLM_GLM_MODEL_NAME",
        "BIFROST_LOCAL_CODER_ENDPOINT_URL",
        "BIFROST_LOCAL_DS_V4_FLASH_ENDPOINT_URL",
        "BIFROST_LOCAL_EMBEDDING_ENDPOINT_URL",
        "BIFROST_LOCAL_REASONER_ENDPOINT_URL",
        "LLM_ENDPOINT_CIDR_ALLOWLIST",
        "LLM_CLOUD_ENDPOINT_HOST_ALLOWLIST",
        "LOCAL_LLM_SHARED_SECRET",
        # Topic provisioner partition cap — local-only tuning knob; k8s does not set it
        "ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS",
        # OMN-15529 / OMN-15362: OnexBot-OCC-Writer App identity for the OCC
        # companion producer. Deliberately NOT propagated to k8s today — the
        # operator mints the App private key onto the .201 bus runtime only
        # (ruling 2026-07-30), and the onex-dev cluster does not run the OCC
        # companion producer. Classified local-only rather than SECRET_KEYS on
        # purpose: SECRET_KEYS asserts "k8s injects this via Secret/Infisical",
        # which would be a false claim here. If the cluster ever runs this
        # producer, ONEXBOT_OCC_* move to SECRET_KEYS and
        # OMNI_OCC_GITHUB_AUTH_MODE moves to the ConfigMap.
        "ONEXBOT_OCC_APP_ID",
        "ONEXBOT_OCC_PRIVATE_KEY",
        "OMNI_OCC_GITHUB_AUTH_MODE",
    }
)

# Keys present in docker-compose x-runtime-env but not yet added to the k8s ConfigMap.
# Each entry here is TECH DEBT that should be resolved by adding to configmap.yaml.
# Tracked in: OMN-4307
# NOTE: Do NOT add new keys here — fix them properly in the ConfigMap instead.
CONFIGMAP_DEBT_KEYS: frozenset[str] = frozenset(
    {
        # Keycloak / auth profile — needed when --profile auth is active
        "KEYCLOAK_ADMIN_URL",
        "KEYCLOAK_REALM",
        "KEYCLOAK_ADMIN_CLIENT_ID",
        "KEYCLOAK_ISSUER",
        "ONEX_SERVICE_CLIENT_ID",
        # Plugin / runtime settings
        "OMNIMEMORY_ENABLED",
        # omnimemory Memgraph integration — not yet in k8s ConfigMap (tracked: OMN-4307)
        "OMNIMEMORY_DB_URL",
        "OMNIMEMORY_MEMGRAPH_HOST",
        "OMNIMEMORY_MEMGRAPH_PORT",
        # arch-graph query/populate EFFECT bolt URI — same Memgraph instance as
        # OMNIMEMORY_MEMGRAPH_*, only reachable from the dev lane today; not
        # yet in k8s ConfigMap (tracked: OMN-14297)
        "ARCH_GRAPH_BOLT_URI",
        # Qdrant vector store connection — not yet in k8s ConfigMap (tracked: OMN-4307)
        "QDRANT_HOST",
        "QDRANT_PORT",
        "ONEX_REGISTRATION_AUTO_ACK",
        "USE_EVENT_ROUTING",
        # Runtime package activation selector. k8s ConfigMap parity is tracked
        # with the OMN-10635 release/deploy follow-up because this PR cannot
        # update the sibling omninode_infra checkout used by the parity job.
        "ONEX_ACTIVE_RUNTIME_PACKAGES",
        # Bifrost contract rendering knobs. k8s ConfigMap parity belongs with
        # the sibling omninode_infra ConfigMap update; tracked by OMN-10943.
        "BIFROST_CONTRACT_PATH",
        "BIFROST_SOURCE_CONTRACT_PATH",
        "BIFROST_VERIFY_ENDPOINTS",
        # OMN-15645: newly bound on the .201 compose lanes (omnimarket#2000 /
        # OMN-15628 fail-fast). k8s ConfigMap parity is OMN-15628's own
        # k8s-manifest-scoped acceptance criterion (In Progress) in the
        # sibling omninode_infra checkout, not this ticket's -- OMN-15645's
        # scope is the .201 compose lanes only.
        "DELEGATION_ROUTING_TIERS_PATH",
        # OpenTelemetry — opt-in observability (empty = disabled)
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "OTEL_TRACES_EXPORTER",
    }
)


# ---------------------------------------------------------------------------
# Reverse-direction classification (k8s -> compose), OMN-15628
# ---------------------------------------------------------------------------

# Keys the k8s manifests bind that describe CLUSTER topology or a managed data
# plane, and therefore have no docker-compose counterpart by construction.
# Every entry below is justified by its live ConfigMap value.
K8S_ONLY_KEYS: frozenset[str] = frozenset(
    {
        # Managed bus (MSK + IAM auth). Local/lab lanes run a PLAINTEXT Redpanda
        # broker, so none of these have a compose analogue.
        #   KAFKA_RUNTIME_TARGET=msk, KAFKA_SECURITY_PROTOCOL=SASL_SSL,
        #   KAFKA_SASL_MECHANISM=AWS_MSK_IAM, KAFKA_MSK_REGION=us-east-1
        "KAFKA_RUNTIME_TARGET",
        "KAFKA_SECURITY_PROTOCOL",
        "KAFKA_SASL_MECHANISM",
        "KAFKA_JAVA_SASL_MECHANISM",
        "KAFKA_NON_JAVA_SASL_MECHANISM",
        "KAFKA_MSK_REGION",
        # Split DSN form. k8s composes the infra DSN from host + port + a
        # Secret-sourced password (see the OMNIBASE_INFRA_DB_URL note in
        # SECRET_KEYS); compose binds the whole DSN in one variable.
        "OMNIBASE_INFRA_DB_HOST",
        "OMNIBASE_INFRA_DB_PORT",
        "POSTGRES_SSLMODE",  # 'require' — managed RDS; local postgres is in-network
        # Cluster-DNS service addresses (*.svc.cluster.local). Compose reaches
        # the same services by compose-network service name / HOST+PORT pairs.
        "CONTRACT_RESOLVER_URL",
        "INTELLIGENCE_API_URL",
        "KREUZBERG_URL",
        "QDRANT_URL",
        # Cluster Infisical bootstrap toggle; the compose lanes gate Infisical on
        # INFISICAL_ADDR being set instead (see config_discovery docs).
        "INFISICAL_REQUIRED",
    }
)

# Keys bound in k8s but NOT bound in docker-compose. Each entry here is TECH
# DEBT: the compose lanes run without a setting the cluster considers part of
# its runtime configuration, so the two surfaces are provably not equivalent.
# Tracked in: OMN-4307 (parity backlog).
# NOTE: Do NOT add new keys here — bind the key in docker-compose instead, or
# classify it in K8S_ONLY_KEYS with a value-backed justification.
COMPOSE_PARITY_DEBT_KEYS: frozenset[str] = frozenset(
    {
        # Runtime feature flags set cluster-side only.
        "ENABLE_PATTERN_ENFORCEMENT",
        "ENABLE_REAL_TIME_EVENTS",
        "KAFKA_ENABLE_INTELLIGENCE",
        "ONEX_BOOT_UNIVERSE_PROVISION",
        # runtime-worker push-validation scratch root (a cluster mount path);
        # the compose worker has no equivalent binding today.
        "ONEX_PUSH_VALIDATION_WORKROOT",
        # skill-lifecycle consumer knobs bound inline on the k8s Deployment but
        # left to in-code defaults in compose.
        "OMNIBASE_INFRA_SKILL_LIFECYCLE_DLQ_TOPIC",
        "OMNIBASE_INFRA_SKILL_LIFECYCLE_SCHEMA_VERSION",
        "OMNIBASE_INFRA_SKILL_LIFECYCLE_HEALTH_CHECK_STALENESS_SECONDS",
    }
)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------


def extract_runtime_env_keys(compose_path: Path) -> set[str]:
    """Extract all keys declared inside the x-runtime-env anchor.

    Uses a regex over the raw YAML text rather than YAML parsing so that
    variable-expansion syntax (e.g. ``${VAR:-default}``) is preserved intact
    and we capture the *key* name without evaluating the value.
    """
    raw = compose_path.read_text()
    block_match = re.search(r"x-runtime-env:.*?(?=\n\S|\Z)", raw, re.DOTALL)
    if not block_match:
        return set()
    block = block_match.group(0)
    keys = re.findall(r"^\s{2}([A-Z0-9_]+):", block, re.MULTILINE)
    return set(keys)


def extract_configmap_keys(configmap_path: Path) -> set[str]:
    """Extract all keys from the ConfigMap ``data:`` section."""
    data = yaml.safe_load(configmap_path.read_text())
    return set(data.get("data", {}).keys())


def _service_env_keys(service: object) -> set[str]:
    """Return the env keys a single compose service declares."""
    if not isinstance(service, dict):
        return set()
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return {str(k) for k in env}
    if isinstance(env, list):
        # ``- KEY=value`` / ``- KEY`` list form
        return {str(item).split("=", 1)[0] for item in env}
    return set()


def extract_compose_bound_keys(compose_path: Path) -> set[str]:
    """Extract every env key bound by ANY service in a compose file.

    The forward check walks only the ``x-runtime-env`` anchor because that is
    the shared surface it governs. The reverse check must ask a broader
    question — "does this key reach a container at all?" — so it takes the union
    over every service's resolved ``environment`` mapping (YAML merge keys and
    anchors are resolved by ``yaml.safe_load``, so anchor-merged services
    contribute the anchor's keys).
    """
    document = yaml.safe_load(compose_path.read_text())
    services = (document or {}).get("services") or {}
    keys: set[str] = set()
    for service in services.values():
        keys |= _service_env_keys(service)
    return keys


def extract_k8s_bound_keys(runtime_dir: Path) -> dict[str, set[str]]:
    """Map every k8s-bound configuration key to the manifests that bind it.

    Sources:
      * ``configmap.yaml`` ``data:`` keys
      * ``deployment-*.yaml`` container ``env:`` entries that carry a literal
        ``value:``

    ``valueFrom`` entries (``secretKeyRef``/``fieldRef``) are deliberately out of
    scope: those are credentials/downward-API values whose compose-side
    counterpart is the host-env + Infisical surface already governed by
    SECRET_KEYS in the forward direction, not a compose literal.
    """
    bound: dict[str, set[str]] = {}

    configmap = runtime_dir / "configmap.yaml"
    for key in extract_configmap_keys(configmap):
        bound.setdefault(key, set()).add(configmap.name)

    for manifest in sorted(runtime_dir.glob("deployment-*.yaml")):
        document = yaml.safe_load(manifest.read_text())
        containers = (
            (document or {})
            .get("spec", {})
            .get("template", {})
            .get("spec", {})
            .get("containers", [])
        )
        for container in containers:
            for entry in container.get("env", []) or []:
                if "value" in entry:
                    bound.setdefault(str(entry["name"]), set()).add(manifest.name)

    return bound


def extract_k8s_env_value(manifest_path: Path, key: str) -> str | None:
    """Return the literal ``value:`` a Deployment binds for ``key``, if any."""
    document = yaml.safe_load(manifest_path.read_text())
    containers = (
        (document or {})
        .get("spec", {})
        .get("template", {})
        .get("spec", {})
        .get("containers", [])
    )
    for container in containers:
        for entry in container.get("env", []) or []:
            if entry.get("name") == key and "value" in entry:
                return str(entry["value"])
    return None


def extract_compose_service_env_value(
    compose_path: Path, service: str, key: str
) -> str | None:
    """Return the resolved env value a compose service binds for ``key``."""
    document = yaml.safe_load(compose_path.read_text())
    services = (document or {}).get("services") or {}
    env = (services.get(service) or {}).get("environment") or {}
    if isinstance(env, dict):
        value = env.get(key)
        return None if value is None else str(value)
    if isinstance(env, list):
        for item in env:
            name, _, value = str(item).partition("=")
            if name == key:
                return value
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
def test_runtime_env_keys_have_k8s_entries() -> None:
    """Every x-runtime-env key is bound in k8s, SECRET_KEYS, or LOCAL_ONLY_KEYS.

    If this test fails, a key was added to docker-compose x-runtime-env without
    a corresponding binding in the k8s manifests (omninode_infra), and it is not
    registered in SECRET_KEYS (for Infisical/k8s Secret sources),
    LOCAL_ONLY_KEYS (for local-dev-only bootstrap variables), or
    CONFIGMAP_DEBT_KEYS (known gaps tracked as tech debt).

    The k8s surface is the SAME one the reverse check uses (OMN-15628):
    ConfigMap ``data`` keys PLUS literal ``value:`` entries on the runtime
    Deployments. It was ConfigMap-only before, which made the two directions
    disagree about what "bound in k8s" means — a key bound inline on a
    Deployment (the placement used for DELEGATION_ROUTING_TIERS_PATH and
    BIFROST_CONTRACT_PATH) read as absent here even though the cluster sets it.

    To fix a failure, choose one of:
      1. Bind the key in k8s (omninode_infra/k8s/onex-dev/runtime/configmap.yaml,
         or inline on the Deployments that need it)
      2. Add the key to SECRET_KEYS in this file if it is sourced from a k8s Secret
      3. Add the key to LOCAL_ONLY_KEYS if it is intentionally absent from k8s
      4. Add temporarily to CONFIGMAP_DEBT_KEYS if the k8s update is blocked
         (but you MUST file a ticket to resolve the debt)
    """
    if K8S_RUNTIME_DIR is None:
        pytest.skip(
            "omninode_infra not found as a sibling — set OMNINODE_INFRA_DIR to run this test"
        )

    compose_keys = extract_runtime_env_keys(COMPOSE_PATH)
    assert compose_keys, (
        f"No x-runtime-env keys extracted from {COMPOSE_PATH}. "
        "Has the anchor been renamed or removed?"
    )

    k8s_keys = set(extract_k8s_bound_keys(K8S_RUNTIME_DIR))
    accounted_for = k8s_keys | SECRET_KEYS | LOCAL_ONLY_KEYS | CONFIGMAP_DEBT_KEYS
    missing = compose_keys - accounted_for

    assert not missing, (
        "Keys in x-runtime-env but not bound anywhere in the onex-dev k8s manifests "
        "(and not in SECRET_KEYS, LOCAL_ONLY_KEYS, or CONFIGMAP_DEBT_KEYS):\n"
        + "\n".join(f"  {k}" for k in sorted(missing))
        + "\n\nFix: add each missing key to one of:\n"
        "  • omninode_infra/k8s/onex-dev/runtime/configmap.yaml  (preferred)\n"
        "  • SECRET_KEYS in tests/ci/test_env_parity.py           (k8s Secret source)\n"
        "  • LOCAL_ONLY_KEYS in tests/ci/test_env_parity.py       (local dev only)\n"
        "  • CONFIGMAP_DEBT_KEYS in tests/ci/test_env_parity.py   (temp — must file ticket)"
    )


@pytest.mark.ci
def test_k8s_bound_keys_are_bound_in_compose() -> None:
    """Reverse parity (OMN-15628): every k8s-bound config key reaches compose.

    A key bound on the onex-dev Deployments/ConfigMap but bound in NO compose
    service means the docker lanes (dev / lab / stability-test / prod on .201)
    run without configuration the cluster treats as required. When the consumer
    resolves that key fail-closed — as omnimarket's delegation routing reducer
    does via ``resolve_required_path_config`` — the container still boots
    healthy and only fails on the first request that touches the seam, so no
    health check and no forward-only parity walk can detect it.

    To fix a failure, choose one of:
      1. Bind the key in docker/docker-compose.infra.yml (preferred — put it on
         the ``x-runtime-env`` anchor if every runtime service needs it)
      2. Add it to K8S_ONLY_KEYS with a value-backed justification if it
         describes cluster topology or a managed data plane
      3. Add it to COMPOSE_PARITY_DEBT_KEYS only if the compose binding is
         genuinely blocked (and file/cite a ticket)
    """
    if K8S_RUNTIME_DIR is None:
        pytest.skip(
            "omninode_infra not found as a sibling — set OMNINODE_INFRA_DIR to run this test"
        )

    k8s_bound = extract_k8s_bound_keys(K8S_RUNTIME_DIR)
    assert k8s_bound, (
        f"No k8s-bound env keys extracted from {K8S_RUNTIME_DIR}. "
        "Have the runtime manifests moved or been renamed?"
    )

    compose_keys = extract_compose_bound_keys(COMPOSE_PATH)
    assert compose_keys, (
        f"No service env keys extracted from {COMPOSE_PATH}. "
        "Has the services block been restructured?"
    )

    accounted_for = compose_keys | K8S_ONLY_KEYS | COMPOSE_PARITY_DEBT_KEYS
    missing = {k: v for k, v in k8s_bound.items() if k not in accounted_for}

    assert not missing, (
        "Keys bound in the onex-dev k8s manifests but bound in NO "
        f"docker-compose service ({COMPOSE_PATH.name}), and not classified as "
        "K8S_ONLY_KEYS or COMPOSE_PARITY_DEBT_KEYS:\n"
        + "\n".join(
            f"  {k}  (k8s source: {', '.join(sorted(missing[k]))})"
            for k in sorted(missing)
        )
        + "\n\nFix: bind each key in docker/docker-compose.infra.yml, or classify it in\n"
        "  • K8S_ONLY_KEYS in tests/ci/test_env_parity.py            (cluster-only, justify with the value)\n"
        "  • COMPOSE_PARITY_DEBT_KEYS in tests/ci/test_env_parity.py (temp — must cite a ticket)"
    )


@pytest.mark.ci
def test_delegation_routing_tiers_path_matches_k8s_pin() -> None:
    """OMN-15628 seam lock: the routing-tiers path agrees across both surfaces.

    ``DELEGATION_ROUTING_TIERS_PATH`` is resolved fail-closed by omnibase_infra's
    delegation routing consumers (omnimarket
    ``handler_delegation_routing._get_config`` →
    ``resolve_required_path_config``), so compose and k8s pointing at different
    in-container paths is a silent request-path break, not a boot failure. Lock
    the two literals together on every runtime-family service/Deployment pair.
    """
    if K8S_RUNTIME_DIR is None:
        pytest.skip(
            "omninode_infra not found as a sibling — set OMNINODE_INFRA_DIR to run this test"
        )

    key = "DELEGATION_ROUTING_TIERS_PATH"

    k8s_values: dict[str, str | None] = {
        manifest: extract_k8s_env_value(K8S_RUNTIME_DIR / manifest, key)
        for manifest in K8S_RUNTIME_FAMILY_DEPLOYMENTS
    }
    unbound_k8s = [m for m, v in k8s_values.items() if v is None]
    assert not unbound_k8s, (
        f"{key} is not bound on these onex-dev runtime Deployments: "
        f"{', '.join(unbound_k8s)}. The compose side is pinned to it; unbinding "
        "one side reopens the OMN-15628 seam."
    )

    distinct_k8s = set(k8s_values.values())
    assert len(distinct_k8s) == 1, (
        f"{key} is pinned to different values across the onex-dev runtime "
        f"Deployments: {k8s_values}"
    )
    expected = next(iter(distinct_k8s))

    compose_values: dict[str, str | None] = {
        service: extract_compose_service_env_value(COMPOSE_PATH, service, key)
        for service in COMPOSE_RUNTIME_FAMILY_SERVICES
    }
    mismatched = {s: v for s, v in compose_values.items() if v != expected}

    assert not mismatched, (
        f"{key} disagrees between docker-compose and the onex-dev k8s pin.\n"
        f"  k8s pin ({', '.join(K8S_RUNTIME_FAMILY_DEPLOYMENTS)}): {expected}\n"
        + "\n".join(
            f"  compose service {s!r}: {v!r}" for s, v in sorted(mismatched.items())
        )
        + f"\n\nFix: bind {key} to the k8s pin in {COMPOSE_PATH.name} "
        "(the x-runtime-env anchor covers all three runtime services at once)."
    )


@pytest.mark.ci
def test_every_runtime_compose_lane_binds_delegation_routing_tiers_path() -> None:
    """OMN-15628 lane coverage: no docker lane can come up without the pin.

    Three shapes of compose file stand up a runtime container:

    * ANCHOR OWNERS — files with their own ``x-runtime-env`` anchor
      (``infra`` = dev/lab base, ``judge``). Each must bind the key in its own
      anchor.
    * LANE OVERLAYS — layered on ``infra`` with ``-f infra -f <lane>``
      (prod, stability-test, dev-lane; see ``resolve_compose_file_args`` in
      scripts/deploy-runtime.sh). Compose merges ``environment`` mappings
      key-by-key across ``-f`` layers, so these inherit the base pin. That
      inheritance holds ONLY while no overlay replaces the mapping wholesale
      with ``environment: !override`` — assert that it does not.
    * STANDALONE — files that layer nothing (e2e). Must bind the key directly.
    """
    docker_dir = COMPOSE_PATH.parent

    # (filename, top-level anchor key) — judge names its anchor differently.
    anchor_owners = (
        ("docker-compose.infra.yml", "x-runtime-env"),
        ("docker-compose.judge.yml", "x-judge-runtime-env"),
    )
    lane_overlays = (
        "docker-compose.prod.yml",
        "docker-compose.stability-test.yml",
        "docker-compose.dev-lane.yml",
    )
    standalone = ("docker-compose.e2e.yml",)

    key = "DELEGATION_ROUTING_TIERS_PATH"

    for filename, anchor_key in anchor_owners:
        raw = (docker_dir / filename).read_text()
        anchor = re.search(rf"^{anchor_key}:.*?(?=\n\S|\Z)", raw, re.DOTALL | re.M)
        assert anchor is not None, f"{filename} has no {anchor_key} anchor"
        assert re.search(rf"^\s{{2}}{key}:", anchor.group(0), re.MULTILINE), (
            f"{filename} owns its own {anchor_key} anchor but does not bind {key}. "
            "Every runtime container it starts will fail closed on the first "
            "delegation-routing request (OMN-15628)."
        )

    for filename in lane_overlays:
        raw = (docker_dir / filename).read_text()
        assert not re.search(r"^\s*environment:\s*!!?override\b", raw, re.MULTILINE), (
            f"{filename} replaces a service's `environment` mapping wholesale with "
            "!override. That severs the compose merge that carries "
            f"{key} (and every other x-runtime-env key) from "
            "docker-compose.infra.yml into this lane. Bind the key explicitly in "
            "this overlay, or drop the !override."
        )

    for filename in standalone:
        raw = (docker_dir / filename).read_text()
        assert re.search(rf"^\s+{key}:", raw, re.MULTILINE), (
            f"{filename} layers no base compose file, so it inherits nothing — "
            f"it must bind {key} on its runtime service directly."
        )


@pytest.mark.ci
def test_compose_path_exists() -> None:
    """Sanity guard: docker-compose file is present at the expected path."""
    assert COMPOSE_PATH.exists(), (
        f"docker-compose file not found at {COMPOSE_PATH}. "
        "Has the docker/ directory been moved or renamed?"
    )


@pytest.mark.ci
def test_extract_runtime_env_finds_keys() -> None:
    """x-runtime-env anchor exists and contains at least one key."""
    keys = extract_runtime_env_keys(COMPOSE_PATH)
    assert keys, (
        f"No x-runtime-env keys found in {COMPOSE_PATH}. "
        "The anchor may have been removed or renamed."
    )
