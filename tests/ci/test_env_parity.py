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

Scope of each direction (read this before extending either — they deliberately
use DIFFERENT k8s surfaces, and the reverse walk is deliberately single-file):

* FORWARD is scoped to the compose ``x-runtime-env`` anchor, which compose
  merges into all three runtime-family services at once. Its k8s counterpart is
  therefore the RUNTIME-FAMILY surface: ConfigMap keys (every runtime Deployment
  ``envFrom``-s ``onex-runtime-config``) plus keys bound inline on ALL THREE
  runtime Deployments. A key bound inline on only one workload does NOT satisfy
  it — see ``test_k8s_family_surface_requires_all_runtime_deployments``.
* REVERSE is scoped to ``docker-compose.infra.yml`` plus the typed service
  manifests under ``docker/catalog/services/`` (rendered by the catalog CLI into
  ``docker/docker-compose.generated.yml``, which is not committed — see
  ``extract_catalog_bound_keys``), and asks the weaker question "does this key
  reach ANY compose container at all?", so it uses the
  UNION of every k8s workload's bindings. ``infra.yml`` is the base file that
  ``resolve_compose_file_args`` layers first for every deployed lane, so a key
  bound there reaches all of them. The standalone lanes (``judge``, ``e2e``)
  are NOT reverse-walked — they run deliberately narrower service sets, so a
  full reverse walk against them would report design, not drift. They are
  covered for the OMN-15628 seam specifically, by value, in
  ``test_delegation_routing_tiers_path_matches_k8s_pin``.

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

import ast
import fnmatch
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

# Typed service manifests the catalog CLI renders into
# docker/docker-compose.generated.yml. Second compose-binding surface for the
# reverse walk — see extract_catalog_bound_keys.
CATALOG_SERVICES_DIR = _REPO_ROOT / "docker" / "catalog" / "services"

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

# Every compose file that stands up a delegation-runtime container, mapped to
# the services in it that must carry the k8s-pinned routing-tiers VALUE (not
# merely the key). ``infra`` is the dev/lab base every deployed lane layers;
# ``judge`` and ``e2e`` are standalone and inherit nothing, so a typo in either
# would otherwise be invisible to a presence-only check.
COMPOSE_RUNTIME_SERVICES_BY_FILE: dict[str, tuple[str, ...]] = {
    "docker-compose.infra.yml": COMPOSE_RUNTIME_FAMILY_SERVICES,
    "docker-compose.judge.yml": ("omninode-runtime", "runtime-effects"),
    "docker-compose.e2e.yml": ("runtime",),
}

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
        # OMN-16843 (compose half) / OMN-15426 (epic). Internal-projection DSN,
        # principal omninode_runtime. SECRET_KEYS is a literal claim about the
        # cluster, and it holds: omninode_infra#803 binds this key on ALL THREE
        # onex-dev runtime Deployments (omninode-runtime, -effects, -worker) via
        # secretKeyRef -> onex-runtime-credentials with `optional: false`, and
        # registers it in k8s/onex-dev/secret-ownership-manifest.yaml. Same
        # shape as OMNIDASH_ANALYTICS_DB_URL above.
        "OMNINODE_INTERNAL_DB_URL",
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
        # OMN-16778: Slack alert delivery credentials, classified exactly as the
        # OnexBot App identity above and for the same reason. The bot token and
        # channel id are minted into the operator's host .env on the bus runtime
        # hosts (.201: /data/omninode/omnibase_infra/.env, mode 0600); the
        # onex-dev cluster carries NO Slack key at all -- `grep -rl SLACK
        # omninode_infra/k8s` returns nothing, verified 2026-08-29 -- and does
        # not run the Slack alerting path. LOCAL_ONLY_KEYS rather than
        # SECRET_KEYS on purpose: SECRET_KEYS asserts "k8s injects this via
        # Secret/Infisical", which would be a false claim today. If the cluster
        # ever runs node_slack_publish_effect, these move to SECRET_KEYS.
        "SLACK_BOT_TOKEN",
        "SLACK_CHANNEL_ID",
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
        #
        # OMN-17502 CORRECTION. The justification that used to stand here —
        # "the render itself only runs where BIFROST_CONTRACT_PATH is bound, so
        # the pin travels with these knobs as one debt item — the cluster
        # binding lands together with BIFROST_CONTRACT_PATH's or not at all" —
        # was already false the day it was written, and BIFROST_CONTRACT_PATH
        # is removed from this set with it. BIFROST_CONTRACT_PATH has been bound
        # non-empty INLINE on all three onex-dev runtime Deployments since
        # omninode_infra#792 (OMN-15628, 2026-08-01), a month before OMN-17150,
        # so it is ON the k8s runtime-family surface and was never debt. The
        # cluster DOES render at boot; the two halves did NOT travel together;
        # and the consequence was OMN-17502 — the released OMN-17150 image
        # fail-closed on the unbound per-lane pin and put omninode-runtime,
        # -effects and -worker into CrashLoopBackOff on onex-dev.
        #
        "BIFROST_SOURCE_CONTRACT_PATH",
        "BIFROST_VERIFY_ENDPOINTS",
        # OMN-15425 (compose half) / OMN-16953 (k8s half). Tenant-projection
        # DSN, principal tenant_projection_writer. Deliberately NOT in
        # SECRET_KEYS: that set is a literal claim that the key is bound on the
        # cluster, and unlike its sibling OMNINODE_INTERNAL_DB_URL (bound by
        # omninode_infra#803 on all three onex-dev runtime Deployments) nothing
        # binds this one yet. Its eventual home is a `secretKeyRef` — it carries
        # an embedded credential and must never land in a ConfigMap — so
        # OMN-16953 moves it to SECRET_KEYS in the same change that makes the
        # claim true. Parking it here records the gap honestly instead of
        # asserting a binding that does not exist.
        "ONEX_TENANT_DB_URL",
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
        # OMN-16689 k8s BYOK credential writer addressing. These are cluster
        # Infisical coordinates, not compose-runtime knobs:
        #   INFISICAL_ENVIRONMENT_SLUG="dev"
        #   INFISICAL_TENANT_CREDENTIAL_SECRET_PATH="/tenant-inference-credentials"
        # The compose lanes in this repo do not run the tenant credential writer
        # Deployment introduced in omninode_infra#1074, and a repo-wide grep in
        # omnibase_infra has no consumer for either key. Binding them in
        # docker-compose.infra.yml would invent a compose contract that no
        # container resolves.
        "INFISICAL_ENVIRONMENT_SLUG",
        "INFISICAL_TENANT_CREDENTIAL_SECRET_PATH",
        # OMN-15750 gateway-attach ingress (omninode_infra#886). Same cluster-DNS
        # category as the block above — both values are *.svc.cluster.local:
        #   GATEWAY_ATTACH_KEYCLOAK_INTROSPECTION_URL=
        #     http://keycloak.auth.svc.cluster.local/realms/omninode/protocol/openid-connect/token/introspect
        #   GATEWAY_ATTACH_KEYCLOAK_JWKS_URL=
        #     http://keycloak.auth.svc.cluster.local/realms/omninode/protocol/openid-connect/certs
        # They address the Keycloak in the cluster's OWN `auth` namespace, which
        # has no compose analogue: docker-compose.infra.yml's `keycloak` service
        # is a dev bootstrap on the compose network (KEYCLOAK_ADMIN_URL=
        # http://keycloak:8080), seeded from docker/keycloak/omninode-realm.json.
        #
        # Binding these in compose would ALSO be wrong on two independent counts,
        # so this is not debt deferred for convenience:
        #   1. No compose lane resolves these refs. node_gateway_attach_effect
        #      reads them by LOGICAL ref (contract.yaml keycloak_introspection_ref
        #      = "gateway.attach.keycloak.introspection", keycloak_jwks_ref =
        #      "gateway.attach.keycloak.jwks") through the secret resolver. The
        #      k8s ConfigMap's ONEX_SECRET_RESOLVER_CONFIG_JSON maps those two
        #      logical names onto these env vars; the compose dev/lab/stability/
        #      prod resolver config (docker/runtime-policy.env
        #      DEV_RUNTIME_*_SECRET_RESOLVER_CONFIG_JSON) declares only llm.* and
        #      slack.bot_token — no gateway.attach.* mapping exists, so no compose
        #      container ever asks for either key.
        #   2. The gateway ingress is not in a deployed compose lane at all.
        #      resolve_compose_file_args (scripts/deploy-runtime.sh) layers
        #      docker-compose.infra.yml + one lane overlay; the gateway services
        #      live in docker-compose.gateway.yml / .gateway-attach-test-lane.yml,
        #      which no lane layers. And OMN-15750's own acceptance criteria
        #      forbid the binding outright: "No broker/Keycloak URL literal in
        #      source, docker-compose, or env — resolved from contract ref at the
        #      effect boundary."
        "GATEWAY_ATTACH_KEYCLOAK_INTROSPECTION_URL",
        "GATEWAY_ATTACH_KEYCLOAK_JWKS_URL",
        # k8s readiness/liveness probe plumbing for the three standalone
        # omnimarket projection-writer Deployments (OMN-15905). Values are 8093
        # (live-events), 8094 (registration), 8095 (delegation) — each is the
        # port that Deployment's OWN readinessProbe (httpGet /ready) and
        # livenessProbe (tcpSocket) target, so the binding exists only to serve
        # the kubelet. The compose counterparts of those three workloads are
        # catalog services (docker/catalog/services/omnimarket-projection-*.yaml)
        # which declare `healthcheck: null` and `ports: null` — there is no probe
        # to answer, so BaseProjectionRunner's opt-in health server stays off and
        # the key has no compose counterpart by construction.
        "PROJECTION_RUNNER_HEALTH_PORT",
        # omniweb Auth.js v5 public callback origin (OMN-16205, promoted into the
        # shared onex-dev ConfigMap by omninode_infra#947, merged 2026-08-25:
        #   AUTH_URL="https://dev.app.omninode.ai"
        # omniweb is not a docker-compose service in this repo at all --
        # docker/docker-compose.infra.yml has no `omniweb` entry, only Postgres
        # migrations that create its DB role (docker/migrations/forward/
        # 052_create_role_omniweb.sql). AUTH_URL is consumed exclusively by the
        # k8s-deployed omniweb NextAuth pod to derive its own public callback
        # origin (see k8s/onex-dev/omniweb/deployment.yaml's valueFrom, and the
        # OMN-15294/OMN-16209 background on why a bind/loopback address there is
        # a live signup-breaking defect) -- no compose lane runs that workload,
        # so there is nothing for this key to bind to by construction, not by
        # deferred convenience.
        "AUTH_URL",
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
# Flags this repo has deleted (OMN-15659)
# ---------------------------------------------------------------------------
# OMN-8779 / OMN-8780 deleted a set of feature flags from omnibase_infra because
# they defaulted to false and were therefore silent non-enforcement gates.
# ``tests/audit/test_no_dead_delegation_flags.py`` keeps them deleted by
# rejecting their names anywhere in this tree.
#
# That puts the reverse walk in a genuine deadlock for any such flag the cluster
# still binds: EVERY remedy the reverse walk prescribes -- bind it in compose,
# list it in K8S_ONLY_KEYS, list it in COMPOSE_PARITY_DEBT_KEYS -- requires
# writing the rejected name. Both gates cannot be satisfied at once. That is the
# defect OMN-15659 fixes: OMN-15628 classified one of these flags as compose
# parity debt, the audit rejected the classification, and dev went red.
#
# Resolution: a flag THIS repo deleted is out of scope for compose parity. Its
# absence from the compose lanes is the intended end state, not a gap. The
# shared onex-dev ConfigMap may still bind it for another owner -- the canonical
# feature-flag registry in ``omnibase_core.feature_flags.registry`` assigns each
# flag an ``owning_repo`` -- and a binding owned by another repo is not
# omnibase_infra parity debt.
#
# The set is READ FROM THE AUDIT rather than restated, so the two gates cannot
# drift apart and so this module never names a rejected flag. Excluding these
# keys does not create a blind spot: the audit itself scans every ``*.yml`` /
# ``*.yaml`` in this tree, so a compose lane that re-bound one of them would
# fail that audit directly.
#
# Fail-closed: a missing file, a renamed symbol, or an empty list raises at
# import rather than silently yielding an empty exclusion set.
_DEAD_FLAG_AUDIT_PATH = (
    _REPO_ROOT / "tests" / "audit" / "test_no_dead_delegation_flags.py"
)
_DEAD_FLAG_AUDIT_SYMBOL = "_DEAD_FLAGS"


def _load_repo_deleted_flags() -> frozenset[str]:
    """Return the flag names the dead-flag audit rejects from this tree."""
    try:
        source = _DEAD_FLAG_AUDIT_PATH.read_text()
    except OSError as exc:  # pragma: no cover - fail closed
        raise RuntimeError(
            f"cannot read the dead-flag audit at {_DEAD_FLAG_AUDIT_PATH}: {exc}. "
            "The reverse parity walk cannot resolve which flags this repo deleted."
        ) from exc

    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == _DEAD_FLAG_AUDIT_SYMBOL
            for target in node.targets
        ):
            continue
        names = frozenset(str(name) for name in ast.literal_eval(node.value))
        if not names:
            raise RuntimeError(
                f"{_DEAD_FLAG_AUDIT_PATH}: {_DEAD_FLAG_AUDIT_SYMBOL} is empty"
            )
        return names

    raise RuntimeError(
        f"{_DEAD_FLAG_AUDIT_SYMBOL} not found in {_DEAD_FLAG_AUDIT_PATH}. "
        "The reverse parity walk cannot resolve which flags this repo deleted."
    )


REPO_DELETED_FLAG_KEYS: frozenset[str] = _load_repo_deleted_flags()


# ---------------------------------------------------------------------------
# Compose YAML loading
# ---------------------------------------------------------------------------
# The lane overlays (prod / stability-test / judge) use the Compose merge
# directives ``!override`` and ``!reset``, which plain ``yaml.safe_load`` refuses
# with ConstructorError. Preserve the tag instead of dropping it: whether a
# service's ``environment`` mapping carries one of those tags is exactly the
# fact the lane-coverage test needs to assert, so it must survive parsing.


class ComposeTagged:
    """A YAML node that carried a Compose merge directive (``!override`` / ``!reset``)."""

    __slots__ = ("tag", "value")

    def __init__(self, tag: str, value: object) -> None:
        self.tag = tag
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"ComposeTagged({self.tag!r}, {self.value!r})"


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that keeps unknown tags as :class:`ComposeTagged` wrappers.

    Deliberately NOT shared with the same-named loader in
    ``tests/unit/infra/test_judge_compose_profile.py``: that one UNWRAPS compose
    merge tags to their bare value, which is the opposite of what this module
    needs. Whether ``environment`` carries ``!override`` / ``!reset`` is the
    fact ``test_every_runtime_compose_lane_binds_delegation_routing_tiers_path``
    asserts on, so the tag has to survive parsing here.
    """


# The complete set of Compose merge directives. Anything else stays a hard
# ConstructorError rather than being silently wrapped — a tag this module does
# not understand should fail loudly, not read as "no directive present".
COMPOSE_MERGE_DIRECTIVES: tuple[str, ...] = ("!override", "!reset")


def _construct_tagged(loader: yaml.SafeLoader, node: yaml.Node) -> ComposeTagged:
    if isinstance(node, yaml.MappingNode):
        value: object = loader.construct_mapping(node, deep=True)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    elif isinstance(node, yaml.ScalarNode):
        value = loader.construct_scalar(node)
    else:  # pragma: no cover - PyYAML emits no other node kinds
        raise TypeError(f"unsupported node type for {node.tag}: {type(node).__name__}")
    return ComposeTagged(node.tag, value)


for _directive in COMPOSE_MERGE_DIRECTIVES:
    _ComposeLoader.add_constructor(_directive, _construct_tagged)


def load_compose(compose_path: Path) -> dict[str, object]:
    """Parse a compose file, tolerating ``!override`` / ``!reset`` directives."""
    # _ComposeLoader extends SafeLoader; the multi-constructor only wraps
    # unknown tags in a ComposeTagged record and never instantiates arbitrary
    # objects, so S506's arbitrary-deserialization concern does not apply. Same
    # justification and same suppression the four sibling compose-parsing tests
    # under tests/unit/infra/ already carry.
    document = yaml.load(compose_path.read_text(), Loader=_ComposeLoader)  # noqa: S506
    return document if isinstance(document, dict) else {}


def compose_services(compose_path: Path) -> dict[str, object]:
    """Return the ``services`` mapping of a compose file (empty if absent)."""
    services = load_compose(compose_path).get("services")
    return services if isinstance(services, dict) else {}


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


def service_environment(service: object) -> object:
    """Return a compose service's raw ``environment`` node (tag preserved)."""
    if not isinstance(service, dict):
        return None
    return service.get("environment")


def _service_env_keys(service: object) -> set[str]:
    """Return the env keys a single compose service declares."""
    env = service_environment(service)
    if isinstance(env, ComposeTagged):
        env = env.value
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
    anchors are resolved at parse time, so anchor-merged services contribute
    the anchor's keys).
    """
    keys: set[str] = set()
    for service in compose_services(compose_path).values():
        keys |= _service_env_keys(service)
    return keys


def extract_catalog_bound_keys(catalog_services_dir: Path) -> set[str]:
    """Extract every env key the service catalog binds on a generated container.

    The reverse walk's question is "does this key reach ANY compose container at
    all?", and ``docker-compose.infra.yml`` is not the whole answer. Services in
    ``docker/catalog/services/*.yaml`` are compose services too — the catalog CLI
    (``src/omnibase_infra/docker/catalog/cli.py``) renders them into
    ``docker/docker-compose.generated.yml``, which is how ``onex up <bundle>``
    starts them. That generated file is NOT committed, so the manifests are the
    tracked surface and the only one CI can read.

    Without this term the walk reports a false positive for any key bound on a
    workload that exists in the catalog but not in ``infra.yml``. The three
    standalone omnimarket projection writers are exactly that shape: k8s runs
    them as their own Deployments, compose runs them from the
    ``omnimarket-projections`` bundle, and ``infra.yml`` has no projection-writer
    service at all (only ``projection-api``). ``KAFKA_CONSUMER_GROUP`` was
    reported as drift on that basis while
    ``docker/catalog/services/omnimarket-projection-delegation.yaml`` had bound it
    the whole time — and NEITHER classification bucket could honestly absorb it:
    K8S_ONLY_KEYS asserts "no docker-compose counterpart by construction" and
    COMPOSE_PARITY_DEBT_KEYS asserts "the compose lanes run without this
    setting", both false claims here. An incomplete surface has to be fixed at
    the surface, not papered over with a classification.

    The key set mirrors ``generator.py``'s ``environment`` assembly exactly —
    ``hardcoded_env`` | ``operational_defaults`` | ``catalog_env`` |
    ``required_env`` — so a manifest field that reaches a container is visible
    here, and one that does not is not.
    """
    keys: set[str] = set()
    for manifest_path in sorted(catalog_services_dir.glob("*.yaml")):
        manifest = yaml.safe_load(manifest_path.read_text())
        if not isinstance(manifest, dict):
            continue
        for field in ("hardcoded_env", "operational_defaults", "catalog_env"):
            mapping = manifest.get(field)
            if isinstance(mapping, dict):
                keys |= {str(k) for k in mapping}
        required = manifest.get("required_env")
        if isinstance(required, list):
            keys |= {str(item) for item in required}
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


def extract_k8s_runtime_family_bound_keys(runtime_dir: Path) -> set[str]:
    """Keys bound for EVERY runtime-family workload, not merely somewhere in k8s.

    This is the correct counterpart for the FORWARD direction. The compose
    ``x-runtime-env`` anchor is merged into all three runtime services at once,
    so "this anchor key exists in k8s" is only true if every runtime workload
    actually receives it. Two sources qualify:

      * ConfigMap ``data`` keys — every runtime Deployment ``envFrom``-s
        ``onex-runtime-config``, so a ConfigMap key reaches all of them.
      * Keys bound inline with a literal ``value:`` on ALL of
        :data:`K8S_RUNTIME_FAMILY_DEPLOYMENTS`.

    A key bound inline on only SOME runtime Deployments (today:
    ``OMNIINTELLIGENCE_PUBLISH_INTROSPECTION``, ``ONEX_PUSH_VALIDATION_WORKROOT``)
    is deliberately excluded — treating it as satisfied would let a per-workload
    binding stand in for an anchor-wide one, which is granularity the union
    surface used by the reverse walk cannot express.
    """
    bound = extract_k8s_bound_keys(runtime_dir)
    per_manifest = [
        {k for k, sources in bound.items() if manifest in sources}
        for manifest in K8S_RUNTIME_FAMILY_DEPLOYMENTS
    ]
    inline_on_every_runtime_workload: set[str] = (
        set.intersection(*per_manifest) if per_manifest else set()
    )
    return (
        extract_configmap_keys(runtime_dir / "configmap.yaml")
        | inline_on_every_runtime_workload
    )


def extract_dockerfile_baked_aliases(dockerfile_path: Path) -> dict[str, str]:
    """Map each in-image path that ``Dockerfile.runtime`` COPYs FROM -> the path it copies TO.

    ``docker/Dockerfile.runtime`` bakes the packaged ``routing_tiers.yaml`` out
    of the installed venv into a stable, interpreter-version-free location
    (OMN-15645). The compose lanes therefore legitimately pin a DIFFERENT
    literal than the onex-dev k8s Deployments, which pin the venv source path —
    both name the same file content.

    Parsed from the Dockerfile rather than hardcoded as a second literal: a
    hardcoded alias table would drift from the COPY the moment either path
    moved, which is the exact failure the value lock exists to prevent.
    """
    if not dockerfile_path.exists():
        return {}

    aliases: dict[str, str] = {}
    # ``COPY --from=... <src> <dst>`` possibly spread over backslash continuations.
    text = re.sub(r"\\\s*\n\s*", " ", dockerfile_path.read_text())
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.upper().startswith("COPY "):
            continue
        operands = [
            token
            for token in stripped.split()[1:]
            if not token.startswith("--") and token.upper() != "COPY"
        ]
        if len(operands) != 2:
            continue
        source, destination = operands
        if source.startswith("/") and destination.startswith("/"):
            aliases[source] = destination
    return aliases


def resolve_baked_aliases(value: str, aliases: dict[str, str]) -> set[str]:
    """Return ``value`` plus every in-image path Dockerfile.runtime bakes it to.

    Glob segments in the COPY source (``python*``) are matched with
    :meth:`Path.match`-style semantics via :func:`fnmatch.fnmatch`, so the
    interpreter-minor wildcard in the Dockerfile lines up with the concrete
    ``python3.12`` literal the k8s manifests pin.
    """
    equivalent = {value}
    for source, destination in aliases.items():
        if source == value or fnmatch.fnmatch(value, source):
            equivalent.add(destination)
    return equivalent


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
    env = service_environment(compose_services(compose_path).get(service))
    if isinstance(env, ComposeTagged):
        env = env.value
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

    The k8s surface is the RUNTIME-FAMILY surface (OMN-15628): ConfigMap
    ``data`` keys PLUS keys bound inline on ALL THREE runtime Deployments. It
    was ConfigMap-only before, which made a key bound inline on a Deployment
    (the placement used for DELEGATION_ROUTING_TIERS_PATH and
    BIFROST_CONTRACT_PATH) read as absent even though the cluster sets it.

    Widening it to the plain union of every k8s workload's bindings would have
    been the easy fix and is WRONG: ``x-runtime-env`` is merged into all three
    runtime services at once, so a key bound inline on one workload only (e.g.
    ``ONEX_PUSH_VALIDATION_WORKROOT`` on runtime-worker) must not satisfy an
    anchor-wide claim. ``extract_k8s_runtime_family_bound_keys`` keeps that
    granularity; ``test_k8s_family_surface_requires_all_runtime_deployments``
    locks it.

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

    k8s_keys = extract_k8s_runtime_family_bound_keys(K8S_RUNTIME_DIR)
    accounted_for = k8s_keys | SECRET_KEYS | LOCAL_ONLY_KEYS | CONFIGMAP_DEBT_KEYS
    missing = compose_keys - accounted_for

    assert not missing, (
        "Keys in x-runtime-env but not bound for EVERY runtime workload in the "
        "onex-dev k8s manifests (and not in SECRET_KEYS, LOCAL_ONLY_KEYS, or "
        "CONFIGMAP_DEBT_KEYS). A key bound inline on only some runtime "
        "Deployments does not count — x-runtime-env reaches all of them:\n"
        + "\n".join(f"  {k}" for k in sorted(missing))
        + "\n\nFix: add each missing key to one of:\n"
        "  • omninode_infra/k8s/onex-dev/runtime/configmap.yaml  (preferred)\n"
        "  • SECRET_KEYS in tests/ci/test_env_parity.py           (k8s Secret source)\n"
        "  • LOCAL_ONLY_KEYS in tests/ci/test_env_parity.py       (local dev only)\n"
        "  • CONFIGMAP_DEBT_KEYS in tests/ci/test_env_parity.py   (temp — must file ticket)"
    )


@pytest.mark.ci
def test_bifrost_render_knobs_are_classified_by_live_cluster_state() -> None:
    """OMN-17502: the Bifrost debt classification must track the live cluster.

    Two facts, both read from the manifests rather than asserted in prose:

    * ``BIFROST_CONTRACT_PATH`` IS bound on the runtime-family surface, so the
      onex-dev lane renders the delegation contract at every container boot.
      The old CONFIGMAP_DEBT_KEYS comment denied this and used the denial to
      justify shipping the OMN-17150 renderer change without its cluster half.
    * ``BIFROST_LANE_OVERLAY_PATH`` is therefore a REQUIRED input on a lane that
      renders — so it is either bound on that same surface or recorded as debt,
      never both and never neither. When the omninode_infra half of OMN-17502
      binds it, this assertion fails until the debt entry is deleted; that is
      the point, so the allowlist cannot outlive the gap it describes.
    """
    if K8S_RUNTIME_DIR is None:
        pytest.skip(
            "omninode_infra not found as a sibling — set OMNINODE_INFRA_DIR to run this test"
        )

    k8s_keys = extract_k8s_runtime_family_bound_keys(K8S_RUNTIME_DIR)
    compose_keys = extract_runtime_env_keys(COMPOSE_PATH)

    assert "BIFROST_CONTRACT_PATH" in k8s_keys, (
        "BIFROST_CONTRACT_PATH is no longer bound on every onex-dev runtime "
        "Deployment. The lane then renders nothing, and the OMN-17502 "
        "correction above (and the classification of BIFROST_LANE_OVERLAY_PATH "
        "as a live gap) must be re-derived, not silently kept."
    )
    assert "BIFROST_CONTRACT_PATH" not in CONFIGMAP_DEBT_KEYS, (
        "BIFROST_CONTRACT_PATH is bound on the cluster; listing it as parity "
        "debt is a false claim (OMN-17502)."
    )

    assert "BIFROST_LANE_OVERLAY_PATH" in compose_keys
    bound = "BIFROST_LANE_OVERLAY_PATH" in k8s_keys
    recorded_as_debt = "BIFROST_LANE_OVERLAY_PATH" in CONFIGMAP_DEBT_KEYS
    assert bound != recorded_as_debt, (
        "BIFROST_LANE_OVERLAY_PATH must be either bound on the onex-dev runtime "
        "family or recorded in CONFIGMAP_DEBT_KEYS — exactly one. "
        f"bound={bound}, recorded_as_debt={recorded_as_debt}. If the manifest "
        "half of OMN-17502 has landed, delete the CONFIGMAP_DEBT_KEYS entry."
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
         the ``x-runtime-env`` anchor if every runtime service needs it), or on
         the owning docker/catalog/services/*.yaml manifest when the workload is
         a catalog service rather than an infra.yml one
      2. Add it to K8S_ONLY_KEYS with a value-backed justification if it
         describes cluster topology or a managed data plane
      3. Add it to COMPOSE_PARITY_DEBT_KEYS only if the compose binding is
         genuinely blocked (and file/cite a ticket)

    Before reaching for 2 or 3, check whether the key is already bound on a
    compose surface this walk does not read — a false positive is fixed at the
    surface, never by a classification that states something untrue.

    Keys in REPO_DELETED_FLAG_KEYS are exempt and CANNOT be classified by any of
    the three remedies above -- the dead-flag audit rejects their names anywhere
    in this tree. See that constant for why (OMN-15659).
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

    catalog_keys = extract_catalog_bound_keys(CATALOG_SERVICES_DIR)
    assert catalog_keys, (
        f"No env keys extracted from the service catalog at {CATALOG_SERVICES_DIR}. "
        "Have the manifests moved, or has the env field naming changed?"
    )

    accounted_for = (
        compose_keys
        | catalog_keys
        | K8S_ONLY_KEYS
        | COMPOSE_PARITY_DEBT_KEYS
        | REPO_DELETED_FLAG_KEYS
    )
    missing = {k: v for k, v in k8s_bound.items() if k not in accounted_for}

    assert not missing, (
        "Keys bound in the onex-dev k8s manifests but bound in NO "
        f"docker-compose service ({COMPOSE_PATH.name}) and no service-catalog "
        f"manifest ({CATALOG_SERVICES_DIR.name}/), and not classified as "
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
    in-container paths is a silent request-path break, not a boot failure.

    The value is locked on EVERY compose file that stands up a runtime container
    (:data:`COMPOSE_RUNTIME_SERVICES_BY_FILE`), not just the ``infra`` base. The
    lane-coverage test below is presence-only, so a typo'd literal in
    ``docker-compose.judge.yml`` or ``docker-compose.e2e.yml`` — the two lanes
    that inherit nothing — would otherwise satisfy both checks while pointing
    the container at a path that does not exist. That is precisely the
    "bound on both sides, different paths" failure this lock exists to catch.

    KNOWN CROSS-REPO DIVERGENCE (discovered here, tracked separately, NOT
    asserted): ``configmap.yaml`` also carries this key, pinned to the stale
    ``/app/contracts/delegation/routing_tiers.yaml``. All three runtime
    Deployments override it inline with the correct site-packages path, and
    inline ``env`` beats ``envFrom`` in Kubernetes, so the runtime family is
    unaffected — but any workload that ``envFrom``-s ``onex-runtime-config``
    WITHOUT an inline override would receive the stale path. The k8s pin used
    below is therefore the Deployment inline value, which is what those runtime
    containers actually see. Fixing the ConfigMap is an ``omninode_infra``
    change that cannot land in this repo's PR; asserting it here would leave a
    permanently-red gate on ``omnibase_infra`` for a defect it cannot fix.
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

    bound_k8s: dict[str, str] = {m: v for m, v in k8s_values.items() if v is not None}
    distinct_k8s = set(bound_k8s.values())
    assert len(distinct_k8s) == 1, (
        f"{key} is pinned to different values across the onex-dev runtime "
        f"Deployments: {bound_k8s}"
    )
    expected = next(iter(distinct_k8s))

    docker_dir = COMPOSE_PATH.parent
    compose_values: dict[str, str | None] = {}
    for filename, services in COMPOSE_RUNTIME_SERVICES_BY_FILE.items():
        compose_path = docker_dir / filename
        assert compose_path.exists(), (
            f"{filename} is listed in COMPOSE_RUNTIME_SERVICES_BY_FILE but does "
            "not exist. Update the map when a compose lane is renamed or removed."
        )
        for service in services:
            compose_values[f"{filename}::{service}"] = (
                extract_compose_service_env_value(compose_path, service, key)
            )

    # Accept either the k8s literal itself or a path Dockerfile.runtime bakes it
    # to. Both name the same file inside the image; see
    # extract_dockerfile_baked_aliases.
    aliases = extract_dockerfile_baked_aliases(docker_dir / "Dockerfile.runtime")
    acceptable = resolve_baked_aliases(expected, aliases)

    mismatched = {s: v for s, v in compose_values.items() if v not in acceptable}

    assert not mismatched, (
        f"{key} disagrees between docker-compose and the onex-dev k8s pin.\n"
        f"  k8s pin ({', '.join(K8S_RUNTIME_FAMILY_DEPLOYMENTS)}): {expected}\n"
        f"  accepted in compose (k8s pin + Dockerfile.runtime-baked aliases of it): "
        f"{sorted(acceptable)}\n"
        + "\n".join(f"  compose {s}: {v!r}" for s, v in sorted(mismatched.items()))
        + f"\n\nFix: bind {key} in each file above to one of the accepted paths "
        "(in infra.yml and judge.yml the runtime-env anchor covers every runtime "
        "service at once; e2e.yml binds on the service directly). If you meant to "
        "introduce a NEW in-image location, add the COPY to docker/Dockerfile.runtime "
        "first — this check reads the aliases from there, it does not take a literal "
        "on trust."
    )

    distinct_compose = set(compose_values.values())
    assert len(distinct_compose) == 1, (
        f"{key} is pinned to different (individually acceptable) paths across the "
        f"compose lanes: {compose_values}. Every lane builds the same "
        "docker/Dockerfile.runtime image, so they must agree on one literal — "
        "divergence here is how a lane-specific edit silently stops matching the "
        "others."
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
      with a Compose merge directive (``environment: !override`` or
      ``environment: !reset``) — assert that no service in the overlay does.
    * STANDALONE — files that layer nothing (e2e). Must bind the key directly.

    The directive check is STRUCTURAL, over the parsed service graph, not a
    regex over raw text: the previous regex matched the literal string
    ``environment: !override`` only, so it was blind to ``!reset`` (which
    severs inheritance identically) and to any reformatting of the same tag.
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
        severed = {
            name: env.tag
            for name, service in compose_services(docker_dir / filename).items()
            if isinstance((env := service_environment(service)), ComposeTagged)
        }
        assert not severed, (
            f"{filename} replaces a service's `environment` mapping wholesale with "
            "a Compose merge directive: "
            + ", ".join(f"{name} -> {tag}" for name, tag in sorted(severed.items()))
            + ". That severs the compose merge that carries "
            f"{key} (and every other x-runtime-env key) from "
            "docker-compose.infra.yml into this lane. Bind the key explicitly in "
            "this overlay, or drop the directive."
        )

    for filename in standalone:
        raw = (docker_dir / filename).read_text()
        assert re.search(rf"^\s+{key}:", raw, re.MULTILINE), (
            f"{filename} layers no base compose file, so it inherits nothing — "
            f"it must bind {key} on its runtime service directly."
        )


def _duplicate_mapping_keys(node: yaml.Node, path: str = "") -> list[str]:
    """Report ``a.b.KEY`` for every mapping key declared more than once."""
    duplicates: list[str] = []
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            name = str(getattr(key_node, "value", key_node))
            where = f"{path}.{name}" if path else name
            if name in seen:
                duplicates.append(where)
            seen.add(name)
            duplicates.extend(_duplicate_mapping_keys(value_node, where))
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            duplicates.extend(_duplicate_mapping_keys(item, f"{path}[{index}]"))
    return duplicates


@pytest.mark.ci
def test_no_duplicate_keys_in_compose_files() -> None:
    """No compose mapping declares the same key twice (OMN-15628 / OMN-15645).

    THIS IS THE CHECK THAT WOULD HAVE CAUGHT THE COLLISION. On 2026-08-02,
    omnibase_infra#2620 (OMN-15645) and #2621 (OMN-15628) each added
    ``DELEGATION_ROUTING_TIERS_PATH`` to the SAME ``x-runtime-env`` anchor with
    a DIFFERENT value, 14 minutes apart. Both PRs were individually green.
    Merged together they produced a duplicate key, and YAML last-wins silently
    elected one value — putting ``dev`` red on the parity gate below with no
    single PR having introduced the failure.

    A duplicate key is never intentional in these files and is invisible to
    every loader-based check in this module, because ``yaml.safe_load``
    collapses it before any assertion runs. It has to be caught at the NODE
    level, pre-construction, which is what this test does.
    """
    docker_dir = COMPOSE_PATH.parent
    compose_files = sorted(docker_dir.glob("docker-compose*.yml"))
    assert compose_files, f"No docker-compose*.yml files found in {docker_dir}"

    offenders: dict[str, list[str]] = {}
    for compose_file in compose_files:
        # yaml.compose stops at the node graph and constructs nothing, so
        # unlike yaml.load below it raises no S506 concern at all.
        node = yaml.compose(compose_file.read_text(), Loader=_ComposeLoader)
        if node is not None and (duplicates := _duplicate_mapping_keys(node)):
            offenders[compose_file.name] = duplicates

    assert not offenders, (
        "Compose files declare duplicate mapping keys. YAML keeps the LAST "
        "occurrence, so the earlier declaration is silently dead — an edit to it "
        "is a no-op, and two PRs that each add the same key to the same block "
        "merge into a wrong value with neither PR ever going red:\n"
        + "\n".join(
            f"  {name}: {', '.join(where)}" for name, where in sorted(offenders.items())
        )
        + "\n\nFix: keep exactly one declaration per key and delete the other."
    )


@pytest.mark.ci
def test_k8s_family_surface_requires_all_runtime_deployments(tmp_path: Path) -> None:
    """The forward k8s surface must not accept a partially-bound key.

    Regression lock for the granularity the OMN-15628 fix could have silently
    traded away. Widening the forward direction from ConfigMap-only to "bound
    anywhere in k8s" would let a key bound inline on ONE runtime Deployment
    satisfy an ``x-runtime-env`` claim that reaches all three — a strictly
    weaker assertion than the one it replaced.

    Driven against the REAL manifests, copied to a temp dir with the binding
    removed from exactly one Deployment. Asserts the two surfaces diverge in
    the expected direction: the union surface still reports the key (it is
    still bound somewhere), the family surface no longer does.

    The probe key is DERIVED, not hardcoded: it must be inline on all three
    runtime Deployments AND absent from the ConfigMap, otherwise the ConfigMap
    term would keep it in the family surface and the mutation would prove
    nothing. ``DELEGATION_ROUTING_TIERS_PATH`` itself does not qualify — see
    the ConfigMap-divergence note on
    ``test_delegation_routing_tiers_path_matches_k8s_pin``.
    """
    if K8S_RUNTIME_DIR is None:
        pytest.skip(
            "omninode_infra not found as a sibling — set OMNINODE_INFRA_DIR to run this test"
        )

    mutated_manifest = "deployment-omninode-runtime-worker.yaml"

    for source in K8S_RUNTIME_DIR.iterdir():
        if source.is_file():
            (tmp_path / source.name).write_text(source.read_text())

    configmap_keys = extract_configmap_keys(tmp_path / "configmap.yaml")
    inline_only_family_keys = sorted(
        extract_k8s_runtime_family_bound_keys(tmp_path) - configmap_keys
    )
    assert inline_only_family_keys, (
        "No key is bound inline on all three runtime Deployments while absent "
        "from the ConfigMap, so the family-vs-union distinction cannot be "
        "probed. If the manifests genuinely moved every inline binding into the "
        "ConfigMap, delete extract_k8s_runtime_family_bound_keys' inline term "
        "rather than leaving this test unable to fail."
    )
    key = inline_only_family_keys[0]

    target = tmp_path / mutated_manifest
    document = yaml.safe_load(target.read_text())
    for container in document["spec"]["template"]["spec"]["containers"]:
        container["env"] = [e for e in container.get("env") or [] if e["name"] != key]
    target.write_text(yaml.safe_dump(document, sort_keys=False))

    assert key in extract_k8s_bound_keys(tmp_path), (
        f"{key} should still appear in the UNION surface — it is still bound on "
        "the other two runtime Deployments. If this fails, the fixture mutation "
        "removed more than intended."
    )
    assert key not in extract_k8s_runtime_family_bound_keys(tmp_path), (
        f"{key} was removed from {mutated_manifest} yet the runtime-FAMILY "
        "surface still reports it as bound. extract_k8s_runtime_family_bound_keys "
        "has been widened to a union and the forward parity check is now weaker "
        "than ConfigMap-only was: a per-workload binding can stand in for an "
        "anchor-wide one (OMN-15628)."
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
