# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Env parity test: every x-runtime-env key has a k8s ConfigMap or Secret entry.

Ensures that every variable in the docker-compose x-runtime-env anchor is
accounted for in the k8s ConfigMap, a known Secret, or the LOCAL_ONLY_KEYS
allowlist. This prevents environment divergence between local docker-compose
and the k8s cluster.

Run with: uv run pytest tests/ci/test_env_parity.py

Ticket: OMN-4307
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
_CONFIGMAP_SUBPATH = "k8s/onex-dev/runtime/configmap.yaml"


def _resolve_configmap_path() -> Path | None:
    """Resolve the ConfigMap path, trying multiple candidate locations."""
    # Env var override takes precedence
    override = os.environ.get("OMNINODE_INFRA_DIR", "").strip()
    if override:
        candidate = Path(override) / _CONFIGMAP_SUBPATH
        if candidate.exists():
            return candidate

    # Try sibling directories relative to the repo root
    candidates: list[Path] = [
        # Direct sibling (local worktree or CI dual-checkout)
        _REPO_ROOT.parent / "omninode_infra" / _CONFIGMAP_SUBPATH,
        # Two levels up (omni_home monorepo layout: omni_home/omnibase_infra)
        _REPO_ROOT.parent.parent / "omninode_infra" / _CONFIGMAP_SUBPATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


CONFIGMAP_PATH = _resolve_configmap_path()

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
        "INFISICAL_ENCRYPTION_KEY",
        "INFISICAL_AUTH_SECRET",
        # Per-service database DSNs — contain credentials, sourced from Infisical at runtime
        "OMNIBASE_INFRA_DB_URL",  # k8s uses OMNIBASE_INFRA_DB_HOST + OMNIBASE_INFRA_DB_PORT + Secret
        "OMNIINTELLIGENCE_DB_URL",  # cross-service DSN with embedded credentials
        "OMNIBASE_INFRA_AGENT_ACTIONS_POSTGRES_DSN",
        "OMNIBASE_INFRA_SKILL_LIFECYCLE_POSTGRES_DSN",
        # Valkey auth
        "VALKEY_PASSWORD",  # injected via Infisical at runtime
        # Keycloak secrets
        "KEYCLOAK_ADMIN_CLIENT_SECRET",
        "ONEX_SERVICE_CLIENT_SECRET",
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
        "OMNICLAUDE_CONTRACTS_ROOT",  # container-internal path — may be k8s-specific
        "ONEX_REGISTRATION_AUTO_ACK",
        "USE_EVENT_ROUTING",
        # OpenTelemetry — opt-in observability (empty = disabled)
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "OTEL_TRACES_EXPORTER",
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
    keys = re.findall(r"^\s{2}([A-Z_]+):", block, re.MULTILINE)
    return set(keys)


def extract_configmap_keys(configmap_path: Path) -> set[str]:
    """Extract all keys from the ConfigMap ``data:`` section."""
    data = yaml.safe_load(configmap_path.read_text())
    return set(data.get("data", {}).keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.ci
def test_runtime_env_keys_have_k8s_entries() -> None:
    """Every x-runtime-env key is in ConfigMap, SECRET_KEYS, or LOCAL_ONLY_KEYS.

    If this test fails, a key was added to docker-compose x-runtime-env without
    a corresponding entry in the k8s ConfigMap (omninode_infra), and it is not
    registered in SECRET_KEYS (for Infisical/k8s Secret sources),
    LOCAL_ONLY_KEYS (for local-dev-only bootstrap variables), or
    CONFIGMAP_DEBT_KEYS (known gaps tracked as tech debt).

    To fix a failure, choose one of:
      1. Add the key to the k8s ConfigMap (omninode_infra/k8s/onex-dev/runtime/configmap.yaml)
      2. Add the key to SECRET_KEYS in this file if it is sourced from a k8s Secret
      3. Add the key to LOCAL_ONLY_KEYS if it is intentionally absent from k8s
      4. Add temporarily to CONFIGMAP_DEBT_KEYS if the ConfigMap update is blocked
         (but you MUST file a ticket to resolve the debt)
    """
    if CONFIGMAP_PATH is None:
        pytest.skip(
            "omninode_infra not found as a sibling — set OMNINODE_INFRA_DIR to run this test"
        )

    compose_keys = extract_runtime_env_keys(COMPOSE_PATH)
    assert compose_keys, (
        f"No x-runtime-env keys extracted from {COMPOSE_PATH}. "
        "Has the anchor been renamed or removed?"
    )

    configmap_keys = extract_configmap_keys(CONFIGMAP_PATH)
    accounted_for = configmap_keys | SECRET_KEYS | LOCAL_ONLY_KEYS | CONFIGMAP_DEBT_KEYS
    missing = compose_keys - accounted_for

    assert not missing, (
        "Keys in x-runtime-env but missing from ConfigMap "
        "(and not in SECRET_KEYS, LOCAL_ONLY_KEYS, or CONFIGMAP_DEBT_KEYS):\n"
        + "\n".join(f"  {k}" for k in sorted(missing))
        + "\n\nFix: add each missing key to one of:\n"
        "  • omninode_infra/k8s/onex-dev/runtime/configmap.yaml  (preferred)\n"
        "  • SECRET_KEYS in tests/ci/test_env_parity.py           (k8s Secret source)\n"
        "  • LOCAL_ONLY_KEYS in tests/ci/test_env_parity.py       (local dev only)\n"
        "  • CONFIGMAP_DEBT_KEYS in tests/ci/test_env_parity.py   (temp — must file ticket)"
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
