# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the runtime bundle decomposition (OMN-9332).

The `runtime` bundle grew from 7 to 21 services; env validation became
all-or-nothing and blocked .201 redeploy because 5 new env vars land in
new services. Decomposition splits runtime into four sub-bundles:
  - runtime-core: 7 original services, zero new env requirements
  - runtime-integrations: Linear/CI/Slack-dependent services
  - runtime-observability-projections: projection/observability consumers
  - runtime-infrastructure: migrations, workers, autoheal
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.resolver import CatalogResolver

CATALOG_DIR = str(Path(__file__).resolve().parents[3] / "docker" / "catalog")
BUNDLES_FILE = (
    Path(__file__).resolve().parents[3] / "docker" / "catalog" / "bundles.yaml"
)

_SUB_BUNDLES = (
    "runtime-core",
    "runtime-integrations",
    "runtime-observability-projections",
    "runtime-infrastructure",
)

_RUNTIME_CORE_SERVICES = frozenset(
    {
        "omninode-runtime",
        "runtime-effects",
        "agent-actions-consumer",
        "skill-lifecycle-consumer",
        "context-audit-consumer",
        "omninode-contract-resolver",
        "intelligence-api",
    }
)

# Env vars considered "pre-growth baseline" for the runtime-core set. These
# are the vars that the 7 runtime-core services already relied on at the last
# good .201 deploy (0.34.0) AND are known to be present in `~/.omnibase/.env`
# on every operator's machine. Anything outside this set indicates drift
# that could block a redeploy.
_PRE_GROWTH_ENV_VARS = frozenset(
    {
        "POSTGRES_PASSWORD",
        "VALKEY_PASSWORD",
        "KEYCLOAK_ADMIN_CLIENT_SECRET",
        "ONEX_SERVICE_CLIENT_SECRET",
        "ONEX_REGISTRATION_AUTO_ACK",
        # LLM endpoints: present in ~/.omnibase/.env since the .201/.200
        # endpoint map was established. Runtime-core services reference them
        # but they are not "new" in the OMN-9332 sense.
        "LLM_CODER_URL",
        "LLM_CODER_FAST_URL",
        "LLM_EMBEDDING_URL",
        "LLM_DEEPSEEK_R1_URL",
        # omnidash projection DSN: present in ~/.omnibase/.env alongside
        # POSTGRES_PASSWORD; used by intelligence-api and projection consumers.
        "OMNIDASH_ANALYTICS_DB_URL",
        # core bundle's Infisical requirements are resolved transitively
        "INFISICAL_CLIENT_ID",
        "INFISICAL_CLIENT_SECRET",
        "INFISICAL_PROJECT_ID",
        "INFISICAL_ENCRYPTION_KEY",
        "INFISICAL_AUTH_SECRET",
    }
)


def _load_bundles() -> dict[str, dict[str, object]]:
    with open(BUNDLES_FILE) as f:
        return yaml.safe_load(f)


@pytest.mark.unit
def test_all_four_sub_bundles_exist() -> None:
    bundles = _load_bundles()
    for name in _SUB_BUNDLES:
        assert name in bundles, f"missing sub-bundle '{name}' in bundles.yaml"


@pytest.mark.unit
def test_runtime_core_has_the_seven_original_services() -> None:
    bundles = _load_bundles()
    declared = set(bundles["runtime-core"]["services"])
    assert declared == _RUNTIME_CORE_SERVICES, (
        f"runtime-core drift: expected {sorted(_RUNTIME_CORE_SERVICES)}, "
        f"got {sorted(declared)}"
    )


@pytest.mark.unit
def test_runtime_core_required_env_is_subset_of_pre_growth() -> None:
    """runtime-core must not demand any env var that did not exist at 0.34.0.

    This is the load-bearing test: if it fails, .201 redeploy is blocked again.
    """
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime-core"])
    drift = resolved.required_env - _PRE_GROWTH_ENV_VARS
    assert not drift, (
        f"runtime-core requires env vars not present at 0.34.0 baseline: {drift}"
    )


@pytest.mark.unit
def test_each_runtime_service_appears_in_exactly_one_sub_bundle() -> None:
    bundles = _load_bundles()
    seen: dict[str, str] = {}
    for sub in _SUB_BUNDLES:
        for svc in bundles[sub]["services"]:
            if svc in seen:
                pytest.fail(
                    f"service '{svc}' appears in both '{seen[svc]}' and '{sub}'"
                )
            seen[svc] = sub


@pytest.mark.unit
def test_union_of_sub_bundles_equals_original_runtime() -> None:
    """No service lost, no service gained during decomposition."""
    bundles = _load_bundles()
    union: set[str] = set()
    for sub in _SUB_BUNDLES:
        union.update(bundles[sub]["services"])
    original = set(bundles["runtime"]["services"])
    # The top-level `runtime` bundle, post-decomposition, is expected to carry
    # an empty `services:` list and compose via `includes`. The authoritative
    # set comes from resolving `runtime`.
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime"])
    # Exclude core-bundle services (postgres/redpanda/valkey/infisical) from
    # the comparison — those come from `includes: [core]`, not from runtime
    # decomposition.
    core_services = {"postgres", "redpanda", "valkey", "infisical", "phoenix"}
    runtime_services_resolved = resolved.service_names - core_services
    assert union == runtime_services_resolved, (
        f"decomposition drift: union={sorted(union)}, "
        f"resolved-runtime-minus-core={sorted(runtime_services_resolved)}"
    )
    assert not original, (
        "`runtime` bundle must have no direct services after decomposition — "
        "it composes via `includes` to preserve operator contract"
    )


@pytest.mark.unit
def test_runtime_bundle_composes_all_sub_bundles_via_includes() -> None:
    """Operator contract: `onex up runtime` must still bring up every service.

    We preserve this by making `runtime` a composed bundle that includes all
    four sub-bundles plus core and tracing.
    """
    bundles = _load_bundles()
    includes = set(bundles["runtime"]["includes"])
    for sub in _SUB_BUNDLES:
        assert sub in includes, f"runtime bundle missing include '{sub}'"


@pytest.mark.unit
def test_runtime_integrations_carries_linear_ci_slack_env() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime-integrations"])
    assert "CI_CALLBACK_TOKEN" in resolved.required_env
    assert "LINEAR_WEBHOOK_SECRET" in resolved.required_env
    assert "WAITLIST_NOTIFIER_SLACK_BOT_TOKEN" in resolved.required_env
    assert "WAITLIST_NOTIFIER_SLACK_CHANNEL_ID" in resolved.required_env


@pytest.mark.unit
def test_runtime_observability_projections_carries_injection_dsn() -> None:
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    resolved = resolver.resolve(bundles=["runtime-observability-projections"])
    assert (
        "OMNIBASE_INFRA_INJECTION_EFFECTIVENESS_POSTGRES_DSN" in resolved.required_env
    )


@pytest.mark.unit
def test_runtime_equivalent_to_sum_of_sub_bundles() -> None:
    """Resolving `runtime` must equal resolving all four sub-bundles plus
    core and tracing (which `runtime` pulls in transitively)."""
    resolver = CatalogResolver(catalog_dir=CATALOG_DIR)
    runtime = resolver.resolve(bundles=["runtime"])
    composed = resolver.resolve(bundles=[*_SUB_BUNDLES, "core", "tracing"])
    assert runtime.service_names == composed.service_names, (
        f"runtime resolution != sum of sub-bundles (+core +tracing):\n"
        f"  runtime-only: {sorted(runtime.service_names - composed.service_names)}\n"
        f"  sub-only:     {sorted(composed.service_names - runtime.service_names)}"
    )
