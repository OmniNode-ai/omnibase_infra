# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel hands the tenant read bus handlers their SecretResolver only where the lane maps one (OMN-23929).

``service_kernel._build_runtime_handler_dependencies`` is resolver Step 2, the
only place a handler gets a constructor argument the resolver has no provider
for. The two omnimarket tenant read bus handlers (node_tenant_savings_read
_effect and node_tenant_delegations_read_effect) take an optional
``secret_resolver``; without one they answer not_configured. These tests drive
the real builder against the real runtime policy contract, so a profile that
loses the mapping, or a builder that stops supplying it, fails here and not as
a dev lane that silently answers not_configured on every tenant read.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.models.model_runtime_policy_contract import (
    ModelRuntimePolicyContract,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver
from omnibase_infra.runtime.service_kernel import (
    TENANT_ANALYTICS_DSN_SECRET_REF,
    _build_runtime_handler_dependencies,
    _tenant_read_bus_dependencies,
)

pytestmark = pytest.mark.unit

CONTRACT_PATH = (
    Path(__file__).parents[3]
    / "contracts"
    / "services"
    / "runtime_policy.contract.yaml"
)

# The gateway block raises on a lane that maps none of these (the prod profile),
# so the builder never reaches the tenant read bus there; the builder-level
# assertions below cover only the profiles that map all of them.
_GATEWAY_KEYCLOAK_REFS = frozenset(
    {
        "gateway.attach.keycloak.issuer",
        "gateway.attach.keycloak.introspection",
        "gateway.attach.keycloak.jwks",
        "gateway.attach.keycloak.admin_client_credentials.client_id",
        "gateway.attach.keycloak.admin_client_credentials.client_secret",
    }
)


def _profile_config(tmp_path: Path, profile: str) -> Path:
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    mappings = [
        m.model_dump(mode="json")
        for m in contract.profiles[profile].secret_resolver_mappings
    ]
    path = tmp_path / f"{profile}-secret-resolver.yaml"
    path.write_text(
        yaml.safe_dump({"enable_convention_fallback": False, "mappings": mappings}),
        encoding="utf-8",
    )
    return path


def test_the_dev_profile_gives_both_handlers_a_resolver(tmp_path: Path) -> None:
    deps = _build_runtime_handler_dependencies(
        postgres_pool=None,
        kafka_bootstrap_servers=None,
        gateway_secret_resolver_config_path=_profile_config(tmp_path, "dev"),
    )
    assert deps is not None
    assert isinstance(
        deps["HandlerTenantSavingsReadBus"]["secret_resolver"], SecretResolver
    )
    assert isinstance(
        deps["HandlerTenantDelegationsReadBus"]["secret_resolver"], SecretResolver
    )
    # A separate dict per handler: the builder must not alias one entry
    # across both handler names.
    assert (
        deps["HandlerTenantSavingsReadBus"]
        is not deps["HandlerTenantDelegationsReadBus"]
    )


def _profiles_without_the_mapping(*, builds_gateway: bool = False) -> list[str]:
    contract = ModelRuntimePolicyContract.model_validate(
        yaml.safe_load(CONTRACT_PATH.read_text(encoding="utf-8"))
    )
    return sorted(
        name
        for name, profile in contract.profiles.items()
        if TENANT_ANALYTICS_DSN_SECRET_REF
        not in {m.logical_name for m in profile.secret_resolver_mappings}
        and (
            not builds_gateway
            or {m.logical_name for m in profile.secret_resolver_mappings}
            >= _GATEWAY_KEYCLOAK_REFS
        )
    )


def test_some_profile_leaves_the_analytics_dsn_unmapped() -> None:
    assert _profiles_without_the_mapping()


@pytest.mark.parametrize("profile", _profiles_without_the_mapping(builds_gateway=True))
def test_profiles_without_the_mapping_leave_the_handlers_closed(
    tmp_path: Path, profile: str
) -> None:
    deps = _build_runtime_handler_dependencies(
        postgres_pool=None,
        kafka_bootstrap_servers=None,
        gateway_secret_resolver_config_path=_profile_config(tmp_path, profile),
    )
    assert deps is None or (
        "HandlerTenantSavingsReadBus" not in deps
        and "HandlerTenantDelegationsReadBus" not in deps
    )


@pytest.mark.parametrize("profile", _profiles_without_the_mapping())
def test_an_unmapped_lane_gets_no_resolver_and_says_why(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    profile: str,
) -> None:
    # Even an ambient secret must not open a lane with no explicit mapping.
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "ambient-value")
    config_path = _profile_config(tmp_path, profile)
    with caplog.at_level(logging.INFO, logger="omnibase_infra.runtime.service_kernel"):
        dependencies = _tenant_read_bus_dependencies(config_path)
    assert dependencies is None
    record = next(
        r for r in caplog.records if "no secret-resolver mapping" in r.message
    )
    assert record.msg == (
        "Tenant read bus: no secret-resolver mapping found for the "
        "analytics database secret on this lane; the handlers answer "
        "not_configured"
    )
    assert record.args == ()


def test_a_lane_that_maps_no_secret_at_all_gets_no_resolver(tmp_path: Path) -> None:
    # This is what the dev profile would read like if the mapping were
    # removed: convention fallback off and an empty mapping list.
    config_path = tmp_path / "empty-secret-resolver.yaml"
    config_path.write_text(
        yaml.safe_dump({"enable_convention_fallback": False, "mappings": []}),
        encoding="utf-8",
    )
    assert _tenant_read_bus_dependencies(config_path) is None


def test_the_logical_name_is_the_one_the_omnimarket_handlers_resolve() -> None:
    assert TENANT_ANALYTICS_DSN_SECRET_REF == "omnidash.analytics.db_url"
