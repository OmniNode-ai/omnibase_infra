# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests proving delegation path reads LLM, bifrost, and security config from contract, not env (OMN-10925).

Tests that require omnibase_core.models.contracts.model_delegation_runtime_profile
are marked xfail until OMN-10919 (PR #1072) merges into the published package.
Tests that work at the YAML-parsing level run unconditionally.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

FIXTURE = (
    Path(__file__).parent.parent.parent.parent
    / "fixtures"
    / "delegation-runtime-profile-test.yaml"
)

_NEEDS_OMN_10919 = pytest.mark.xfail(
    reason="Requires omnibase_core model_delegation_runtime_profile (OMN-10919 / PR #1072 not yet merged)",
    strict=False,
)


# ── YAML-level tests (unconditional) ─────────────────────────────────────────


@pytest.mark.unit
def test_fixture_yaml_contains_llm_backends() -> None:
    """Contract fixture YAML declares llm_backends at the expected key path."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    llm_backends = raw.get("llm_backends", {})
    assert isinstance(llm_backends, dict)
    assert len(llm_backends) > 0, "llm_backends must have at least one entry"


@pytest.mark.unit
def test_fixture_yaml_llm_backends_have_bifrost_ref() -> None:
    """Each llm_backend entry in the fixture declares a bifrost_endpoint_ref."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    llm_backends = raw.get("llm_backends", {})
    for key, entry in llm_backends.items():
        assert isinstance(entry, dict), f"llm_backends[{key!r}] must be a dict"
        assert "bifrost_endpoint_ref" in entry, (
            f"llm_backends[{key!r}] must declare bifrost_endpoint_ref"
        )


@pytest.mark.unit
def test_fixture_yaml_contains_security_section() -> None:
    """Contract fixture YAML declares a security section."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    security = raw.get("security", {})
    assert isinstance(security, dict)
    assert len(security) > 0, "security section must not be empty"


@pytest.mark.unit
def test_fixture_yaml_security_has_endpoint_cidr_allowlist_ref() -> None:
    """Security section declares endpoint_cidr_allowlist_ref."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    security = raw.get("security", {})
    assert "endpoint_cidr_allowlist_ref" in security, (
        "security must declare endpoint_cidr_allowlist_ref"
    )


@pytest.mark.unit
def test_fixture_yaml_security_has_shared_secret_ref() -> None:
    """Security section declares shared_secret_ref."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    security = raw.get("security", {})
    assert "shared_secret_ref" in security, "security must declare shared_secret_ref"


@pytest.mark.unit
def test_fixture_yaml_security_has_broker_allowlist_ref() -> None:
    """Security section declares broker_allowlist_ref."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    security = raw.get("security", {})
    assert "broker_allowlist_ref" in security, (
        "security must declare broker_allowlist_ref"
    )


# ── DelegationProfileConfigLoader-level tests ────────────────────────────────


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_delegation_llm_backends_from_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DelegationProfileConfigLoader.llm_backend_config() returns backends without env vars."""
    monkeypatch.delenv("LLM_CODER_URL", raising=False)
    monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)

    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    llm_backends = loader.llm_backend_config()
    assert llm_backends is not None
    # The fixture declares a "default" backend
    assert "default" in llm_backends


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_delegation_security_config_from_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DelegationProfileConfigLoader.security_config() returns config without env vars."""
    monkeypatch.delenv("LLM_ENDPOINT_CIDR_ALLOWLIST", raising=False)
    monkeypatch.delenv("LOCAL_LLM_SHARED_SECRET", raising=False)

    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    security = loader.security_config()
    assert security is not None
    assert hasattr(security, "endpoint_cidr_allowlist_ref")
    assert hasattr(security, "shared_secret_ref")


# ── PluginDelegation-level tests ─────────────────────────────────────────────


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_plugin_delegation_contract_llm_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PluginDelegation.contract_llm_backends returns dict from contract."""
    monkeypatch.delenv("LLM_CODER_URL", raising=False)
    monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)

    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    plugin = PluginDelegation(contract_path=FIXTURE)
    backends = plugin.contract_llm_backends
    assert isinstance(backends, dict)
    assert len(backends) > 0


@pytest.mark.unit
def test_plugin_delegation_contract_llm_backends_empty_without_contract(
    tmp_path: Path,
) -> None:
    """PluginDelegation.contract_llm_backends returns empty dict when contract is missing."""
    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    plugin = PluginDelegation(contract_path=tmp_path / "nonexistent.yaml")
    assert plugin.contract_llm_backends == {}


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_plugin_delegation_contract_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PluginDelegation.contract_security returns security config from contract."""
    monkeypatch.delenv("LLM_ENDPOINT_CIDR_ALLOWLIST", raising=False)
    monkeypatch.delenv("LOCAL_LLM_SHARED_SECRET", raising=False)

    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    plugin = PluginDelegation(contract_path=FIXTURE)
    security = plugin.contract_security
    assert security is not None


@pytest.mark.unit
def test_plugin_delegation_contract_security_none_without_contract(
    tmp_path: Path,
) -> None:
    """PluginDelegation.contract_security returns None when contract is missing."""
    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    plugin = PluginDelegation(contract_path=tmp_path / "nonexistent.yaml")
    assert plugin.contract_security is None


# ── BifrostConfig contract-sourced loading ───────────────────────────────────


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_load_bifrost_config_from_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_bifrost_config_from_contract returns a config populated from the contract."""
    monkeypatch.delenv("LLM_CODER_URL", raising=False)
    monkeypatch.delenv("LLM_CODER_FAST_URL", raising=False)

    from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config_loader_bifrost import (
        load_bifrost_config_from_contract,
    )

    config = load_bifrost_config_from_contract(FIXTURE)
    assert config is not None
    # The fixture's llm_backends.default.bifrost_endpoint_ref is "local-bifrost"
    assert len(config.backends) > 0


@pytest.mark.unit
def test_load_bifrost_config_from_contract_returns_none_without_contract(
    tmp_path: Path,
) -> None:
    """load_bifrost_config_from_contract returns None when contract is missing."""
    from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config_loader_bifrost import (
        load_bifrost_config_from_contract,
    )

    result = load_bifrost_config_from_contract(tmp_path / "nonexistent.yaml")
    assert result is None


@pytest.mark.unit
def test_load_bifrost_config_from_env_still_works_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """load_bifrost_config_from_env remains usable as env-var fallback path."""
    monkeypatch.setenv("LLM_CODER_URL", "http://192.168.86.201:8000")

    from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config_loader_bifrost import (
        load_bifrost_config_from_env,
    )

    config = load_bifrost_config_from_env()
    assert config is not None
    assert "local-coder-30b" in config.backends
    assert config.backends["local-coder-30b"].base_url == "http://192.168.86.201:8000"
