# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests proving delegation path reads Kafka bootstrap from contract, not env (OMN-10924).

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


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_delegation_runtime_does_not_require_kafka_bootstrap_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DelegationProfileConfigLoader provides bootstrap_servers without KAFKA_BOOTSTRAP_SERVERS."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.delenv("KAFKA_BROKER_ALLOWLIST", raising=False)

    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    bus_config = loader.event_bus_config()
    assert bus_config.bootstrap_servers == ["redpanda:9092"]


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_plugin_delegation_reads_bootstrap_from_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PluginDelegation exposes bootstrap_servers from contract after initialize."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    plugin = PluginDelegation(contract_path=FIXTURE)
    assert plugin.contract_bootstrap_servers == ["redpanda:9092"]


@pytest.mark.unit
def test_plugin_delegation_bootstrap_servers_empty_without_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """PluginDelegation returns empty list when contract file is missing."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    from omnibase_infra.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    plugin = PluginDelegation(contract_path=tmp_path / "nonexistent.yaml")
    assert plugin.contract_bootstrap_servers == []


@pytest.mark.unit
def test_fixture_yaml_contains_bootstrap_servers() -> None:
    """Contract fixture YAML declares bootstrap_servers at the expected key path."""
    raw = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    event_bus = raw.get("event_bus", {})
    assert isinstance(event_bus, dict)
    bootstrap_servers = event_bus.get("bootstrap_servers", [])
    assert isinstance(bootstrap_servers, list)
    assert len(bootstrap_servers) > 0
    assert all(":" in s for s in bootstrap_servers), (
        "Each bootstrap server must be host:port format"
    )


@pytest.mark.unit
@_NEEDS_OMN_10919
def test_contract_bootstrap_servers_are_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bootstrap servers from contract are strings, compatible with EventBusKafka."""
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)

    from omnibase_infra.runtime.delegation_profile_config_loader import (
        DelegationProfileConfigLoader,
    )

    loader = DelegationProfileConfigLoader(contract_path=FIXTURE)
    bus_config = loader.event_bus_config()
    servers = bus_config.bootstrap_servers
    assert isinstance(servers, list)
    assert all(isinstance(s, str) for s in servers)
    assert all(":" in s for s in servers), "Each server must be host:port format"
