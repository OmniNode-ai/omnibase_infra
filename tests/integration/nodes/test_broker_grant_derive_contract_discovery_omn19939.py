# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Entry-point discovery derives grants for the shipped gateway (AC1 and AC4)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.event_bus.model_bus_group_describe import (
    ModelBusGroupDescribe,
)
from omnibase_infra.runtime import gateway_forwarder
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from tests.unit.nodes.node_broker_grant_derive_compute.test_broker_grant_derivation import (
    grant_keys,
    shipped_config,
)

pytestmark = [pytest.mark.integration]


def test_discovered_contract_derives_gateway_grants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discover and execute the real handler, granting inspection only on request."""
    monkeypatch.delenv("KAFKA_TOPIC_NAMESPACE", raising=False)
    monkeypatch.setenv("ONEX_ACTIVE_RUNTIME_PACKAGES", "omnibase-infra")
    node_name = "node_broker_grant_derive_compute"
    manifest = discover_contracts()
    assert not [e for e in manifest.errors if e.entry_point_name == node_name]
    discovered = [c for c in manifest.contracts if c.entry_point_name == node_name]
    assert len(discovered) == 1
    contract = discovered[0]
    assert contract.name == node_name
    assert contract.package_name == "omnibase_infra"
    assert contract.contract_path.resolve() == (
        Path(__file__).resolve().parents[3]
        / "src/omnibase_infra/nodes"
        / node_name
        / "contract.yaml"
    )
    assert contract.handler_routing is not None
    route = contract.handler_routing.handlers[0]
    assert route.operation == "broker.grant.derive"
    handler_class = getattr(
        importlib.import_module(route.handler.module), route.handler.name
    )
    raw_contract = yaml.safe_load(contract.contract_path.read_text())
    input_ref = raw_contract["input_model"]
    input_class = getattr(
        importlib.import_module(input_ref["module"]), input_ref["name"]
    )

    config = shipped_config(tmp_path)
    resolved, _ = gateway_forwarder.build_gateway_transports(config)
    request = input_class.model_validate_json(resolved.model_dump_json())
    handler = handler_class()
    result = handler.handle(request)
    grants = {
        (g.broker, g.resource_type, g.resource, g.operation) for g in result.grants
    }
    assert grants == grant_keys(resolved)
    assert result.principal == config.forwarder.tenant_identity.principal_id
    assert config.forwarder.cloud_bus is not None
    cloud = config.forwarder.cloud_bus.cloud_broker_ref
    slug = config.forwarder.tenant_identity.tenant_slug
    webhook = f"tenant-{slug}.onex.cmd.github.webhook-delivery.v1"
    inbound = f"tenant-{slug}-gateway-forwarder-inbound"
    outbound = f"tenant-{slug}-gateway-forwarder-outbound"
    assert (cloud, "TOPIC", webhook, "READ") in grants
    assert (cloud, "TOPIC", webhook, "DESCRIBE") in grants
    assert (cloud, "GROUP", inbound, "READ") in grants
    assert not any(g[1] == "GROUP" and g[3] == "DESCRIBE" for g in grants)

    inspected = resolved.model_copy(
        update={
            "described_groups": (
                ModelBusGroupDescribe(broker=cloud, group=inbound),
                ModelBusGroupDescribe(broker="local", group=outbound),
            )
        }
    )
    inspected_result = handler.handle(
        input_class.model_validate_json(inspected.model_dump_json())
    )
    inspected_grants = {
        (g.broker, g.resource_type, g.resource, g.operation)
        for g in inspected_result.grants
    }
    expected_inspection = {
        (cloud, "GROUP", inbound, "DESCRIBE"),
        ("local", "GROUP", outbound, "DESCRIBE"),
    }
    assert inspected_grants == grants | expected_inspection
    assert {
        g for g in inspected_grants if g[1] == "GROUP" and g[3] == "DESCRIBE"
    } == expected_inspection
