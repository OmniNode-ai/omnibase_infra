# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-8714: runtime-effects must join omnimemory-network.

When Memgraph is deployed via omnimemory/docker-compose.yml it lives on
omnimemory-network only. runtime-effects must also join that network so the
memory plugin handshake succeeds and the health probe passes.
"""

from __future__ import annotations

import pytest
import yaml

from tests.unit.docker.conftest import COMPOSE_FILE_PATH

pytestmark = [pytest.mark.unit]


def _get_service_networks(compose: dict, service_key: str) -> list[str]:
    services = compose.get("services", {})
    service = services.get(service_key, {})
    networks = service.get("networks", [])
    if isinstance(networks, dict):
        return list(networks.keys())
    return list(networks)


def test_runtime_effects_joins_omnimemory_network() -> None:
    """runtime-effects must declare omnimemory-network so it can reach Memgraph."""
    compose = yaml.safe_load(COMPOSE_FILE_PATH.read_text())
    networks = _get_service_networks(compose, "runtime-effects")
    assert "omnimemory-network" in networks, (
        "omninode-runtime-effects is missing omnimemory-network. "
        "Without it, PluginMemory cannot reach Memgraph when started via the "
        "omnimemory compose stack. Add omnimemory-network to the service's "
        "networks list and declare it as external. (OMN-8714)"
    )


def test_runtime_effects_retains_infra_network() -> None:
    """runtime-effects must still be on omnibase-infra-network."""
    compose = yaml.safe_load(COMPOSE_FILE_PATH.read_text())
    networks = _get_service_networks(compose, "runtime-effects")
    assert "omnibase-infra-network" in networks, (
        "omninode-runtime-effects lost its primary omnibase-infra-network membership."
    )


def test_omnimemory_network_declared_external() -> None:
    """omnimemory-network must be declared as external so Docker resolves it."""
    compose = yaml.safe_load(COMPOSE_FILE_PATH.read_text())
    top_networks = compose.get("networks", {})
    assert "omnimemory-network" in top_networks, (
        "omnimemory-network is not declared in the top-level networks section. "
        "Add it as an external network so Docker can resolve it. (OMN-8714)"
    )
    net_config = top_networks["omnimemory-network"]
    assert net_config is not None
    assert net_config.get("external") is True, (
        "omnimemory-network must be declared with external: true. (OMN-8714)"
    )
