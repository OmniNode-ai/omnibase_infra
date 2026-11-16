# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22362: core's resolver lookup finds the one resolver this package installs.

``onex run-node`` is refused for every node on a host whose installed omnibase_infra registers
nothing under ``onex.node_broker_resolvers``. These tests take no patch of the entry-point
lookup: they go through the installed distribution metadata, then through
``ModelNodeBrokerBinding.for_node``, the call ``onex run-node`` makes, over real overlay and lane
declaration files in a scratch workspace config root.

Golden chain: an in-memory overlay and a Redpanda overlay each resolve through the installed
registration. Error chain: a node with no overlay is refused naming the overlay path. Control: a
distribution that registers nothing leaves the lookup refused with the "no node broker resolver
is installed" message, which is the state of the merge base.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_core.models.event_bus.model_node_broker_binding import (
    NODE_BROKER_RESOLVER_ENTRY_POINT_GROUP,
    ModelNodeBrokerBinding,
)
from omnibase_core.models.event_bus.model_node_broker_resolution_error import (
    ModelNodeBrokerResolutionError,
)

_NODE = "node_installed_resolver_probe_effect"
_LANE_BROKER = "installed-lane-broker.test:19092"
_HANDLER = (
    "omnibase_infra.handlers.handler_node_broker_resolve:HandlerNodeBrokerResolve"
)


@pytest.fixture
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace config root with a lane declaration, selected the way a deployment selects it."""
    for name in (
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
        "OMNIBASE_PATH",
        "OMNI_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "config-owner"
    declaration = root / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "plain": {
                        "broker": _LANE_BROKER,
                        "security_protocol": "PLAINTEXT",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMNIBASE_PATH", str(tmp_path / "workspace"))
    return root


def _declare(root: Path, body: object) -> Path:
    path = root / Path("config", "onex", "overlays") / _NODE / "overlay.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_installed_distribution_registers_exactly_this_resolver() -> None:
    providers = list(entry_points(group=NODE_BROKER_RESOLVER_ENTRY_POINT_GROUP))

    assert [(ep.name, ep.value) for ep in providers] == [("overlay", _HANDLER)]


def test_inmemory_overlay_resolves_through_the_installed_registration(
    config_root: Path,
) -> None:
    overlay = _declare(config_root, {"event_bus": {"type": "inmemory"}})

    binding = ModelNodeBrokerBinding.for_node(_NODE)

    assert binding.bus_type == EnumEventBusType.INMEMORY
    assert binding.overlay_path == overlay
    assert binding.client_security() == {}


def test_redpanda_overlay_resolves_to_the_lane_broker_through_the_installed_registration(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _declare(config_root, {"event_bus": {"type": "kafka", "lane": "plain"}})
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "ambient-wrong-broker:1")

    binding = ModelNodeBrokerBinding.for_node(_NODE)

    assert binding.bus_type == EnumEventBusType.KAFKA
    assert binding.lane == "plain"
    assert binding.bootstrap_servers == _LANE_BROKER


def test_node_with_no_overlay_is_refused_naming_the_overlay_path(
    config_root: Path,
) -> None:
    with pytest.raises(ModelNodeBrokerResolutionError) as refused:
        ModelNodeBrokerBinding.for_node(_NODE)

    assert _NODE in str(refused.value)
    assert "overlay.yaml" in str(refused.value)


def test_control_a_distribution_registering_nothing_leaves_run_node_with_no_resolver(
    config_root: Path,
) -> None:
    _declare(config_root, {"event_bus": {"type": "inmemory"}})

    with patch(
        "omnibase_core.models.event_bus.model_node_broker_binding.entry_points",
        return_value=[],
    ):
        with pytest.raises(ModelNodeBrokerResolutionError) as refused:
            ModelNodeBrokerBinding.for_node(_NODE)

    assert "no node broker resolver is installed" in str(refused.value)
