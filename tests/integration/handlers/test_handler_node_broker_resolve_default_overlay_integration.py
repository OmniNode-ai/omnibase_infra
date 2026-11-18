# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24526: the registered node broker resolver binds a node with no overlay on the shared default lane.

The resolver is looked up through the ``onex.node_broker_resolvers`` entry-point
group, the way ``onex run-node`` finds it, and asked for a node that ships no
overlay of its own while a ``run_node_default`` overlay sits in the workspace
config root.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_infra.handlers.handler_node_broker_resolve import (
    HandlerNodeBrokerResolve,
)

pytestmark = pytest.mark.integration

_NODE = "node_default_overlay_integration_probe_effect"
_BROKER = "lane-broker.test:19092"


def test_registered_resolver_binds_a_node_without_overlay_on_the_default_lane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "KAFKA_BOOTSTRAP_SERVERS",
        "KAFKA_SASL_USERNAME",
        "KAFKA_SASL_PASSWORD",
        "OMNI_HOME",
        "ONEX_SKILL_OVERLAY_ROOTS",
    ):
        monkeypatch.delenv(name, raising=False)
    config_root = tmp_path / "config-owner"
    lanes = config_root / "config" / "lab" / "ci_bus_lanes.yaml"
    lanes.parent.mkdir(parents=True)
    lanes.write_text(
        yaml.safe_dump(
            {"lanes": {"plain": {"broker": _BROKER, "security_protocol": "PLAINTEXT"}}}
        ),
        encoding="utf-8",
    )
    default = (
        config_root
        / "config"
        / "onex"
        / "overlays"
        / "run_node_default"
        / "overlay.yaml"
    )
    default.parent.mkdir(parents=True)
    default.write_text(
        yaml.safe_dump({"event_bus": {"type": "kafka", "lane": "plain"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMNIBASE_PATH", str(tmp_path / "workspace"))

    registered = [ep.load() for ep in entry_points(group="onex.node_broker_resolvers")]
    assert HandlerNodeBrokerResolve in registered

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.bus_type == EnumEventBusType.KAFKA
    assert binding.bootstrap_servers == _BROKER
    assert binding.overlay_path == default
