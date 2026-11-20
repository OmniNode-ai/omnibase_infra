# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: a node with no event_bus overlay is reached through one shared overlay.

Most nodes ship no overlay of their own, so ``onex run-node`` was refused for every
one of them. When the node's own overlay is missing or holds no ``event_bus`` key, the
resolver now reads one shared overlay named ``run_node_default`` with the same lookup
order (the ``ONEX_SKILL_OVERLAY_ROOTS`` roots first, then the workspace config root)
and binds the node on the lane that overlay declares; the binding's overlay path is
the default file that answered:

* a node with no overlay and a default overlay ``{event_bus: {type: kafka, lane}}``
  resolves to the lane's broker with the default file as the overlay path;
* the same answer comes through the ``ONEX_SKILL_OVERLAY_ROOTS`` roots;
* a node overlay with its own ``event_bus`` wins over the default;
* a node overlay present without an ``event_bus`` key falls back to the default;
* with neither overlay the original refusal for the node stands, naming the node's
  own overlay path;
* a malformed ``event_bus`` in the node's overlay is refused naming that file and
  never falls back, even though a good default overlay exists;
* a default overlay without an ``event_bus`` key leaves the original refusal; and
* an in-memory default overlay yields an in-memory binding.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_core.models.event_bus.model_node_broker_resolution_error import (
    ModelNodeBrokerResolutionError,
)
from omnibase_infra.handlers.handler_node_broker_resolve import (
    HandlerNodeBrokerResolve,
)
from omnibase_infra.models.workspace import ModelNodeBrokerResolveRequest

pytestmark = pytest.mark.unit

_NODE = "node_default_overlay_probe_effect"
_LANE_BROKER = "lane-broker.test:19092"
_DEFAULT = "run_node_default"
_ENV_NAMES = (
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
    "OMNIBASE_PATH",
    "OMNI_HOME",
    "ONEX_SKILL_OVERLAY_ROOTS",
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A workspace config root holding a lane declaration, selected the way a deployment selects it."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    config_root = tmp_path / "config-owner"
    declaration = config_root / "config" / "lab" / "ci_bus_lanes.yaml"
    declaration.parent.mkdir(parents=True)
    declaration.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "plain": {
                        "broker": _LANE_BROKER,
                        "security_protocol": "PLAINTEXT",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(config_root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMNIBASE_PATH", str(tmp_path / "workspace"))
    return config_root


def _declare(workspace: Path, body: object, overlay_key: str = _NODE) -> Path:
    path = workspace / Path("config", "onex", "overlays") / overlay_key / "overlay.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_no_node_overlay_with_a_default_overlay_resolves_to_the_lane_broker(
    workspace: Path,
) -> None:
    default = _declare(
        workspace, {"event_bus": {"type": "kafka", "lane": "plain"}}, _DEFAULT
    )

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.bus_type == EnumEventBusType.KAFKA
    assert binding.lane == "plain"
    assert binding.bootstrap_servers == _LANE_BROKER
    assert binding.overlay_path == default


def test_the_default_overlay_answers_through_the_overlay_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same shared overlay under a declared overlay root, ahead of the
    workspace default."""
    for name in _ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "store-root"
    default = root / _DEFAULT / "overlay.yaml"
    default.parent.mkdir(parents=True)
    default.write_text(
        yaml.safe_dump({"event_bus": {"type": "kafka", "lane": "plain"}}),
        encoding="utf-8",
    )
    config_root = tmp_path / "config-owner"
    declaration = config_root / "config" / "lab" / "ci_bus_lanes.yaml"
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
    # The workspace default would answer inmemory, so a kafka binding proves the
    # roots were read ahead of the workspace config root, the node overlay order.
    _declare(config_root, {"event_bus": {"type": "inmemory"}}, _DEFAULT)
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(config_root))

    binding = HandlerNodeBrokerResolve().handle(
        ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=config_root)
    )

    assert binding.bus_type == EnumEventBusType.KAFKA
    assert binding.bootstrap_servers == _LANE_BROKER
    assert binding.overlay_path == default


def test_a_node_overlay_with_its_own_event_bus_wins_over_the_default(
    workspace: Path,
) -> None:
    node_overlay = _declare(workspace, {"event_bus": {"type": "inmemory"}})
    _declare(workspace, {"event_bus": {"type": "kafka", "lane": "plain"}}, _DEFAULT)

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.bus_type == EnumEventBusType.INMEMORY
    assert binding.overlay_path == node_overlay


def test_a_node_overlay_without_an_event_bus_key_falls_back_to_the_default(
    workspace: Path,
) -> None:
    _declare(workspace, {"comment_author_logins": ["someone"]})
    default = _declare(
        workspace, {"event_bus": {"type": "kafka", "lane": "plain"}}, _DEFAULT
    )

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.bus_type == EnumEventBusType.KAFKA
    assert binding.lane == "plain"
    assert binding.bootstrap_servers == _LANE_BROKER
    assert binding.overlay_path == default


def test_with_neither_overlay_the_original_refusal_names_the_nodes_overlay_path(
    workspace: Path,
) -> None:
    expected = workspace / Path("config", "onex", "overlays") / _NODE / "overlay.yaml"

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(expected)
    assert str(expected) in raised.value.message
    assert "the file does not exist" in raised.value.message


def test_a_malformed_node_event_bus_never_falls_back_to_a_good_default(
    workspace: Path,
) -> None:
    node_overlay = _declare(workspace, {"event_bus": {"type": "kafka"}})
    _declare(workspace, {"event_bus": {"type": "kafka", "lane": "plain"}}, _DEFAULT)

    with pytest.raises(
        ModelNodeBrokerResolutionError, match="invalid event_bus"
    ) as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(node_overlay)


def test_a_default_overlay_without_an_event_bus_key_leaves_the_refusal(
    workspace: Path,
) -> None:
    _declare(workspace, {"comment_author_logins": ["someone"]})
    _declare(workspace, {"note": "no bus here"}, _DEFAULT)
    expected = workspace / Path("config", "onex", "overlays") / _NODE / "overlay.yaml"

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(expected)
    assert "no event_bus key" in raised.value.message


def test_an_inmemory_default_overlay_yields_an_inmemory_binding(
    workspace: Path,
) -> None:
    default = _declare(workspace, {"event_bus": {"type": "inmemory"}}, _DEFAULT)

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.bus_type == EnumEventBusType.INMEMORY
    assert binding.bootstrap_servers is None
    assert binding.overlay_path == default
