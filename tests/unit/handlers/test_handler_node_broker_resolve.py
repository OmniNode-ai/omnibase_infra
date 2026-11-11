# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22362: a node's broker is the ``event_bus`` its contract overlay declares.

Operator ruling 2026-11-03: "The default broker should be in the contract overlay." These are
the falsifiers for the resolver omnibase_infra registers with omnibase_core. Each drives the real
handler over real overlay and lane-declaration files in a scratch workspace config root; none
would pass at the merge base, where ``onex run-node`` read ``$KAFKA_BOOTSTRAP_SERVERS``:

* a Redpanda overlay (``type: kafka`` with a lane) resolves to the lane's declared broker and
  client transport while ``KAFKA_BOOTSTRAP_SERVERS`` names another broker;
* an in-memory overlay resolves to an in-memory binding with no transport;
* a node with no overlay, or an overlay with no ``event_bus``, is refused naming the overlay path.
"""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import EntryPoint
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from omnibase_core.enums.enum_event_bus_type import EnumEventBusType
from omnibase_core.models.event_bus.model_node_broker_binding import (
    ModelNodeBrokerBinding,
)
from omnibase_core.models.event_bus.model_node_broker_resolution_error import (
    ModelNodeBrokerResolutionError,
)
from omnibase_core.protocols.runtime.protocol_node_broker_resolver import (
    ProtocolNodeBrokerResolver,
)
from omnibase_infra.handlers.handler_node_broker_resolve import (
    HandlerNodeBrokerResolve,
)
from omnibase_infra.models.workspace import ModelNodeBrokerResolveRequest

pytestmark = pytest.mark.unit

_NODE = "node_overlay_probe_effect"
_LANE_BROKER = "lane-broker.test:19092"
_MARKER = "credential-marker-never-printed"
_ENV_NAMES = (
    "KAFKA_BOOTSTRAP_SERVERS",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
    "OMNIBASE_PATH",
    "OMNI_HOME",
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
                    "scram": {
                        "broker": _LANE_BROKER,
                        "security_protocol": "SASL_PLAINTEXT",
                        "sasl_mechanism": "SCRAM-SHA-256",
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


def _declare(workspace: Path, body: object) -> Path:
    path = workspace / Path("config", "onex", "overlays") / _NODE / "overlay.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_redpanda_overlay_resolves_to_the_lane_broker_and_ignores_the_environment(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlay = _declare(workspace, {"event_bus": {"type": "kafka", "lane": "plain"}})
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "ambient-wrong-broker:1")

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.bus_type == EnumEventBusType.KAFKA
    assert binding.lane == "plain"
    assert binding.bootstrap_servers == _LANE_BROKER
    assert binding.overlay_path == overlay
    assert binding.client_security() == {"security.protocol": "PLAINTEXT"}


def test_inmemory_overlay_resolves_to_an_inmemory_binding_with_no_transport(
    workspace: Path,
) -> None:
    overlay = _declare(workspace, {"event_bus": {"type": "inmemory"}})

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.bus_type == EnumEventBusType.INMEMORY
    assert binding.bootstrap_servers is None
    assert binding.client_security() == {}
    assert binding.overlay_path == overlay


def test_sasl_lane_on_a_host_holding_the_sasl_environment_resolves_with_that_identity(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _declare(workspace, {"event_bus": {"type": "kafka", "lane": "scram"}})
    monkeypatch.setenv("KAFKA_SASL_USERNAME", "dev-principal")
    monkeypatch.setenv("KAFKA_SASL_PASSWORD", _MARKER)

    binding = HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert binding.client_security() == {
        "security.protocol": "SASL_PLAINTEXT",
        "sasl.mechanism": "SCRAM-SHA-256",
        "sasl.username": "dev-principal",
        "sasl.password": _MARKER,
    }
    assert _MARKER not in repr(binding)


def test_sasl_lane_with_no_identity_is_refused_naming_the_overlay_never_the_secret(
    workspace: Path,
) -> None:
    overlay = _declare(workspace, {"event_bus": {"type": "kafka", "lane": "scram"}})

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(overlay)
    assert "holds no identity" in raised.value.message


def test_lane_the_declaration_does_not_bind_is_refused_naming_the_overlay(
    workspace: Path,
) -> None:
    overlay = _declare(workspace, {"event_bus": {"type": "kafka", "lane": "nowhere"}})

    with pytest.raises(ModelNodeBrokerResolutionError, match="nowhere") as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(overlay)


def test_missing_overlay_is_refused_naming_the_overlay_path_even_with_the_old_variable_set(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "ambient-wrong-broker:1")
    expected = workspace / Path("config", "onex", "overlays") / _NODE / "overlay.yaml"

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(expected)
    assert str(expected) in raised.value.message


def test_overlay_without_event_bus_is_refused_naming_the_overlay_path(
    workspace: Path,
) -> None:
    overlay = _declare(workspace, {"comment_author_logins": ["someone"]})

    with pytest.raises(ModelNodeBrokerResolutionError) as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(overlay)
    assert "no event_bus key" in raised.value.message


@pytest.mark.parametrize(
    "event_bus",
    [
        {"type": "kafka"},
        {"type": "inmemory", "lane": "plain"},
        {"type": "cloud"},
        {"type": "kafka", "lane": "plain", "bootstrap": "x:1"},
        "kafka",
    ],
)
def test_malformed_event_bus_is_refused_and_names_fields_not_values(
    workspace: Path, event_bus: object
) -> None:
    overlay = _declare(workspace, {"event_bus": event_bus})

    with pytest.raises(
        ModelNodeBrokerResolutionError, match="invalid event_bus"
    ) as raised:
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)

    assert raised.value.overlay_path == str(overlay)


def test_no_workspace_root_is_refused_without_naming_a_default(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OMNIBASE_PATH")

    with pytest.raises(ModelNodeBrokerResolutionError, match="OMNIBASE_PATH"):
        HandlerNodeBrokerResolve().resolve_node_broker(_NODE)


def test_handle_takes_the_workspace_root_from_the_request_not_the_environment(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _declare(workspace, {"event_bus": {"type": "inmemory"}})
    monkeypatch.delenv("OMNIBASE_PATH")

    binding = HandlerNodeBrokerResolve().handle(
        ModelNodeBrokerResolveRequest(overlay_key=_NODE, workspace_root=tmp_path)
    )

    assert binding.bus_type == EnumEventBusType.INMEMORY


def test_handler_satisfies_the_core_protocol() -> None:
    assert isinstance(HandlerNodeBrokerResolve(), ProtocolNodeBrokerResolver)


def test_pyproject_registers_this_handler_as_the_one_resolver() -> None:
    root = Path(__file__).resolve().parents[3]
    registered = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["entry-points"]["onex.node_broker_resolvers"]
    assert registered == {
        "overlay": "omnibase_infra.handlers.handler_node_broker_resolve:HandlerNodeBrokerResolve"
    }


def test_run_node_dials_the_overlay_lane_through_this_registered_resolver(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path: core's command, the entry-point lookup, this handler, the overlay."""
    from omnibase_core.cli.cli_run_node import run_node

    _declare(workspace, {"event_bus": {"type": "kafka", "lane": "plain"}})
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "ambient-wrong-broker:1")
    registered = EntryPoint(
        name="overlay",
        value="omnibase_infra.handlers.handler_node_broker_resolve:HandlerNodeBrokerResolve",
        group="onex.node_broker_resolvers",
    )
    with (
        patch(
            "omnibase_core.models.event_bus.model_node_broker_binding.entry_points",
            return_value=[registered],
        ),
        patch(
            "omnibase_core.cli.cli_run_node._resolve_node_topics",
            return_value=(Path("/unused"), "cmd.v1", "evt.v1", False),
        ),
        patch(
            "omnibase_core.cli.cli_run_node.publish_and_poll",
            return_value={"payload": {"status": "success"}},
        ) as dial,
    ):
        result = CliRunner().invoke(run_node, [_NODE, "--input", "{}"])

    assert result.exit_code == 0, result.output
    assert dial.call_args.kwargs["bootstrap_servers"] == _LANE_BROKER
    assert dial.call_args.kwargs["client_security"] == {
        "security.protocol": "PLAINTEXT"
    }


def test_run_node_refuses_a_node_with_no_overlay_through_this_registered_resolver(
    workspace: Path,
) -> None:
    from omnibase_core.cli.cli_run_node import run_node

    registered = EntryPoint(
        name="overlay",
        value="omnibase_infra.handlers.handler_node_broker_resolve:HandlerNodeBrokerResolve",
        group="onex.node_broker_resolvers",
    )
    with (
        patch(
            "omnibase_core.models.event_bus.model_node_broker_binding.entry_points",
            return_value=[registered],
        ),
        patch("omnibase_core.cli.cli_run_node.publish_and_poll") as dial,
    ):
        result = CliRunner().invoke(run_node, [_NODE, "--input", "{}"])

    envelope = json.loads(result.output)
    assert result.exit_code == 1
    assert envelope["overlay_path"] == str(
        workspace / Path("config", "onex", "overlays") / _NODE / "overlay.yaml"
    )
    dial.assert_not_called()


def test_binding_type_is_the_core_model() -> None:
    assert HandlerNodeBrokerResolve.resolve_node_broker.__annotations__["return"] in (
        "ModelNodeBrokerBinding",
        ModelNodeBrokerBinding,
    )
