# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: a second stability host advertises its own broker addresses.

A Kafka client bootstraps on one address and then connects to whatever the
broker's metadata advertises, so a second host that served the lane with the
first host's advertised addresses would redirect every client that bootstraps
on it to the first broker. ``docker-compose.stability-host-broker.yml`` is the
layer a host applies to serve the lane under its own address; the two values
come from that host's environment and the layer carries none.

Docker Compose is not needed: ``_resolve`` expands the same ``${NAME:?msg}``
form compose expands, on the layer's own command.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"
LAYER_FILE = REPO_ROOT / "docker" / "docker-compose.stability-host-broker.yml"

MEMORY_DIAL = "${STABILITY_TEST_REDPANDA_MEMORY:-12G}"
KAFKA_NAME = "STABILITY_TEST_REDPANDA_ADVERTISED_KAFKA_ADDR"
PANDAPROXY_NAME = "STABILITY_TEST_REDPANDA_ADVERTISED_PANDAPROXY_ADDR"
KAFKA_FLAG = "--advertise-kafka-addr"
PANDAPROXY_FLAG = "--advertise-pandaproxy-addr"

# Neutral example values (the reserved .invalid domain): the external listener
# names the address the host's clients bootstrap on, on the published port.
HOST_KAFKA = (
    "internal://redpanda:9092,external://broker.example.invalid:39092,"
    "lan://lan.example.invalid:39093"
)
HOST_PANDAPROXY = "internal://redpanda:8082,external://broker.example.invalid:28082"

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?([^}]*)\}")

pytestmark = pytest.mark.unit


class _SafeLoader(yaml.SafeLoader):
    """Test-local YAML loader with Docker Compose tag support."""


def _construct(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


_SafeLoader.add_constructor("!override", _construct)


def _lane_file() -> Path:
    """The lane's own overlay, found by the broker dial its command declares."""
    found = [
        path
        for path in sorted(DOCKER_DIR.glob("docker-compose*.yml"))
        if MEMORY_DIAL in path.read_text(encoding="utf-8") and path != LAYER_FILE
    ]
    assert len(found) == 1, found
    return found[0]


BASE_FILE = _lane_file()


def _load(path: Path) -> dict[str, Any]:
    loader = _SafeLoader(path.read_text(encoding="utf-8"))
    try:
        loaded = loader.get_single_data()
    finally:
        loader.dispose()
    assert isinstance(loaded, dict)
    return loaded


def _redpanda(path: Path) -> dict[str, Any]:
    loaded = _load(path)
    service = loaded["services"]["redpanda"]
    assert isinstance(service, dict)
    return service


def _command(path: Path) -> list[str]:
    return [str(part) for part in _redpanda(path)["command"]]


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _resolve(text: str, env: dict[str, str]) -> str:
    """Expand ``${NAME:?msg}``: unset or empty is a refusal that carries the message."""

    def _expand(match: re.Match[str]) -> str:
        value = env.get(match.group(1), "")
        if not value:
            raise LookupError(f"{match.group(1)}: {match.group(2)}")
        return value

    return _VAR.sub(_expand, text)


def _layer_command(env: dict[str, str]) -> list[str]:
    return [_resolve(part, env) for part in _command(LAYER_FILE)]


HOST_ENV = {KAFKA_NAME: HOST_KAFKA, PANDAPROXY_NAME: HOST_PANDAPROXY}


def test_layer_replaces_both_advertised_addresses() -> None:
    """Golden. The failing control: with no layer file this test cannot load."""
    command = _layer_command(HOST_ENV)

    assert _value_after(command, KAFKA_FLAG) == HOST_KAFKA
    assert _value_after(command, PANDAPROXY_FLAG) == HOST_PANDAPROXY
    base = _command(BASE_FILE)
    for flag in (KAFKA_FLAG, PANDAPROXY_FLAG):
        assert _value_after(command, flag) != _value_after(base, flag)


def test_layer_differs_from_the_lane_command_in_the_two_values_only() -> None:
    """Seam: the layer's command and the lane's own command cannot drift apart.

    A compose list is replaced whole, so the layer repeats the lane's command.
    Fails when either side adds, drops, reorders or retunes any other argument.
    """
    base = _command(BASE_FILE)
    layer = _command(LAYER_FILE)

    assert len(layer) == len(base)
    differing = {
        index for index, (a, b) in enumerate(zip(base, layer, strict=True)) if a != b
    }
    assert differing == {
        base.index(KAFKA_FLAG) + 1,
        base.index(PANDAPROXY_FLAG) + 1,
    }
    assert _VAR.fullmatch(layer[base.index(KAFKA_FLAG) + 1]).group(1) == KAFKA_NAME
    assert (
        _VAR.fullmatch(layer[base.index(PANDAPROXY_FLAG) + 1]).group(1)
        == PANDAPROXY_NAME
    )


def test_layer_overrides_nothing_but_the_redpanda_command() -> None:
    loaded = _load(LAYER_FILE)

    assert set(loaded) == {"services"}
    assert set(loaded["services"]) == {"redpanda"}
    assert set(loaded["services"]["redpanda"]) == {"command"}


def test_a_host_that_sets_no_value_is_refused_by_name() -> None:
    """Error chain: no silent fallback to another host's address."""
    for missing in (KAFKA_NAME, PANDAPROXY_NAME):
        env = {key: value for key, value in HOST_ENV.items() if key != missing}
        with pytest.raises(LookupError, match=missing):
            _layer_command(env)


def test_unrelated_names_do_not_move_the_address() -> None:
    """OMN-12832: a shared name leaked from a shell and re-pointed the broker."""
    env = {
        **HOST_ENV,
        "REDPANDA_ADVERTISE_HOST": "elsewhere.example.invalid",
        "STABILITY_TEST_REDPANDA_ADVERTISE_HOST": "elsewhere.example.invalid",
        "DEV_REDPANDA_ADVERTISE_HOST": "elsewhere.example.invalid",
    }
    command = _layer_command(env)

    assert _value_after(command, KAFKA_FLAG) == HOST_KAFKA
    assert not any("elsewhere" in part for part in command)


def test_the_lane_file_itself_stays_unlayered() -> None:
    """A host that does not apply the layer keeps the lane's advertised addresses."""
    text = BASE_FILE.read_text(encoding="utf-8")

    assert KAFKA_NAME not in text
    assert PANDAPROXY_NAME not in text


def test_seam_advertised_external_port_is_the_published_port() -> None:
    """Seam: advertised listener to client bootstrap.

    A client bootstraps on ``<host>:<published port>`` and is then handed the
    advertised ``external://`` address. If that port is not the published one
    the redirect lands on a port nothing serves. Read from the lane's own
    ``ports`` and ``--kafka-addr``, for the lane's value and for a host's, so
    either side changing shape fails here.
    """
    redpanda = _redpanda(BASE_FILE)
    command = _command(BASE_FILE)
    listen = re.search(
        r"external://0\.0\.0\.0:(\d+)", _value_after(command, "--kafka-addr")
    )
    assert listen is not None

    published: dict[str, str] = {}
    for entry in redpanda["ports"]:
        host_port, container_port = str(entry).rsplit(":", 2)[-2:]
        published[container_port] = host_port
    published_port = published[listen.group(1)]

    for value in (_value_after(command, KAFKA_FLAG), HOST_KAFKA):
        external = [item for item in value.split(",") if item.startswith("external://")]
        assert len(external) == 1, value
        assert external[0].endswith(f":{published_port}"), (external, published_port)


def test_every_value_keeps_the_internal_listener() -> None:
    """In-network clients reach the broker on the internal listener."""
    command = _command(BASE_FILE)

    for value in (_value_after(command, KAFKA_FLAG), HOST_KAFKA):
        assert value.startswith("internal://redpanda:9092,")
    for value in (_value_after(command, PANDAPROXY_FLAG), HOST_PANDAPROXY):
        assert value.startswith("internal://redpanda:8082,")
