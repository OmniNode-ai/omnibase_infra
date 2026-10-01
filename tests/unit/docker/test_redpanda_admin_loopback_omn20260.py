# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The base compose file publishes the Redpanda Admin API on loopback only (OMN-20260).

The dev lane broker runs with ``admin_api_require_auth`` false, because its
healthcheck, the deploy-runtime.sh broker probe and the rpk one-shots call the
Admin API with no credentials. Published on every host interface, that let any
LAN or tailnet host create SCRAM users and rewrite cluster config. In-network
consumers reach ``redpanda:9644`` and never needed the host publish.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
INFRA_COMPOSE = REPO_ROOT / "docker" / "docker-compose.infra.yml"
ADMIN_CONTAINER_PORT = "9644"


def _construct_compose_value(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    assert isinstance(node, yaml.ScalarNode)
    return loader.construct_scalar(node)


class _ComposeLoader(yaml.SafeLoader):
    """Test-local loader that tolerates Docker Compose merge tags."""


for _tag in ("!override", "!reset"):
    _ComposeLoader.add_constructor(_tag, _construct_compose_value)


def _admin_publishes(compose_path: Path) -> list[str]:
    compose = yaml.load(compose_path.read_text(encoding="utf-8"), Loader=_ComposeLoader)  # noqa: S506
    assert isinstance(compose, dict)
    ports = compose["services"]["redpanda"].get("ports", [])
    return [
        str(p)
        for p in ports
        if str(p).rsplit(":", 1)[-1].split("/", 1)[0] == ADMIN_CONTAINER_PORT
    ]


def _is_loopback_publish(publish: str) -> bool:
    return publish.startswith("127.0.0.1:")


@pytest.mark.unit
def test_admin_api_is_published_on_loopback_only() -> None:
    publishes = _admin_publishes(INFRA_COMPOSE)
    assert publishes, "redpanda no longer publishes its admin port; update this test"
    open_publishes = [p for p in publishes if not _is_loopback_publish(p)]
    assert not open_publishes, (
        f"redpanda admin API published on all interfaces: {open_publishes}. "
        "admin_api_require_auth is false on this broker; bind it to 127.0.0.1 (OMN-20260)."
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("publish", "loopback"),
    [
        ("${REDPANDA_ADMIN_PORT:-9644}:9644", False),
        ("0.0.0.0:9644:9644", False),
        ("127.0.0.1:${REDPANDA_ADMIN_PORT:-9644}:9644", True),
    ],
)
def test_loopback_check_refuses_an_all_interfaces_publish(
    publish: str, loopback: bool
) -> None:
    """Positive control: the predicate the guard relies on refuses the pre-fix shape."""
    assert _is_loopback_publish(publish) is loopback


@pytest.mark.unit
def test_private_ip_scan_exempts_only_a_loopback_port_publish() -> None:
    """The security scan's exemption covers a loopback port bind and nothing else."""
    from tests.unit.docker.test_docker_security import (
        LOOPBACK_PORT_PUBLISH_PATTERN,
        PRIVATE_IP_PATTERN,
    )

    publish = '    ports:\n      - "127.0.0.1:${REDPANDA_ADMIN_PORT:-9644}:9644"\n'
    assert PRIVATE_IP_PATTERN.findall(publish), "control: the raw scan flags it"
    assert not PRIVATE_IP_PATTERN.findall(
        LOOPBACK_PORT_PUBLISH_PATTERN.sub("", publish)
    )

    dependency = '    environment:\n      KAFKA_BOOTSTRAP_SERVERS: "127.0.0.1:9092"\n'
    assert PRIVATE_IP_PATTERN.findall(
        LOOPBACK_PORT_PUBLISH_PATTERN.sub("", dependency)
    ), "a loopback host dependency must still fail the scan"
