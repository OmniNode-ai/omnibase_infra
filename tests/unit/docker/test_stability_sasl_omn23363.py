# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lab proof lane can require SASL on its broker, opt-in per host (OMN-23363).

The proof lane's overlay had no SASL step for its Redpanda, so the lane could not
require per-machine logins (OMN-21616). The dev lane's model (OMN-18012) is a
SCRAM-user one-shot, then a one-shot that sets the superuser and flips
``enable_sasl``, plus SASL client settings on every broker client.

The overlay is shared by every host that runs the lane, and a host whose broker
has no SASL must keep its clients on PLAINTEXT. An empty-string default is not
inert: ``apply_environment_overrides()`` takes any set variable. So the client
settings come from one host-local env file (``LANE_BROKER_AUTH_ENV_FILE``,
``/dev/null`` when unset), and both one-shots exit 0 without touching the broker
when that file sets no principal.

The overlay is located by the one-shot it declares rather than by name.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_DIR = REPO_ROOT / "docker"
AUTH_ENV_FILE_PATH = "${LANE_BROKER_AUTH_ENV_FILE:-/dev/null}"
SCRAM_USER = "lane-redpanda-scram-user"
SASL_ENABLE = "lane-redpanda-sasl-enable"
NO_AUTH_FILE_EXIT = "no LANE_BROKER_AUTH_ENV_FILE on this host; broker stays PLAINTEXT"

# Every service the lane runs that connects to its broker, plus the partition-cap
# one-shot, which needs RPK_USER/RPK_PASS once enable_sasl is true.
BROKER_CLIENTS = (
    "omninode-runtime",
    "runtime-effects",
    "projection-api",
    "runtime-worker",
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
    "tenant-projection-writer",
    "redpanda-partition-cap",
)
AUTH_KEYS = (
    "KAFKA_SECURITY_PROTOCOL",
    "KAFKA_SASL_MECHANISM",
    "KAFKA_SASL_USERNAME",
    "KAFKA_SASL_PASSWORD",
)


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


def _proof_lane_overlay() -> Path:
    declares = re.compile(rf"^  {re.escape(SASL_ENABLE)}:\s*$", re.MULTILINE)
    found = [
        path
        for path in sorted(DOCKER_DIR.glob("docker-compose.*.yml"))
        if path.name != "docker-compose.dev-lane.yml"
        and declares.search(path.read_text(encoding="utf-8"))
    ]
    assert len(found) == 1, (
        f"expected one overlay declaring {SASL_ENABLE}, found {found}"
    )
    return found[0]


def _services() -> dict[str, dict[str, object]]:
    loader = _ComposeLoader(_proof_lane_overlay().read_text(encoding="utf-8"))
    try:
        compose = loader.get_single_data()
    finally:
        loader.dispose()
    assert isinstance(compose, dict)
    services = compose["services"]
    assert isinstance(services, dict)
    return services


def _env_keys(service: dict[str, object]) -> set[str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return {str(item).split("=", 1)[0] for item in env}
    assert isinstance(env, dict)
    return {str(key) for key in env}


def _command(name: str) -> str:
    command = _services()[name]["command"]
    assert isinstance(command, list)
    return "\n".join(str(line) for line in command)


@pytest.mark.unit
@pytest.mark.parametrize("name", BROKER_CLIENTS)
def test_stability_sasl_client_reads_the_host_auth_env_file(name: str) -> None:
    assert _services()[name].get("env_file") == [
        {"path": AUTH_ENV_FILE_PATH, "required": False}
    ], f"{name} must read the lane's broker auth settings from {AUTH_ENV_FILE_PATH}"


@pytest.mark.unit
@pytest.mark.parametrize("name", BROKER_CLIENTS)
def test_stability_sasl_client_sets_no_literal_auth(name: str) -> None:
    literal = _env_keys(_services()[name]) & set(AUTH_KEYS)
    assert not literal, (
        f"{name} sets {sorted(literal)} in environment:, which overrides the host "
        "auth file and would put a PLAINTEXT host's clients on SASL or the reverse"
    )


@pytest.mark.unit
def test_stability_sasl_one_shots_are_ordered_before_the_partition_cap() -> None:
    services = _services()
    scram, enable = services[SCRAM_USER], services[SASL_ENABLE]
    assert scram["depends_on"] == {"redpanda": {"condition": "service_healthy"}}
    assert enable["depends_on"] == {
        SCRAM_USER: {"condition": "service_completed_successfully"}
    }
    cap_depends = services["redpanda-partition-cap"]["depends_on"]
    assert isinstance(cap_depends, dict)
    assert cap_depends.get(SASL_ENABLE) == {
        "condition": "service_completed_successfully"
    }
    for service in (scram, enable):
        assert service.get("env_file") == [
            {"path": AUTH_ENV_FILE_PATH, "required": False}
        ]
        assert service.get("restart") == "no"
        assert "profiles" not in service


@pytest.mark.unit
@pytest.mark.parametrize("name", [SCRAM_USER, SASL_ENABLE])
def test_stability_sasl_one_shot_is_a_no_op_without_the_auth_file(name: str) -> None:
    command = _command(name)
    guard = command.index(NO_AUTH_FILE_EXIT)
    first_rpk = command.index("/usr/bin/rpk")
    assert guard < first_rpk, (
        f"{name} must exit before any rpk call on a PLAINTEXT host"
    )


@pytest.mark.unit
def test_stability_sasl_enable_flips_the_broker_and_proves_both_directions() -> None:
    command = _command(SASL_ENABLE)
    assert "rpk cluster config set superusers" in command
    assert "rpk cluster config set enable_sasl true" in command
    assert "rpk cluster config get enable_sasl" in command
    assert "UNAUTHENTICATED client reached the Kafka listener" in command
    assert "RPK_SASL_MECHANISM=SCRAM-SHA-256" in command
    scram_command = _command(SCRAM_USER)
    assert "rpk security user create" in scram_command
    assert "--mechanism SCRAM-SHA-256" in scram_command
