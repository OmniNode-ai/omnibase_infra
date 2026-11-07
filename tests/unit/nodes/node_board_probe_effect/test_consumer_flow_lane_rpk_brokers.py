# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every rpk call pins the internal broker while keeping credentials in-container."""

from __future__ import annotations

import shlex
import subprocess
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_lane import (
    ModelConsumerFlowLane,
)

pytestmark = pytest.mark.unit


def test_rpk_pins_internal_broker_and_preserves_credentials_and_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEV_KAFKA_SASL_PASSWORD", "host-password-must-not-be-forwarded")
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "consumed", "")

    lane = ConsumerFlowLane(
        containers=ModelConsumerFlowLane.from_compose_project("omnibase-infra"),
        docker="fake-docker",
        base_url="http://projection.test",
        runner=runner,
    )

    assert lane.rpk("topic", "consume", "t", "-o", "1:2") == "consumed"
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:6] == [
        "fake-docker",
        "exec",
        "-i",
        "omnibase-infra-redpanda",
        "sh",
        "-c",
    ]
    script = argv[6]
    rpk_args = shlex.split(script)
    broker_index = rpk_args.index("brokers=redpanda:9092")
    assert rpk_args[broker_index - 1] == "-X"
    # The credential rides rpk's environment; a -X pass= flag would be expanded
    # into the in-container rpk argv, where `ps` shows it to every user (OMN-17427).
    # OMN-21485: SASL is set only when the broker container carries a login.
    assert script.startswith('if [ -n "${DEV_KAFKA_SASL_USERNAME:-}" ]; then ')
    assert 'export RPK_USER="$DEV_KAFKA_SASL_USERNAME"' in script
    assert 'RPK_PASS="$DEV_KAFKA_SASL_PASSWORD"' in script
    assert "RPK_SASL_MECHANISM=SCRAM-SHA-256" in script
    assert 'rpk "$@" ' in script
    assert "-X pass" not in script
    assert "-X user" not in script
    assert script.count("$DEV_KAFKA_SASL_PASSWORD") == 1
    assert "host-password-must-not-be-forwarded" not in " ".join(argv)
    assert all("DEV_KAFKA_SASL_PASSWORD" not in arg for arg in argv[:6] + argv[7:])
    assert argv[7:] == ["rpk", "topic", "consume", "t", "-o", "1:2"]


@pytest.mark.parametrize(
    ("username", "expected"),
    [("", "user= mech="), ("lane-user", "user=lane-user mech=SCRAM-SHA-256")],
)
def test_rpk_sets_sasl_only_when_the_broker_carries_a_login(
    username: str, expected: str
) -> None:
    """OMN-21485: the stability broker takes no SCRAM login; dev's does."""
    calls: list[list[str]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    ConsumerFlowLane(
        containers=ModelConsumerFlowLane.from_compose_project("omnibase-infra"),
        docker="fake-docker",
        base_url="http://projection.test",
        runner=runner,
    ).rpk("topic", "list")
    script = calls[0][6]
    fake_rpk = 'rpk() { echo "user=${RPK_USER:-} mech=${RPK_SASL_MECHANISM:-}"; }; '
    shell = subprocess.run(
        ["sh", "-c", fake_rpk + script, "rpk", "topic", "list"],
        capture_output=True,
        text=True,
        check=True,
        env={"DEV_KAFKA_SASL_USERNAME": username, "PATH": "/usr/bin:/bin"},
    )
    assert shell.stdout.strip() == expected
