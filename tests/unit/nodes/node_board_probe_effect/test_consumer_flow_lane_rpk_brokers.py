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
        docker="fake-docker", base_url="http://projection.test", runner=runner
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
    assert script.startswith('RPK_USER="$DEV_KAFKA_SASL_USERNAME" ')
    assert 'RPK_PASS="$DEV_KAFKA_SASL_PASSWORD"' in script
    assert "RPK_SASL_MECHANISM=SCRAM-SHA-256" in script
    assert 'rpk "$@" ' in script
    assert "-X pass" not in script
    assert "-X user" not in script
    assert script.count("$DEV_KAFKA_SASL_PASSWORD") == 1
    assert "host-password-must-not-be-forwarded" not in " ".join(argv)
    assert all("DEV_KAFKA_SASL_PASSWORD" not in arg for arg in argv[:6] + argv[7:])
    assert argv[7:] == ["rpk", "topic", "consume", "t", "-o", "1:2"]
