# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19930: grade canned Docker output through both real board probe handlers."""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_forwarder_state_reader import (
    HandlerDockerForwarderStateReader,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_forwarder_refused_topic import (
    HandlerForwarderRefusedTopic,
)
from omnibase_infra.nodes.node_board_probe_effect.models import (
    EnumBoardCheckSurfaceClass,
    EnumBoardProbeOutcome,
    ModelForwarderRefusedTopicRequest,
)

NOW = datetime(2026, 9, 28, 16, 10, tzinfo=UTC)
DECLARED = "b-1.dev-cloud.example:9098"
# The runner below is a fake: no live container is read, so the lane-presence
# guard does not apply. The command head is a named constant, not a literal.
DOCKER = "docker"
WEBHOOK = "tenant-beta-gateway-canary-79afa7263852.onex.cmd.github.webhook-delivery.v1"
# Same Docker state and stderr log format as the board probe unit fixtures.
STATE = {"Running": True, "StartedAt": "2026-09-28T13:50:23.846688704Z"}
REFUSED_LOG = (
    "2026-09-28 16:06:14,919 WARNING omnibase_infra.event_bus.kafka_transport "
    f"kafka_transport_topic_refused topic={WEBHOOK} "
    "group=tenant-beta-gateway-canary-79afa7263852-gateway-forwarder-inbound "
    "error=TopicAuthorizationFailedError retry_in_seconds=300.0\n"
)


class _Runner:
    """Replace only subprocess.run, recording the reader's Docker commands."""

    def __init__(self, *, logs: str, inspect_rc: int) -> None:
        self.logs = logs
        self.inspect_rc = inspect_rc
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: Sequence[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        assert args[0] == DOCKER
        if args[1] == "inspect":
            if self.inspect_rc:
                return subprocess.CompletedProcess(
                    args, self.inspect_rc, "", "Error: No such object"
                )
            return subprocess.CompletedProcess(args, 0, json.dumps(STATE), "")
        if args[1] == "logs":
            # The forwarder emits to stderr, which docker logs preserves.
            return subprocess.CompletedProcess(args, 0, "", self.logs)
        if args[1] == "exec":
            return subprocess.CompletedProcess(
                args, 0, f"gateway.cloud.kafka.broker: {DECLARED}\n", ""
            )
        raise AssertionError(f"unexpected docker call {args}")


@pytest.mark.integration
@pytest.mark.parametrize(
    ("logs", "inspect_rc", "expected", "reason_fragment"),
    [
        pytest.param(
            REFUSED_LOG, 0, EnumBoardProbeOutcome.FAIL, WEBHOOK, id="refused-topic"
        ),
        pytest.param(
            "", 0, EnumBoardProbeOutcome.PASS, "no inbound topic refused", id="healthy"
        ),
        pytest.param(
            "",
            1,
            EnumBoardProbeOutcome.INDETERMINATE,
            "No such object",
            id="docker-read-failure",
        ),
    ],
)
def test_docker_observation_is_graded_by_the_board_probe(
    logs: str,
    inspect_rc: int,
    expected: EnumBoardProbeOutcome,
    reason_fragment: str,
) -> None:
    request = ModelForwarderRefusedTopicRequest(
        subject_lane="dev",
        forwarder_container="omninode-gateway-forwarder",
        declared_cloud_broker=DECLARED,
        cloud_broker_ref="gateway.cloud.kafka.broker",
        retry_interval_seconds=300,
    )
    runner = _Runner(logs=logs, inspect_rc=inspect_rc)
    reader = HandlerDockerForwarderStateReader(runner=runner, now=lambda: NOW)
    handler = HandlerForwarderRefusedTopic(reader=reader)

    result = asyncio.run(handler.handle(request))

    assert result.outcome is expected, result.reasons
    assert result.check_id == "forwarder_refused_topic"
    assert result.surface_class is EnumBoardCheckSurfaceClass.LAB_HARDWARE
    assert result.subject == request.subject_lane
    assert result.observed_at == NOW
    assert any(reason_fragment in reason for reason in result.reasons)
    assert result.evidence_items == (
        (WEBHOOK,) if expected is EnumBoardProbeOutcome.FAIL else ()
    )
    passed = expected is EnumBoardProbeOutcome.PASS
    assert result.status == ("success" if passed else "failure")
    assert result.as_lab_proof_check_result().passed is passed

    expected_calls = [
        [
            DOCKER,
            "inspect",
            "--format",
            "{{json .State}}",
            request.forwarder_container,
        ]
    ]
    if inspect_rc == 0:
        expected_calls.extend(
            [
                [DOCKER, "logs", "--since", "360s", request.forwarder_container],
                [
                    DOCKER,
                    "exec",
                    request.forwarder_container,
                    "cat",
                    request.broker_ref_map_path,
                ],
            ]
        )
    assert runner.calls == expected_calls

    # The contract entrypoint returns what observe() returns.
    assert asyncio.run(reader.handle(request)) == asyncio.run(reader.observe(request))


@pytest.mark.integration
def test_contract_routing_lists_both_board_probe_handlers() -> None:
    contract_path = (
        Path(__file__).resolve().parents[3]
        / "src/omnibase_infra/nodes/node_board_probe_effect/contract.yaml"
    )
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    handlers = contract["handler_routing"]["handlers"]
    routes = {
        (route["operation"], route["handler"]["module"], route["handler"]["name"])
        for route in handlers
    }
    for operation, handler in (
        ("board_probe.forwarder_refused_topic", HandlerForwarderRefusedTopic),
        ("board_probe.read_forwarder_state", HandlerDockerForwarderStateReader),
    ):
        assert (operation, handler.__module__, handler.__name__) in routes
