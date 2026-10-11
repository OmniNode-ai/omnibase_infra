# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every rpk call in the C28 collector must pin the internal broker listener."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_collection import (
    observe_lane,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_constants import (
    APPLIED_TOPIC,
    BROKER_CONTAINER,
    BROKER_INTERNAL_ADDRESS,
    SEAM_DLQ,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_lane import (
    ConsumerFlowLane,
)

pytestmark = pytest.mark.integration

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "unit/nodes/node_board_probe_effect/fixtures/consumer_flow_recorded.json"
)


class _RecordedLane:
    def __init__(self, recorded: dict[str, Any]) -> None:
        self.recorded = recorded
        self.sample = 0
        self.hwm = 10
        self.envelope: dict[str, Any] = {}
        self.calls: list[list[str]] = []

    def http(self, _url: str, *, timeout: float) -> io.BytesIO:
        assert timeout == 30
        samples = self.recorded["kinds"]["samples"]
        rows = samples[self.sample % len(samples)]
        self.sample += 1
        return io.BytesIO(
            json.dumps(
                {
                    "rows": rows,
                    "row_count": len(rows),
                    "row_limit": 500,
                    "next_cursor": None,
                }
            ).encode()
        )

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv.copy())
        assert argv[0] == "docker"
        if argv[1] == "inspect":
            out = json.dumps(
                [
                    {
                        "Id": argv[-1],
                        "State": {
                            "Status": "running",
                            "StartedAt": "2026-09-26T00:00:00Z",
                            "Health": {"Status": "healthy"},
                        },
                    }
                ]
            )
        elif argv[1] == "logs":
            out = "runtime alive\n"
            if "--since" in argv:
                out += (
                    "validation error for ModelConsumerFlowStallAlertTrigger\n"
                    "metric_name=boundary_swallow_prevented dlq_routed=true "
                    f"x topic={APPLIED_TOPIC} "
                    f"correlation_id={self.envelope['correlation_id']}\n"
                )
        elif "psql" in argv[-1]:
            out = "local.omnimarket.projection_consumer_flow.consume.1.0.1\n"
        elif "describe" in argv:
            self.hwm += 1
            out = f"PARTITION HIGH-WATERMARK\n0 {self.hwm}\n"
        elif "produce" in argv:
            self.envelope = json.loads(kwargs["input"])
            out = "Produced to partition 0 at offset 42"
        elif "consume" in argv:
            out = (
                json.dumps(self.envelope)
                if SEAM_DLQ in argv
                else json.dumps({"value": json.dumps({"payload": {}}), "offset": 10})
            )
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, out, "")


def test_c28_collection_pins_internal_broker_on_every_rpk_call() -> None:
    recorded = _RecordedLane(json.loads(FIXTURE.read_text()))
    lane = ConsumerFlowLane(
        docker="docker",
        base_url="http://projection.test",
        runner=recorded.run,
        urlopen=recorded.http,
        sleep=lambda _: None,
    )

    observation = observe_lane(
        lane, samples=2, interval=0, settle_seconds=0, injection_wait=0
    )

    assert len(observation["kinds"]["samples"]) == 2
    assert observation["boot"]["injection"]["offset"] == 42
    assert observation["boot"]["injection"]["dlq_copies"] >= 1
    rpk_calls = [argv for argv in recorded.calls if "rpk" in argv]
    assert rpk_calls, "C28 collection must execute rpk through Docker"
    assert any(
        "consume" in argv and SEAM_DLQ in argv and ":" in argv[argv.index("-o") + 1]
        for argv in rpk_calls
    ), "C28 collection must exercise the offset-range DLQ consume"
    for argv in rpk_calls:
        assert argv[:6] == ["docker", "exec", "-i", BROKER_CONTAINER, "sh", "-c"]
        assert argv[7] == "rpk"
        assert f'-X brokers="{BROKER_INTERNAL_ADDRESS}"' in argv[6], argv
        assert all("ts.net" not in arg for arg in argv), argv
