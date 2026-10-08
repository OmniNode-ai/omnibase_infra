# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Drive the C28 collector and board probe together with recorded lane data."""

from __future__ import annotations

import asyncio
import io
import json
import subprocess
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers._consumer_flow_collection import (
    walk,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    HandlerConsumerFlow,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
    HandlerDockerConsumerFlowTarget,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)
from scripts.ci import c28_consumer_flow_probe as script

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
        rows = self.recorded["kinds"]["samples"][self.sample % 4]
        self.sample += 1
        return io.BytesIO(
            json.dumps(
                {"rows": rows, "row_count": 3, "row_limit": 500, "next_cursor": None}
            ).encode()
        )

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(argv)
        if argv[0] == "fake-pytest":
            xml = Path(
                next(
                    arg.split("=", 1)[1]
                    for arg in argv
                    if arg.startswith("--junitxml=")
                )
            )
            branch = next((name for name in script.BRANCHES if name in xml.name), None)
            names = [
                script.AST_GATE_TEST,
                "counter[event_bus]",
                "counter[raw_event_projection]",
            ]
            cases = "".join(
                f'<testcase name="{name}">'
                + (
                    "<failure/>"
                    if branch
                    and (name == script.AST_GATE_TEST or f"[{branch}]" in name)
                    else ""
                )
                + "</testcase>"
                for name in names
            )
            xml.write_text(f"<testsuite>{cases}</testsuite>")
            return subprocess.CompletedProcess(argv, int(branch is not None), "", "")

        assert argv[0] == "fake-docker"
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
                    f"x topic={script.APPLIED_TOPIC} "
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
                if script.GENERIC_DLQ in argv
                else json.dumps({"value": json.dumps({"payload": {}}), "offset": 10})
            )
        else:
            raise AssertionError(argv)
        return subprocess.CompletedProcess(argv, 0, out, "")


def test_recorded_lane_runs_collection_grading_and_receipt(tmp_path: Path) -> None:
    wiring = tmp_path / script.WIRING_MODULE
    wiring.parent.mkdir(parents=True)
    original = "\n".join(
        f'def {factory}():\n    flow_counters.register("group")\n'
        for factory in script.BRANCHES.values()
    )
    wiring.write_text(original)
    lane = _RecordedLane(json.loads(FIXTURE.read_text()))
    target = HandlerDockerConsumerFlowTarget(
        runner=lane.run,
        urlopen=lane.http,
        sleep=lambda _: None,
        repo_root=tmp_path,
    )
    record = tmp_path / "c28.json"
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        base_url="http://projection.test",
        samples=2,
        sample_interval=0,
        settle_seconds=0,
        injection_wait=0,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
        record=record,
    )

    result = asyncio.run(HandlerConsumerFlow(target=target).handle(request))

    receipt = json.loads(record.read_text())
    assert result.outcome == "PASS", result.reasons
    assert result.check_id == "consumer_flow"
    assert result.as_lab_proof_check_result().passed
    assert receipt["result"] == result.model_dump(mode="json")
    assert receipt["observation"]["read_ok"] is True
    assert receipt["observation"]["boot"]["injection"]["offset"] == 42
    assert wiring.read_text() == original
    assert lane.sample >= 2
    assert any("produce" in argv for argv in lane.calls)
    assert sum(argv[0] == "fake-pytest" for argv in lane.calls) == 3


class _RedeployedLane(_RecordedLane):
    """A recorded lane whose containers are recreated right after the first read."""

    def __init__(self, recorded: dict[str, Any]) -> None:
        super().__init__(recorded)
        self.refused = False

    def http(self, url: str, *, timeout: float) -> io.BytesIO:
        if not self.refused:
            self.refused = True
            raise urllib.error.URLError(
                ConnectionRefusedError(111, "Connection refused")
            )
        return super().http(url, timeout=timeout)

    def run(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        proc = super().run(argv, **kwargs)
        if argv[0] == "fake-docker" and argv[1] == "inspect" and self.refused:
            body = json.loads(proc.stdout)
            body[0]["Id"] = f"redeployed-{argv[-1]}"
            body[0]["State"]["StartedAt"] = "2026-10-08T00:58:19Z"
            proc.stdout = json.dumps(body)
        return proc


def test_mid_run_redeploy_is_remeasured_through_the_full_handler(
    tmp_path: Path,
) -> None:
    wiring = tmp_path / script.WIRING_MODULE
    wiring.parent.mkdir(parents=True)
    wiring.write_text(
        "\n".join(
            f'def {factory}():\n    flow_counters.register("group")\n'
            for factory in script.BRANCHES.values()
        )
    )
    lane = _RedeployedLane(json.loads(FIXTURE.read_text()))
    target = HandlerDockerConsumerFlowTarget(
        runner=lane.run,
        urlopen=lane.http,
        sleep=lambda _: None,
        repo_root=tmp_path,
    )
    request = ModelConsumerFlowRequest(
        subject_lane="dev",
        docker_bin="fake-docker",
        base_url="http://projection.test",
        samples=2,
        sample_interval=0,
        settle_seconds=0,
        injection_wait=0,
        attempts=3,
        pytest_cmd="fake-pytest",
        scratch=tmp_path,
    )

    result = asyncio.run(HandlerConsumerFlow(target=target).handle(request))

    assert lane.refused
    assert result.outcome == "PASS", result.reasons


class _WindowLane:
    """A projection API serving a fixed window of full pages, as the live lane does."""

    def __init__(self, pages: int, row_limit: int) -> None:
        self.rows = [
            {"projection_cursor": str(i + 1), "consumer_group": f"g{i}"}
            for i in range(pages * row_limit)
        ]
        self.row_limit = row_limit
        self.queries: list[dict[str, str]] = []

    def page(self, query: dict[str, str]) -> dict[str, Any]:
        self.queries.append(query)
        start = int(query.get("since", "0"))
        rows = self.rows[start : start + self.row_limit]
        more = start + self.row_limit < len(self.rows)
        return {
            "rows": rows,
            "row_count": len(rows),
            "row_limit": self.row_limit,
            "next_cursor": str(start + self.row_limit) if more else None,
        }


def test_full_window_walk_proves_end_of_final_full_page() -> None:
    lane = _WindowLane(pages=4, row_limit=500)

    walked = walk(lane)  # type: ignore[arg-type]

    assert len(walked["pages"]) == 4
    assert len(walked["rows"]) == 2000
    assert walked["pages"][-1].get("end_proof") == {
        "since": "2000",
        "beyond_row_count": 0,
        "reread_next_cursor": None,
        "proven": True,
    }
    assert lane.queries[-1] == {"since": "2000"}
