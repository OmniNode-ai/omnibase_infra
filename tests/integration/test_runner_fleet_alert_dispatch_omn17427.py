# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read job facts through the real contract dispatch and terminal publication.

Only the external gh/ssh subprocess boundary is replaced. Contract discovery,
payload binding, handlers, result application and envelope serialization run
in process, so a lost jobs/alerts field or misrouted event fails the readback.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra import nodes
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator.models.model_runner_fleet_maintain_completed_event import (
    ModelRunnerFleetMaintainCompletedEvent,
)
from omnibase_infra.nodes.node_runner_health_snapshot_effect.handlers.handler_runner_fleet_snapshot import (
    HandlerRunnerFleetSnapshot,
)
from omnibase_infra.observability.runner_health.model_runner_fleet_config import (
    ModelRunnerFleetConfig,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
    _select_dispatch_result_output_topic,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.service_dispatch_result_applier import (
    build_contract_result_applier,
)

_PREFIX = "onex.cmd.omnibase-infra."
_START = f"{_PREFIX}runner-fleet-maintain-start.v1"
_TERMINAL = "onex.evt.omnibase-infra.runner-fleet-maintain-completed.v1"
_REPO = "OmniNode-ai/omnibase_infra"


class _Process:
    returncode = 0

    def __init__(self, stdout: str) -> None:
        self.stdout = stdout.encode()

    async def communicate(self) -> tuple[bytes, bytes]:
        return self.stdout, b""


@pytest.mark.asyncio
@pytest.mark.parametrize("overdue", [False, True], ids=["below-bounds", "overdue"])
async def test_job_alerts_reach_published_terminal_event(
    monkeypatch: pytest.MonkeyPatch, overdue: bool
) -> None:
    now = datetime.now(tz=UTC)

    def timestamp(minutes: int) -> str:
        return (now - timedelta(minutes=minutes)).isoformat()

    labels = ["self-hosted", "omnibase-ci"]
    jobs = [
        {
            "id": 101,
            "name": "waiting",
            "status": "queued",
            "created_at": timestamp(31 if overdue else 29),
            "labels": labels,
        },
        {
            "id": 102,
            "name": "also waiting",
            "status": "queued",
            "created_at": timestamp(5),
            "labels": list(reversed(labels)),
        },
        {
            "id": 103,
            "name": "running",
            "status": "in_progress",
            "created_at": timestamp(400),
            "started_at": timestamp(361 if overdue else 359),
            "labels": labels,
            "runner_name": "omninode-runner-1",
        },
    ]
    calls: list[tuple[str, ...]] = []

    async def spawn(*args: object, **_kwargs: object) -> _Process:
        argv = tuple(str(arg) for arg in args)
        calls.append(argv)
        if argv[:2] == ("gh", "api"):
            endpoint = argv[2]
            if endpoint.endswith("/actions/runners"):
                return _Process(
                    "\n".join(
                        json.dumps(runner)
                        for runner in [
                            {
                                "name": "omninode-runner-1",
                                "status": "online",
                                "busy": True,
                            },
                            {
                                "name": "omninode-runner-2",
                                "status": "offline",
                                "busy": False,
                            },
                        ]
                    )
                )
            if endpoint == f"/repos/{_REPO}/actions/runs/42/jobs":
                assert "--method=GET" in argv
                return _Process(json.dumps({"jobs": jobs}))
            if endpoint == f"/repos/{_REPO}/actions/runs":
                return _Process(
                    json.dumps(
                        {
                            "id": 42,
                            "name": "CI",
                            "status": "in_progress",
                            "created_at": timestamp(400),
                            "event": "pull_request",
                        }
                    )
                )
        if argv[:3] == ("gh", "run", "list"):
            return _Process("[]")
        if argv[0] == "ssh":
            if "buildx" in argv[2]:
                return _Process("OK")
            if argv[2].startswith("{ df -P"):
                return _Process("37")
            return _Process("")
        raise AssertionError(f"Unexpected external call: {argv}")

    monkeypatch.setattr("asyncio.create_subprocess_exec", spawn)
    config = ModelRunnerFleetConfig(
        version="1.0",
        github_org="OmniNode-ai",
        runner_host="runner-host.invalid",
        runner_group="omnibase-ci",
        runner_name_prefix="omninode-runner",
        expected_count=2,
        watch_repos=(_REPO,),
    )
    root = Path(nodes.__file__).parent
    manifest = discover_contracts_from_paths(
        [
            root / node / "contract.yaml"
            for node in (
                "node_runner_fleet_maintain_orchestrator",
                "node_runner_health_snapshot_effect",
                "node_runner_fleet_health_compute",
            )
        ]
    )
    assert not manifest.errors
    routes = {}
    bus = AsyncMock(spec=ProtocolEventBusLike)
    for contract in manifest.contracts:
        assert contract.event_bus is not None and contract.handler_routing is not None
        applier = build_contract_result_applier(
            event_bus=bus,
            contract_path=contract.contract_path,
            publish_topics=contract.event_bus.publish_topics,
            output_topic=_select_dispatch_result_output_topic(contract),
            output_topic_map=load_published_events_map(contract.contract_path),
            allowed_output_topics=contract.event_bus.publish_topics,
        )
        for entry in contract.handler_routing.handlers:
            cls = getattr(
                importlib.import_module(entry.handler.module), entry.handler.name
            )
            handler = cast(
                "ProtocolHandleable",
                (
                    HandlerRunnerFleetSnapshot(config=config)
                    if cls is HandlerRunnerFleetSnapshot
                    else cls()
                ),
            )
            topics = (
                (entry.topic,) if entry.topic else contract.event_bus.subscribe_topics
            )
            for topic in topics:
                assert topic not in routes
                routes[topic] = (
                    _make_dispatch_callback(handler, entry.event_model),
                    applier,
                )

    cid = uuid4()
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload={"correlation_id": str(cid)},
        correlation_id=cid,
        event_type="ModelRunnerFleetMaintainStartCommand",
    )
    topic = _START
    hops: list[str] = []
    for _ in range(5):
        callback, applier = routes[topic]
        result = await callback(envelope)
        assert result is not None and result.status is EnumDispatchStatus.SUCCESS
        bus.publish_envelope.reset_mock()
        await applier.apply(result, correlation_id=cid)
        bus.publish_envelope.assert_awaited_once()
        published = bus.publish_envelope.await_args
        assert published is not None
        hops.append(topic)
        topic = published.kwargs["topic"]
        # Rehydrate the actual published bytes; do not hand the next handler
        # the previous handler's in-memory output object.
        envelope = ModelEventEnvelope[object].model_validate_json(
            published.kwargs["envelope"].model_dump_json()
        )
    assert topic == _TERMINAL, hops
    completed = ModelRunnerFleetMaintainCompletedEvent.model_validate(envelope.payload)
    assert completed.correlation_id == completed.verdict.correlation_id == cid
    alerts = {alert.kind.value: alert for alert in completed.verdict.alerts}
    expected = {"runner_offline"}
    if overdue:
        expected.update({"runner_stuck", "queue_age_exceeded"})
    assert set(alerts) == expected
    assert alerts["runner_offline"].subject == "omninode-runner-2"
    if overdue:
        queue = alerts["queue_age_exceeded"]
        assert queue.repo == _REPO and queue.queued_count == 2
        assert queue.labels == tuple(sorted(labels))
        assert queue.observed_seconds is not None
        assert 31 * 60 <= queue.observed_seconds < 32 * 60
        stuck = alerts["runner_stuck"]
        assert stuck.subject == "omninode-runner-1" and stuck.run_id == 42
        assert stuck.job == "running" and stuck.workflow == "CI"
        assert stuck.observed_seconds is not None
        assert 361 * 60 <= stuck.observed_seconds < 362 * 60
    assert (
        sum(argv[2].endswith("/jobs") for argv in calls if argv[:2] == ("gh", "api"))
        == 1
    )
    print(f"LAB_READBACK hops={len(hops)} topic={topic} alerts={sorted(alerts)}")
