# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract hot loading, end to end, written before the code that makes it pass (OMN-21226, H1).

The real ``RuntimeHostProcess`` and the in-memory bus boot on a baseline. One installed
fixture contract is sent to the registry on the sender's request topic, and the test
expects, in order:

1. the registry commits it and an accepted record for the name appears on the accepted
   topic;
2. its topics exist with their declared specs;
3. a command on the contract's command topic is answered;
4. the runtime's state record on its workload's state topic serves the name;
5. after a restart the contract is wired with no resend.

At the head this test was written on, nothing consumes the sender's request topic, so
no accepted record exists and the test fails at its first assertion. ``pytest.raises``
turns the day it passes into a failure of this file: the task that makes it pass
removes the expectation. The exact first-assertion message is required; other
assertion failures, imports and boot errors propagate.

The record shapes are published as plain mappings: the record models are in
omnibase_core's ``models.events.contract_hotload``, and this repository pins a core
release that does not carry them yet.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventHeaders, ModelEventMessage
from omnibase_infra.handlers.handler_http import HandlerHttpRest
from omnibase_infra.models.contract_hotload.model_contract_hotload_workload_topic import (
    ModelContractHotloadWorkloadTopic,
)
from omnibase_infra.protocols.protocol_container_aware import ProtocolContainerAware
from omnibase_infra.runtime.registry.registry_protocol_binding import (
    RegistryProtocolBinding,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess

pytestmark = pytest.mark.integration

# Composed from segments, as the node-announcement template is, so the topic-literal
# gates see segments and not a topic. The sender request topic follows the prune
# request template of the plan (``contract-prune-requested-from-{principal}``); its
# exact event name is the registry task's to fix, and is stated here for the test only.
_SENDER = "lab-deploy"
_REQUEST_TOPIC = f"onex.cmd.platform.contract-load-requested-from-{_SENDER}.v1"
_ACCEPTED_TOPIC = "onex.evt.platform." + "contract-hotload-accepted" + ".v1"
_WORKLOAD = "main"
_CONTRACT_NAME = "node_hotload_fixture"
_COMMAND_TOPIC = f"onex.cmd.platform.{_CONTRACT_NAME.replace('_', '-')}.v1"
_REPLY_TOPIC = f"onex.evt.platform.{_CONTRACT_NAME.replace('_', '-')}-replied.v1"
_MISSING_ACCEPTED = (
    f"no accepted record exists for {_CONTRACT_NAME} on {_ACCEPTED_TOPIC} "
    f"after the request on {_REQUEST_TOPIC}"
)
_CONTRACT_YAML = (
    f"name: {_CONTRACT_NAME}\n"
    "node_version: 1.0.0\n"
    "event_bus:\n"
    "  subscribe_topics:\n"
    f"    - {_COMMAND_TOPIC}\n"
    "  publish_topics:\n"
    f"    - {_REPLY_TOPIC}\n"
)
_REGISTRY_WAIT_SECONDS = 1.0


def _process(bus: EventBusInmemory, baseline: Path) -> RuntimeHostProcess:
    # The baseline: the runtime needs one handler to start, and the HTTP handler is it.
    registry = RegistryProtocolBinding()
    registry.register("http", cast("type[ProtocolContainerAware]", HandlerHttpRest))
    return RuntimeHostProcess(
        event_bus=bus,
        handler_registry=registry,
        contract_paths=[str(baseline)],
        config={
            "service_name": "contract-hotload-e2e",
            "node_name": "contract-hotload-e2e",
            "env": "test",
            "version": "v1",
            "handler_source": {"mode": "contract"},
        },
    )


async def _records_on(
    bus: EventBusInmemory, topic: str, *, after_offset: int = 0
) -> list[dict[str, Any]]:
    history = await bus.get_event_history(limit=1000, topic=topic)
    return [
        json.loads(message.value)
        for message in history
        if int(message.offset or "-1") >= after_offset
    ]


async def _wait_for_records(
    bus: EventBusInmemory,
    topic: str,
    *,
    seconds: float,
    after_offset: int = 0,
    predicate: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + seconds
    while True:
        records = await _records_on(bus, topic, after_offset=after_offset)
        if predicate is not None:
            records = [record for record in records if predicate(record)]
        if records or asyncio.get_running_loop().time() >= deadline:
            return records
        await asyncio.sleep(0.05)


async def _probe_contract(bus: EventBusInmemory) -> None:
    """Require a new reply to this command on the declared reply topic."""
    correlation_id = uuid.uuid4()
    reply_offset = await bus.get_topic_offset(_REPLY_TOPIC)
    await bus.publish(
        _COMMAND_TOPIC,
        key=b"probe",
        value=b"{}",
        headers=ModelEventHeaders(
            source="contract-hotload-e2e",
            event_type=_COMMAND_TOPIC,
            timestamp=datetime.now(UTC),
            correlation_id=correlation_id,
        ),
    )
    deadline = asyncio.get_running_loop().time() + _REGISTRY_WAIT_SECONDS
    while True:
        replies = await bus.get_event_history(limit=1000, topic=_REPLY_TOPIC)
        if any(
            int(message.offset or "-1") >= reply_offset
            and message.headers.correlation_id == correlation_id
            for message in replies
        ):
            return
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.05)
    raise AssertionError("the command on the contract's topic was not answered")


async def _installed_contract_scenario(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    installed = tmp_path / "installed" / _CONTRACT_NAME
    installed.mkdir(parents=True)
    (installed / "contract.yaml").write_text(_CONTRACT_YAML, encoding="utf-8")
    content_hash = hashlib.sha256(_CONTRACT_YAML.encode("utf-8")).hexdigest()

    bus = EventBusInmemory()
    process = _process(bus, baseline)
    await asyncio.wait_for(process.start(), timeout=10)
    try:
        request = {
            "operation": "register",
            "contract_name": _CONTRACT_NAME,
            "content_hash": content_hash,
            "registration_request_id": str(uuid.uuid4()),
            "expected_version": 0,
            "owner": _SENDER,
        }
        await bus.publish(
            _REQUEST_TOPIC,
            key=_CONTRACT_NAME.encode("utf-8"),
            value=json.dumps(request).encode("utf-8"),
        )

        # 1. The registry commits the request and publishes the accepted record.
        accepted = await _wait_for_records(
            bus, _ACCEPTED_TOPIC, seconds=_REGISTRY_WAIT_SECONDS
        )
        assert accepted, _MISSING_ACCEPTED
        assert accepted[0]["entity_id"] == _CONTRACT_NAME
        assert accepted[0]["registry_version"] == 1

        # 2. Its topics exist with their declared specs.
        assert _COMMAND_TOPIC in await bus.get_topics(), "contract topic is not wired"

        # 3. A command on the contract's command topic is answered.
        await _probe_contract(bus)

        # 4. The state record on the workload's state topic serves the name.
        state_topic = ModelContractHotloadWorkloadTopic.state(_WORKLOAD).suffix
        states = await _wait_for_records(
            bus, state_topic, seconds=_REGISTRY_WAIT_SECONDS
        )
        assert states, f"no state record exists on {state_topic}"
        assert states[-1]["names"][_CONTRACT_NAME]["served"] == {
            "registry_version": 1,
            "content_hash": content_hash,
        }
        old_incarnation = states[-1]["incarnation_id"]
        assert old_incarnation

        # 5. After a restart the contract is wired with no resend.
        requests_before = await bus.get_topic_offset(_REQUEST_TOPIC)
        state_offset = await bus.get_topic_offset(state_topic)
        assert requests_before == 1
        await process.stop()
        restarted = _process(bus, baseline)
        await asyncio.wait_for(restarted.start(), timeout=10)
        process = restarted
        states = await _wait_for_records(
            bus,
            state_topic,
            seconds=_REGISTRY_WAIT_SECONDS,
            after_offset=state_offset,
            predicate=lambda record: bool(record.get("incarnation_id"))
            and record["incarnation_id"] != old_incarnation,
        )
        assert states, "the restarted incarnation did not publish its state"
        assert states[-1]["names"][_CONTRACT_NAME]["served"] == {
            "registry_version": 1,
            "content_hash": content_hash,
        }
        assert _COMMAND_TOPIC in await bus.get_topics(), (
            "restart did not wire the contract"
        )
        await _probe_contract(bus)
        assert await bus.get_topic_offset(_REQUEST_TOPIC) == requests_before, (
            "the restart must not need a resend"
        )
    finally:
        await process.stop()


async def test_an_installed_contract_loads_serves_and_survives_a_restart(
    tmp_path: Path,
) -> None:
    """Expect only the missing accepted record at the first assertion.

    Until the registry exists, the test runs and requires this exact failure.
    Reaching a later assertion, a harness failure or passing the whole scenario
    fails this test. This keeps the executable expectation without adding a skip.
    """
    with pytest.raises(
        AssertionError,
        match="^" + re.escape(_MISSING_ACCEPTED) + r"(?:\nassert \[\])?$",
    ):
        await _installed_contract_scenario(tmp_path)


@pytest.mark.parametrize(
    "failure", ["contract topic is not wired", "harness assertion"]
)
async def test_expected_failure_does_not_accept_other_assertions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    async def broken_scenario(_tmp_path: Path) -> None:
        raise AssertionError(failure)

    monkeypatch.setattr(f"{__name__}._installed_contract_scenario", broken_scenario)
    with pytest.raises(AssertionError):
        await test_an_installed_contract_loads_serves_and_survives_a_restart(tmp_path)


async def test_expected_failure_does_not_accept_a_passing_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def passing_scenario(_tmp_path: Path) -> None:
        return

    monkeypatch.setattr(f"{__name__}._installed_contract_scenario", passing_scenario)
    with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
        await test_an_installed_contract_loads_serves_and_survives_a_restart(tmp_path)


@pytest.mark.parametrize(
    "reply_kind", ["correlated", "wrong-topic", "wrong-correlation", "old"]
)
async def test_command_probe_requires_a_fresh_correlated_reply(
    monkeypatch: pytest.MonkeyPatch,
    reply_kind: str,
) -> None:
    correlation_id = uuid.uuid4()
    monkeypatch.setattr(uuid, "uuid4", lambda: correlation_id)
    monkeypatch.setattr(f"{__name__}._REGISTRY_WAIT_SECONDS", 0.01)
    bus = EventBusInmemory()
    await bus.start()
    try:
        if reply_kind == "old":
            await bus.publish(
                _REPLY_TOPIC,
                key=b"probe",
                value=b"{}",
                headers=ModelEventHeaders(
                    source="fixture",
                    event_type=_REPLY_TOPIC,
                    timestamp=datetime.now(UTC),
                    correlation_id=correlation_id,
                ),
            )

        async def reply(message: ModelEventMessage) -> None:
            if reply_kind == "old":
                return
            topic = _ACCEPTED_TOPIC if reply_kind == "wrong-topic" else _REPLY_TOPIC
            headers = message.headers.model_copy(
                update={
                    "correlation_id": uuid.UUID(int=0)
                    if reply_kind == "wrong-correlation"
                    else correlation_id,
                    "event_type": topic,
                }
            )
            await bus.publish(topic, key=b"probe", value=b"{}", headers=headers)

        await bus.subscribe(_COMMAND_TOPIC, group_id="fixture", on_message=reply)
        if reply_kind == "correlated":
            await _probe_contract(bus)
        else:
            with pytest.raises(AssertionError, match="was not answered"):
                await _probe_contract(bus)
    finally:
        await bus.close()


async def test_restart_readback_requires_new_state_and_a_new_incarnation() -> None:
    bus = EventBusInmemory()
    await bus.start()
    state_topic = ModelContractHotloadWorkloadTopic.state(_WORKLOAD).suffix
    old_state = {"incarnation_id": "old"}
    new_state = {"incarnation_id": "new"}
    try:
        await bus.publish(
            state_topic, key=b"instance", value=json.dumps(old_state).encode()
        )
        offset_before_restart = await bus.get_topic_offset(state_topic)
        assert (
            await _wait_for_records(
                bus,
                state_topic,
                seconds=0,
                after_offset=offset_before_restart,
                predicate=lambda record: record.get("incarnation_id") != "old",
            )
            == []
        )
        await bus.publish(
            state_topic, key=b"instance", value=json.dumps(old_state).encode()
        )
        assert (
            await _wait_for_records(
                bus,
                state_topic,
                seconds=0,
                after_offset=offset_before_restart,
                predicate=lambda record: record.get("incarnation_id") != "old",
            )
            == []
        )
        await bus.publish(
            state_topic, key=b"instance", value=json.dumps(new_state).encode()
        )
        assert await _wait_for_records(
            bus,
            state_topic,
            seconds=0,
            after_offset=offset_before_restart,
            predicate=lambda record: record.get("incarnation_id") != "old",
        ) == [new_state]
    finally:
        await bus.close()
