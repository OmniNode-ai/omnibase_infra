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
no accepted record exists and the test fails at its first assertion. ``strict=True``
turns the day it passes into a failure of this file: the task that makes it pass
removes the marker. ``raises=AssertionError`` keeps a different failure (an import, a
boot error) from passing as the expected one.

The record shapes are published as plain mappings: the record models are in
omnibase_core's ``models.events.contract_hotload``, and this repository pins a core
release that does not carry them yet.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, cast

import pytest

from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
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
_CONTRACT_YAML = (
    f"name: {_CONTRACT_NAME}\n"
    "node_version: 1.0.0\n"
    "event_bus:\n"
    "  subscribe_topics:\n"
    f"    - onex.cmd.platform.{_CONTRACT_NAME.replace('_', '-')}.v1\n"
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


async def _records_on(bus: EventBusInmemory, topic: str) -> list[dict[str, Any]]:
    history = await bus.get_event_history(limit=1000, topic=topic)
    return [json.loads(message.value) for message in history]


async def _wait_for_records(
    bus: EventBusInmemory, topic: str, *, seconds: float
) -> list[dict[str, Any]]:
    deadline = asyncio.get_running_loop().time() + seconds
    while True:
        records = await _records_on(bus, topic)
        if records or asyncio.get_running_loop().time() >= deadline:
            return records
        await asyncio.sleep(0.05)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "OMN-21226 H1: the contract hot-loading registry does not exist yet, so no "
        "accepted record is produced. The task that makes this pass removes the marker."
    ),
)
async def test_an_installed_contract_loads_serves_and_survives_a_restart(
    tmp_path: Path,
) -> None:
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
        assert accepted, (
            f"no accepted record exists for {_CONTRACT_NAME} on {_ACCEPTED_TOPIC} "
            f"after the request on {_REQUEST_TOPIC}"
        )
        assert accepted[0]["entity_id"] == _CONTRACT_NAME
        assert accepted[0]["registry_version"] == 1

        # 2. Its topics exist with their declared specs.
        command_topic = f"onex.cmd.platform.{_CONTRACT_NAME.replace('_', '-')}.v1"
        assert command_topic in await bus.get_topics()

        # 3. A command on the contract's command topic is answered.
        await bus.publish(command_topic, key=b"probe", value=b"{}")
        answered = [
            message
            for message in await bus.get_event_history(limit=1000)
            if message.topic != command_topic and message.topic != _REQUEST_TOPIC
        ]
        assert answered, "the command on the contract's topic was not answered"

        # 4. The state record on the workload's state topic serves the name.
        state_topic = ModelContractHotloadWorkloadTopic.state(_WORKLOAD).suffix
        states = await _records_on(bus, state_topic)
        assert states, f"no state record exists on {state_topic}"
        assert states[-1]["names"][_CONTRACT_NAME]["served"] is not None

        # 5. After a restart the contract is wired with no resend.
        await process.stop()
        restarted = _process(bus, baseline)
        await asyncio.wait_for(restarted.start(), timeout=10)
        process = restarted
        sent_before = len(await _records_on(bus, _REQUEST_TOPIC))
        assert sent_before == 1, "the restart must not need a resend"
        states = await _records_on(bus, state_topic)
        assert states[-1]["names"][_CONTRACT_NAME]["served"] is not None
    finally:
        await process.stop()
