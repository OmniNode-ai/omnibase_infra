# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A validator-triggered event reaches the ledger projection through the real dispatch seam (OMN-22002).

Wires the REAL ``node_validation_ledger_projection_compute`` contract through
``wire_from_manifest`` and dispatches a triggered event on its contract-declared
topic. The handler is the real ``HandlerValidatorEventLedgerProject``; nothing in
the projection is mocked, because it performs no I/O.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.enums.generated.enum_platform_topic import EnumPlatformTopic
from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.event_bus.topic_constants import derive_event_type_alias_for_topic
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_CONTRACT_NAME = "node_validation_ledger_projection_compute"


class _RecordingApplier:
    """Stands in for the kernel's intent applier and records what dispatch hands it."""

    def __init__(self) -> None:
        self.results: list[object] = []

    async def apply(self, result: object, correlation_id: object = None) -> None:
        self.results.append(result)


@pytest.mark.asyncio
async def test_triggered_event_dispatches_to_the_validator_projection_handler() -> None:
    topic = EnumPlatformTopic.EVT_VALIDATOR_TRIGGERED_V1.value
    contracts = [c for c in discover_contracts().contracts if c.name == _CONTRACT_NAME]
    assert contracts, f"discover_contracts() did not find {_CONTRACT_NAME!r}"
    engine = MessageDispatchEngine()
    applier = _RecordingApplier()
    report = await wire_from_manifest(
        manifest=ModelAutoWiringManifest(contracts=(contracts[0],), errors=()),
        dispatch_engine=engine,
        event_bus=None,
        container=MagicMock(name="container"),
        subscribe_immediately=False,
        result_appliers_by_contract={_CONTRACT_NAME: applier},
    )
    assert report.total_failed == 0, [
        (r.contract_name, r.reason) for r in report.results
    ]
    engine.freeze()

    correlation_id = uuid4()
    message = ModelEventMessage(
        topic=topic,
        value=json.dumps(
            {
                "validator_name": "ci_inheritance_check",
                "severity": "FAIL",
                "finding_count": 1,
                "locations": ["a.py:1"],
                "repository": "omnibase_infra",
                "head_sha": "f30b9f461344ba16dbec05ab21e29b40f11dbfc2",
                "correlation_id": str(correlation_id),
            }
        ).encode(),
        headers=ModelEventHeaders(
            source="omnibase_core",
            event_type=topic,
            correlation_id=correlation_id,
            timestamp=datetime(2026, 11, 2, 12, 0, tzinfo=UTC),
        ),
        offset="11",
        partition=0,
    )
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=message,
        correlation_id=correlation_id,
        event_type=derive_event_type_alias_for_topic(topic),
    )

    result = await engine.dispatch(topic=topic, envelope=envelope)

    assert result.status == EnumDispatchStatus.SUCCESS, result.error_message
    (intent,) = result.output_intents
    assert intent.intent_type == "validation_ledger.append"
    payload = intent.payload
    assert payload.repo_id == "omnibase_infra"
    assert payload.run_id == correlation_id
    assert (payload.kafka_topic, payload.kafka_partition, payload.kafka_offset) == (
        topic,
        0,
        11,
    )
