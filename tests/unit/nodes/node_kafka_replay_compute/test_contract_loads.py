# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract tests for node_kafka_replay_compute."""

from __future__ import annotations

import importlib
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from omnibase_core.enums.replay.enum_replay_mode import EnumReplayMode
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.nodes.node_kafka_replay_compute.models import ModelKafkaReplayInput
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback
from tests.unit.nodes.node_kafka_replay_compute._defb_corpus import (
    FakeReplayConsumer,
    replay_cases,
)

_CONTRACT_PATH = (
    Path(__file__).resolve().parents[4]
    / "src"
    / "omnibase_infra"
    / "nodes"
    / "node_kafka_replay_compute"
    / "contract.yaml"
)


def _load_contract() -> dict[str, object]:
    with _CONTRACT_PATH.open() as f:
        loaded = yaml.safe_load(f)
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.unit
def test_contract_loads_with_compute_shape() -> None:
    contract = _load_contract()

    assert contract["name"] == "node_kafka_replay_compute"
    assert contract["node_type"] == "COMPUTE_GENERIC"
    assert contract["event_bus"] == {"subscribe_topics": [], "publish_topics": []}
    assert contract["metadata"]["transport_type"] == "KAFKA"

    input_model = contract["input_model"]
    output_model = contract["output_model"]
    definitions = contract["definitions"]
    assert isinstance(definitions, dict)
    progress_model = definitions["progress_model"]
    assert isinstance(input_model, dict)
    assert isinstance(output_model, dict)
    assert isinstance(progress_model, dict)
    assert input_model["name"] == "ModelKafkaReplayInput"
    assert output_model["name"] == "ModelKafkaReplayOutput"
    assert progress_model["name"] == "ModelKafkaReplayProgress"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_section", "class_name"),
    [
        ("input_model", "ModelKafkaReplayInput"),
        ("output_model", "ModelKafkaReplayOutput"),
        ("definitions.progress_model", "ModelKafkaReplayProgress"),
    ],
)
def test_contract_model_imports(model_section: str, class_name: str) -> None:
    contract = _load_contract()
    section: object = contract
    for part in model_section.split("."):
        assert isinstance(section, dict)
        section = section[part]
    assert isinstance(section, dict)

    module = importlib.import_module(str(section["module"]))
    assert getattr(module, class_name).__name__ == class_name


@pytest.mark.unit
@pytest.mark.asyncio
async def test_contract_handler_routing_uses_production_defb_wiring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contract-loaded handler must auto-wire through the real def-B adapter."""
    contract = _load_contract()
    routing = contract["handler_routing"]
    assert isinstance(routing, dict)
    handlers = routing["handlers"]
    assert isinstance(handlers, list)
    handler_ref = handlers[0]["handler"]
    assert isinstance(handler_ref, dict)

    module = importlib.import_module(str(handler_ref["module"]))
    handler_cls = getattr(module, str(handler_ref["name"]))
    case = replay_cases()[0]
    monkeypatch.setattr(
        module,
        "_default_consumer_factory",
        lambda _command: FakeReplayConsumer(case.records),
    )

    handler = handler_cls()
    callback = _make_dispatch_callback(handler, None)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=case.command.model_dump(mode="json"),
        correlation_id=uuid4(),
        event_type="ModelKafkaReplayInput",
    )

    result = await callback(envelope)

    assert result is not None
    assert result.status is EnumDispatchStatus.SUCCESS
    assert len(result.output_events) == 1
    output = result.output_events[0]
    assert output.events_replayed == case.expected_events
    assert output.last_offset_per_topic == case.expected_last_offset


@pytest.mark.unit
def test_input_reuses_core_replay_mode_and_rejects_protected_runtime() -> None:
    command = ModelKafkaReplayInput(
        topics=["dispatch_worker-completed.v1"],
        target_cluster_bootstrap="localhost:19092",
        target_consumer_group="omn-10392-replay",
        mode=EnumReplayMode.REPLAYING,
    )

    assert command.mode is EnumReplayMode.REPLAYING

    with pytest.raises(ValueError, match=r"protected \.201 runtime"):
        ModelKafkaReplayInput(
            topics=["dispatch_worker-completed.v1"],
            target_cluster_bootstrap="10.0.0.201:9092",
            target_consumer_group="omn-10392-replay",
        )
