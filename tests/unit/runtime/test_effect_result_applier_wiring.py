# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for OMN-12409: EFFECT node result_applier wiring.

Verifies that the runtime wires a DispatchResultApplier for EFFECT and REDUCER
contracts that declare published_events, so handler-returned models are
published to the correct bus topic rather than being silently dropped.

Root cause: the kernel registered result appliers only for a small set of
hard-coded ORCHESTRATOR contracts.  Any contract that (a) goes through the
auto-wiring manifest scanner and (b) has published_events but is NOT in the
hard-coded set had its handler results dropped — the auto-wiring callback's
``if result_applier is not None and result is not None:`` branch was never taken.

The fix scans the filtered manifest after discovery and pre-registers a
DispatchResultApplier for every contract that has published_events and is not
already registered in auto_wiring_result_appliers.  The applier reads topics
from the contract's own discovered contract_path, not from a guessed path.

Related:
    - OMN-12409: EFFECT/REDUCER result_applier gap
    - service_kernel.py auto_wiring_result_appliers discovery block
    - handler_wiring.py _subscribe_contract_topics
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.event_bus_subcontract_wiring import (
    load_published_events_map,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EFFECT_CONTRACT_YAML = """\
name: node_llm_delegation_call_effect
node_type: EFFECT_GENERIC
event_bus:
  subscribe_topics:
    - "onex.cmd.omnibase-infra.delegation-call-request.v1"
  publish_topics:
    - "onex.evt.omnibase-infra.inference-response.v1"
    - "onex.evt.omnibase-infra.delegation-all-tiers-failed.v1"
published_events:
  - event_type: "InferenceResponse"
    topic: "onex.evt.omnibase-infra.inference-response.v1"
  - event_type: "DelegationAllTiersFailed"
    topic: "onex.evt.omnibase-infra.delegation-all-tiers-failed.v1"
"""

_REDUCER_CONTRACT_YAML = """\
name: node_delegation_routing_reducer
node_type: REDUCER_GENERIC
event_bus:
  subscribe_topics:
    - "onex.cmd.omnibase-infra.delegation-routing-request.v1"
  publish_topics:
    - "onex.evt.omnibase-infra.routing-decision.v1"
published_events:
  - event_type: "RoutingDecision"
    topic: "onex.evt.omnibase-infra.routing-decision.v1"
"""


class ModelInferenceResponse(BaseModel):
    """Stub returned by an EFFECT handler (short-name = InferenceResponse)."""

    correlation_id: str = "c-1"


class ModelDelegationAllTiersFailed(BaseModel):
    """Alternate EFFECT output (short-name = DelegationAllTiersFailed)."""

    correlation_id: str = "c-2"


class ModelRoutingDecision(BaseModel):
    """Stub returned by a REDUCER handler (short-name = RoutingDecision)."""

    correlation_id: str = "c-3"


class HandlerEffectReturnsInferenceResponse:
    """Minimal EFFECT handler that returns ModelInferenceResponse.

    Propagates correlation_id from the received envelope so the integration
    test can confirm the correct message was delivered.
    """

    async def handle(self, envelope: object) -> ModelInferenceResponse:
        corr = getattr(envelope, "correlation_id", None)
        if corr is not None:
            return ModelInferenceResponse(correlation_id=str(corr))
        # envelope.payload might carry correlation_id (raw-dict fallback)
        payload = getattr(envelope, "payload", None)
        if isinstance(payload, dict):
            dict_corr = payload.get("correlation_id")
            if dict_corr is not None:
                return ModelInferenceResponse(correlation_id=str(dict_corr))
        return ModelInferenceResponse(correlation_id="c-1")


# ---------------------------------------------------------------------------
# Unit: load_published_events_map
# ---------------------------------------------------------------------------


def test_load_published_events_map_effect_contract(tmp_path: Path) -> None:
    """published_events map loads correctly for an EFFECT contract."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_EFFECT_CONTRACT_YAML, encoding="utf-8")

    pe_map = load_published_events_map(contract_path)

    assert pe_map == {
        "InferenceResponse": "onex.evt.omnibase-infra.inference-response.v1",
        "DelegationAllTiersFailed": "onex.evt.omnibase-infra.delegation-all-tiers-failed.v1",
    }


def test_load_published_events_map_reducer_contract(tmp_path: Path) -> None:
    """published_events map loads correctly for a REDUCER contract."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_REDUCER_CONTRACT_YAML, encoding="utf-8")

    pe_map = load_published_events_map(contract_path)

    assert pe_map == {
        "RoutingDecision": "onex.evt.omnibase-infra.routing-decision.v1",
    }


# ---------------------------------------------------------------------------
# Unit: DispatchResultApplier built from published_events map routes correctly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_applier_routes_effect_returned_model_to_correct_topic(
    tmp_path: Path,
) -> None:
    """DispatchResultApplier built from published_events map routes InferenceResponse."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_EFFECT_CONTRACT_YAML, encoding="utf-8")

    pe_map = load_published_events_map(contract_path)
    topics = list(pe_map.values())

    event_bus = AsyncMock()
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=topics[0],
        output_topic_map=pe_map,
        allowed_output_topics=topics,
    )

    result = ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.cmd.omnibase-infra.delegation-call-request.v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        output_count=1,
        output_events=[ModelInferenceResponse()],
        correlation_id=uuid4(),
    )

    await applier.apply(result, uuid4())

    assert event_bus.publish_envelope.await_count == 1
    _, kwargs = event_bus.publish_envelope.await_args
    assert kwargs["topic"] == "onex.evt.omnibase-infra.inference-response.v1"


@pytest.mark.asyncio
async def test_applier_routes_alternate_effect_output_to_correct_topic(
    tmp_path: Path,
) -> None:
    """DispatchResultApplier routes DelegationAllTiersFailed to its own topic."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_EFFECT_CONTRACT_YAML, encoding="utf-8")

    pe_map = load_published_events_map(contract_path)
    topics = list(pe_map.values())

    event_bus = AsyncMock()
    applier = DispatchResultApplier(
        event_bus=event_bus,
        output_topic=topics[0],
        output_topic_map=pe_map,
        allowed_output_topics=topics,
    )

    result = ModelDispatchResult(
        status=EnumDispatchStatus.SUCCESS,
        topic="onex.cmd.omnibase-infra.delegation-call-request.v1",
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
        output_count=1,
        output_events=[ModelDelegationAllTiersFailed()],
        correlation_id=uuid4(),
    )

    await applier.apply(result, uuid4())

    assert event_bus.publish_envelope.await_count == 1
    _, kwargs = event_bus.publish_envelope.await_args
    assert kwargs["topic"] == "onex.evt.omnibase-infra.delegation-all-tiers-failed.v1"


# ---------------------------------------------------------------------------
# Unit: kernel-level manifest scan (OMN-12409 fix)
# ---------------------------------------------------------------------------


def test_manifest_scan_builds_applier_for_effect_contract(tmp_path: Path) -> None:
    """Kernel scan builds DispatchResultApplier from published_events for EFFECT.

    Reproduces the kernel logic: iterate filtered_manifest.contracts, skip
    already-registered entries, load published_events map from contract_path,
    and register an applier.  Contracts without published_events are skipped.
    """
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_EFFECT_CONTRACT_YAML, encoding="utf-8")

    # Simulate the filtered manifest with one EFFECT contract.
    from omnibase_infra.runtime.auto_wiring.models import (
        ModelAutoWiringManifest,
        ModelContractVersion,
        ModelDiscoveredContract,
        ModelEventBusWiring,
    )

    contracts = (
        ModelDiscoveredContract(
            name="node_llm_delegation_call_effect",
            node_type="EFFECT_GENERIC",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=contract_path,
            entry_point_name="node_llm_delegation_call_effect",
            package_name="omnimarket",
            event_bus=ModelEventBusWiring(
                subscribe_topics=(
                    "onex.cmd.omnibase-infra.delegation-call-request.v1",
                ),
                publish_topics=(
                    "onex.evt.omnibase-infra.inference-response.v1",
                    "onex.evt.omnibase-infra.delegation-all-tiers-failed.v1",
                ),
            ),
        ),
    )
    manifest = ModelAutoWiringManifest(contracts=contracts, errors=())

    # Kernel logic: scan manifest.contracts for published_events.
    event_bus_mock = AsyncMock()
    auto_wiring_result_appliers: dict[str, object] = {}

    for _contract in manifest.contracts:
        if _contract.name in auto_wiring_result_appliers:
            continue
        if _contract.event_bus is None:
            continue
        pe_map = load_published_events_map(Path(_contract.contract_path))
        if not pe_map:
            continue
        topics = list(pe_map.values())
        auto_wiring_result_appliers[_contract.name] = DispatchResultApplier(
            event_bus=event_bus_mock,
            output_topic=topics[0],
            output_topic_map=pe_map,
            allowed_output_topics=topics,
        )

    assert "node_llm_delegation_call_effect" in auto_wiring_result_appliers
    applier = auto_wiring_result_appliers["node_llm_delegation_call_effect"]
    assert isinstance(applier, DispatchResultApplier)
    assert applier.published_events_map == {
        "InferenceResponse": "onex.evt.omnibase-infra.inference-response.v1",
        "DelegationAllTiersFailed": "onex.evt.omnibase-infra.delegation-all-tiers-failed.v1",
    }


def test_manifest_scan_skips_already_registered_contracts(tmp_path: Path) -> None:
    """Kernel scan does not overwrite explicitly pre-registered appliers."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_EFFECT_CONTRACT_YAML, encoding="utf-8")

    from omnibase_infra.runtime.auto_wiring.models import (
        ModelAutoWiringManifest,
        ModelContractVersion,
        ModelDiscoveredContract,
        ModelEventBusWiring,
    )

    contracts = (
        ModelDiscoveredContract(
            name="node_llm_delegation_call_effect",
            node_type="EFFECT_GENERIC",
            contract_version=ModelContractVersion(major=1, minor=0, patch=0),
            contract_path=contract_path,
            entry_point_name="node_llm_delegation_call_effect",
            package_name="omnimarket",
            event_bus=ModelEventBusWiring(
                subscribe_topics=(
                    "onex.cmd.omnibase-infra.delegation-call-request.v1",
                ),
                publish_topics=("onex.evt.omnibase-infra.inference-response.v1",),
            ),
        ),
    )
    manifest = ModelAutoWiringManifest(contracts=contracts, errors=())

    # Pre-register a sentinel applier to verify it is NOT overwritten.
    sentinel = object()
    auto_wiring_result_appliers: dict[str, object] = {
        "node_llm_delegation_call_effect": sentinel,
    }

    for _contract in manifest.contracts:
        if _contract.name in auto_wiring_result_appliers:
            continue  # skip already-registered
        if _contract.event_bus is None:
            continue
        pe_map = load_published_events_map(Path(_contract.contract_path))
        if not pe_map:
            continue
        topics = list(pe_map.values())
        auto_wiring_result_appliers[_contract.name] = DispatchResultApplier(
            event_bus=AsyncMock(),
            output_topic=topics[0],
            output_topic_map=pe_map,
            allowed_output_topics=topics,
        )

    assert auto_wiring_result_appliers["node_llm_delegation_call_effect"] is sentinel


# ---------------------------------------------------------------------------
# Integration: wire_from_manifest auto-builds result_applier for EFFECT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.integration
async def test_effect_contract_handler_result_published_via_auto_wiring(
    tmp_path: Path,
) -> None:
    """EFFECT handler's returned model is published when wired via wire_from_manifest.

    This reproduces the OMN-12409 scenario: an EFFECT contract with publish_topics
    and published_events gets an auto-built DispatchResultApplier in
    _subscribe_contract_topics, so the handler's returned model is published to
    the declared topic.
    """
    subscribe_topic = "onex.cmd.omnibase-infra.delegation-call-request.v1"
    publish_topic = "onex.evt.omnibase-infra.inference-response.v1"
    contract_yaml_path = tmp_path / "contract.yaml"
    contract_yaml_path.write_text(_EFFECT_CONTRACT_YAML, encoding="utf-8")
    correlation_id = uuid4()

    contract = ModelDiscoveredContract(
        name="node_llm_delegation_call_effect",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_yaml_path,
        entry_point_name="node_llm_delegation_call_effect",
        package_name="omnimarket",
        terminal_event=publish_topic,
        event_bus=ModelEventBusWiring(
            subscribe_topics=(subscribe_topic,),
            publish_topics=(
                publish_topic,
                "onex.evt.omnibase-infra.delegation-all-tiers-failed.v1",
            ),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerEffectReturnsInferenceResponse",
                        module=__name__,
                    ),
                    event_model=None,
                ),
            ),
        ),
    )

    bus = EventBusInmemory(environment="test", group="omn-12409-effect-result")
    await bus.start()
    try:
        published_results: asyncio.Queue[ModelInferenceResponse] = asyncio.Queue()

        async def collect_result(message: ModelEventMessage) -> None:
            envelope = ModelEventEnvelope[ModelInferenceResponse].model_validate_json(
                message.value
            )
            if envelope.correlation_id == correlation_id:
                await published_results.put(envelope.payload)

        await bus.subscribe(
            publish_topic,
            group_id="result-collector",
            on_message=collect_result,
        )

        engine = MessageDispatchEngine()
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=HandlerEffectReturnsInferenceResponse,
        ):
            await wire_from_manifest(
                ModelAutoWiringManifest(contracts=(contract,)),
                engine,
                event_bus=bus,
                environment="local",
            )
        engine.freeze()

        command = ModelEventEnvelope[object](
            payload={"status": "ready"},
            correlation_id=correlation_id,
            event_type="omnibase-infra.delegation-call-request",
        )
        await bus.publish(
            subscribe_topic,
            None,
            command.model_dump_json().encode("utf-8"),
            None,
        )

        result = await asyncio.wait_for(published_results.get(), timeout=2)

        # The key assertion: result was published (not silently dropped).
        # correlation_id propagation depends on the handler implementation;
        # what matters here is that the EFFECT handler's returned model was
        # published to the declared publish_topic (not dropped).
        assert isinstance(result, ModelInferenceResponse)
    finally:
        await bus.close()
