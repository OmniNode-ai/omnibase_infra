# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression for OMN-15188: KafkaContractSource's dynamic-materialization
parser must preserve per-entry ``topic`` for topic_match contracts, the same
way ``discovery.py``'s static parser already does (OMN-13825/OMN-14580).

Found live 2026-07-26 while verifying the OMN-15147 worker leg: a freshly
built, Trivy-clean ``omninode-runtime``/``omninode-runtime-effects`` image
crash-looped on cold boot with::

    ONEX_CORE_064_DUPLICATE_REGISTRATION: Dispatcher with ID
    'dispatcher.auto.node_codegen_outcome_reducer.HandlerCodegenOutcomeReducer.
    reduce_codegen_outcome_3af675a4' is already registered.

``node_codegen_outcome_reducer`` (omnimarket) declares ``routing_strategy:
topic_match`` with 4 handler_routing entries, all routing the SAME
``operation: reduce_codegen_outcome`` to the SAME handler class
(``HandlerCodegenOutcomeReducer``), differing only by ``topic`` + a per-topic
``event_model`` -- the exact legitimate shape OMN-14580 fixed for
``node_swarm_subtask_state_reducer`` by folding ``topic`` into
``_derive_handler_entry_key``'s digest.

That fix only works if the parser that builds ``ModelHandlerRoutingEntry``
from parsed contract YAML actually threads ``topic`` through.
``omnibase_infra.runtime.auto_wiring.discovery._parse_handler_routing`` (the
static, entry_points-based discovery path) does this correctly -- confirmed
live by loading the real node_codegen_outcome_reducer contract.yaml and
running it through ``wire_from_manifest``: 4 distinct dispatcher IDs, no
crash. But ``KafkaContractSource._build_handler_routing`` (the dynamic,
Kafka-contract-registration materialization path used by
``materialize_cached_contract``, OMN-1654/OMN-11244/OMN-11247) built
``ModelHandlerRoutingEntry`` WITHOUT ``topic`` at all -- silently dropping it.
With no topic to fold, `_derive_handler_entry_key`` computes the SAME
operation-only digest for all 4 entries
(``sha1("reduce_codegen_outcome")[:8]`` == ``3af675a4``, matching the crash
log byte-for-byte), so the 2nd handler_routing entry collides with the 1st
at ``engine.register_dispatcher`` and raises
``ONEX_CORE_064_DUPLICATE_REGISTRATION``.

This drives the REAL, unmodified ``KafkaContractSource._build_handler_routing``
and ``materialize_cached_contract`` against the real 4-entry contract shape
mirrored from
``omnimarket/src/omnimarket/nodes/node_codegen_outcome_reducer/contract.yaml``
-- not a synthetic single-entry surrogate -- so a regression here reproduces
the identical crash the runtime hit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.models.handlers import ModelHandlerDescriptor
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.enums.enum_materialization_status import (
    EnumMaterializationStatus,
)
from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

pytestmark = pytest.mark.integration

_HANDLER_MODULE = (
    "omnimarket.nodes.node_codegen_outcome_reducer.handlers."
    "handler_codegen_outcome_reducer"
)
_HANDLER_NAME = "HandlerCodegenOutcomeReducer"
_OPERATION = "reduce_codegen_outcome"

# (topic, event_model name, event_model module) triples copied from the real
# node_codegen_outcome_reducer/contract.yaml handler_routing.handlers entries.
_ENTRIES = (
    (
        "onex.evt.omnimarket.codegen-llm-generated.v1",
        "ModelLlmGenerateResult",
        "omnimarket.codegen.models",
    ),
    (
        "onex.evt.omnimarket.generated-code-validation-completed.v1",
        "ModelGeneratedCodeValidation",
        "omnimarket.codegen.models",
    ),
    (
        "onex.evt.omnimarket.mypy-check-completed.v1",
        "ModelMypyCheckResult",
        "omnimarket.codegen.models",
    ),
    (
        "onex.evt.omnimarket.contract-serialize-completed.v1",
        "ModelContractDocument",
        "omnimarket.contract_assembly.models",
    ),
)


def _contract_config() -> dict:
    """Raw contract dict shaped like the real contract.yaml (parsed YAML)."""
    return {
        "name": "node_codegen_outcome_reducer",
        "node_type": "reducer",
        "handler": {
            "module": _HANDLER_MODULE,
            "class": _HANDLER_NAME,
        },
        "handler_routing": {
            "routing_strategy": "topic_match",
            "handlers": [
                {
                    "topic": topic,
                    "operation": _OPERATION,
                    "message_category": "event",
                    "event_model": {"name": model_name, "module": model_module},
                    "handler": {"name": _HANDLER_NAME, "module": _HANDLER_MODULE},
                }
                for topic, model_name, model_module in _ENTRIES
            ],
        },
        "event_bus": {
            "subscribe_topics": [topic for topic, _, _ in _ENTRIES],
            "publish_topics": [
                "onex.evt.omnimarket.codegen-validation-outcome.v1",
                "onex.evt.omnimarket.codegen-typecheck-outcome.v1",
                "onex.evt.omnimarket.codegen-serialize-outcome.v1",
            ],
        },
    }


def _make_descriptor() -> ModelHandlerDescriptor:
    return ModelHandlerDescriptor(
        handler_id="proto.node_codegen_outcome_reducer",
        name="node_codegen_outcome_reducer",
        version="1.0.0",
        handler_kind="reducer",
        input_model="omnibase_infra.models.TestInput",
        output_model="omnibase_infra.models.TestOutput",
        handler_class=f"{_HANDLER_MODULE}.{_HANDLER_NAME}",
        contract_path="kafka://dev/contracts/node_codegen_outcome_reducer",
        contract_config=_contract_config(),
    )


class HandlerCodegenOutcomeReducerProbe:
    """Zero-arg stand-in -- the real handler is also zero-arg constructed."""

    async def handle(self, envelope: object) -> None:
        return None


def test_build_handler_routing_preserves_topic_per_entry() -> None:
    """OMN-15188: KafkaContractSource._build_handler_routing must thread the
    contract-declared ``topic`` through for each handler_routing entry -- the
    same as discovery.py's _parse_handler_routing already does. Before the
    fix every entry.topic was None regardless of the YAML value."""
    handler_routing = KafkaContractSource._build_handler_routing(_contract_config())
    assert handler_routing is not None
    topics = [entry.topic for entry in handler_routing.handlers]
    assert topics == [topic for topic, _, _ in _ENTRIES], (
        "KafkaContractSource._build_handler_routing dropped per-entry topic "
        "(OMN-15188): all handler_routing entries must retain their "
        "contract-declared topic so _derive_handler_entry_key can fold it "
        "into the dispatcher-ID digest and avoid collisions."
    )


@pytest.mark.asyncio
async def test_materialize_cached_contract_wires_four_distinct_dispatchers() -> None:
    """OMN-15188: 4 topic_match entries, same handler + operation, materialized
    via the dynamic Kafka-contract-registration path must not collide on
    dispatcher registration (ONEX_CORE_064_DUPLICATE_REGISTRATION)."""
    source = KafkaContractSource(environment="local")
    descriptor = _make_descriptor()
    source._cache.add("node_codegen_outcome_reducer", descriptor)

    dispatch_engine = MessageDispatchEngine()
    event_bus = MagicMock(spec=ProtocolEventBusLike)
    event_bus.subscribe = AsyncMock(return_value=AsyncMock())

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerCodegenOutcomeReducerProbe,
    ):
        result = await source.materialize_cached_contract(
            node_name="node_codegen_outcome_reducer",
            dispatch_engine=dispatch_engine,
            event_bus=event_bus,
            environment="local",
        )

    assert result.status == EnumMaterializationStatus.MATERIALIZED, (
        f"expected MATERIALIZED, got {result.status} -- pre-fix this collided "
        "with ONEX_CORE_064_DUPLICATE_REGISTRATION on the 2nd handler_routing "
        "entry and was swallowed as REJECTED by materialize_cached_contract's "
        "broad except (the same class of defect that crash-looped "
        "omninode-runtime/-effects when it surfaces uncaught, per OMN-15188)."
    )
    assert len(result.registered_handlers) == 4
    assert len(set(result.registered_handlers)) == 4, (
        "all 4 topic_match entries must register distinct dispatcher IDs "
        "(OMN-15188: same handler + operation across 4 topics collided on "
        "one shared ID before the fix, because topic was dropped during "
        "dynamic-contract materialization)"
    )
