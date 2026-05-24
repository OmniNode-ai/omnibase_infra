# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for Kafka contract default_handler materialization."""

from __future__ import annotations

from textwrap import dedent
from uuid import uuid4

import pytest

from omnibase_infra.runtime.kafka_contract_source import KafkaContractSource

pytestmark = pytest.mark.integration


def test_kafka_contract_source_materializes_default_handler_shorthand() -> None:
    """Kafka-discovered contracts preserve default_handler routing for wiring."""
    source = KafkaContractSource(environment="test", graceful_mode=False)
    node_name = f"test.default_handler.{uuid4().hex[:8]}"
    contract_yaml = dedent(
        """\
        handler_id: "effect.test.default_handler"
        name: "Default Handler Effect"
        contract_version:
          major: 1
          minor: 0
          patch: 0
        descriptor:
          node_archetype: "effect"
        input_model: "tests.fixtures.handler_proof_noop.ModelProofNoopRequest"
        output_model: "omnibase_core.models.dispatch.ModelHandlerOutput"
        metadata:
          handler_class: "tests.fixtures.handler_proof_noop.HandlerProofNoop"
        event_bus:
          subscribe_topics:
            - onex.cmd.test.default-handler.v1
          publish_topics:
            - onex.evt.test.default-handler-completed.v1
          consumer_purpose: consume
        handler_routing:
          default_handler: tests.fixtures.handler_proof_noop:HandlerProofNoop
        """
    )

    assert source.on_contract_registered(node_name, contract_yaml) is True
    descriptor = source.get_cached_descriptor(node_name)
    assert descriptor is not None

    contract = source._build_materialization_contract(
        node_name=node_name,
        descriptor=descriptor,
        environment="test",
    )

    assert contract.handler_routing is not None
    assert len(contract.handler_routing.handlers) == 1
    entry = contract.handler_routing.handlers[0]
    assert entry.handler.module == "tests.fixtures.handler_proof_noop"
    assert entry.handler.name == "HandlerProofNoop"
    assert contract.event_bus is not None
    assert contract.event_bus.subscribe_topics == ("onex.cmd.test.default-handler.v1",)
