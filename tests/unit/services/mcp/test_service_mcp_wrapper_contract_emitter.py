# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the generated-COMPUTE -> MCP wrapper-contract emitter (OMN-12841).

These tests exercise the generation-layer of Option B (contract-native): the
node-generation pipeline emits, alongside the generated COMPUTE node, a thin
declarative ORCHESTRATOR wrapper contract that is the MCP-exposed entity. The
orchestrator-only MCP gate is NOT relaxed (Option A rejected); the wrapper is
an ORCHESTRATOR that satisfies the existing gate.

Source of truth for these tests (the wrapper contract MUST be):

- ``node_type: ORCHESTRATOR_GENERIC`` (so the discovery/sync gates accept it).
- ``mcp.expose: true`` with the requested ``tool_name`` set.
- declarative ``handler_routing`` that routes the inbound invocation envelope to
  the generated COMPUTE handler (no custom node logic).
- emitted ALONGSIDE the generated COMPUTE contract (both artifacts present).
"""

from __future__ import annotations

import pytest
import yaml

from omnibase_infra.models.mcp.model_generated_compute_node_spec import (
    ModelGeneratedComputeNodeSpec,
)
from omnibase_infra.services.mcp.service_mcp_wrapper_contract_emitter import (
    ServiceMCPWrapperContractEmitter,
)

_COMPUTE_CONTRACT_YAML = """\
name: node_sentiment_classifier
node_type: COMPUTE_GENERIC
contract_version:
  major: 1
  minor: 0
  patch: 0
node_version: "1.0.0"
description: Classify customer review sentiment.
input_model:
  name: ModelSentimentInput
  module: generated.models
  description: Review text input.
output_model:
  name: ModelSentimentOutput
  module: generated.models
  description: Sentiment + confidence.
"""


@pytest.fixture
def spec() -> ModelGeneratedComputeNodeSpec:
    return ModelGeneratedComputeNodeSpec(
        node_name="node_sentiment_classifier",
        description="Classify customer review sentiment.",
        tool_name="node_sentiment_classifier",
        compute_contract_yaml=_COMPUTE_CONTRACT_YAML,
        compute_handler_module=(
            "generated.node_sentiment_classifier.handlers.handler_sentiment_classifier"
        ),
        compute_handler_class="HandlerSentimentClassifier",
        invocation_input_model_module="generated.models",
        invocation_input_model_name="ModelSentimentInput",
        invocation_output_model_module="generated.models",
        invocation_output_model_name="ModelSentimentOutput",
    )


class TestWrapperContractEmitter:
    """Generation layer: emit BOTH compute + orchestrator wrapper contracts."""

    def test_emits_both_compute_and_orchestrator_contracts(
        self, spec: ModelGeneratedComputeNodeSpec
    ) -> None:
        emission = ServiceMCPWrapperContractEmitter().emit(spec)

        # Both artifacts are present.
        assert emission.compute_contract_yaml == _COMPUTE_CONTRACT_YAML
        assert emission.wrapper_contract_yaml

        compute = yaml.safe_load(emission.compute_contract_yaml)
        wrapper = yaml.safe_load(emission.wrapper_contract_yaml)

        assert compute["node_type"] == "COMPUTE_GENERIC"
        assert wrapper["node_type"] == "ORCHESTRATOR_GENERIC"

    def test_wrapper_name_is_orchestrator_suffixed(
        self, spec: ModelGeneratedComputeNodeSpec
    ) -> None:
        emission = ServiceMCPWrapperContractEmitter().emit(spec)
        wrapper = yaml.safe_load(emission.wrapper_contract_yaml)

        assert wrapper["name"] == "node_sentiment_classifier_orchestrator"

    def test_wrapper_declares_mcp_expose_and_tool_name(
        self, spec: ModelGeneratedComputeNodeSpec
    ) -> None:
        emission = ServiceMCPWrapperContractEmitter().emit(spec)
        wrapper = yaml.safe_load(emission.wrapper_contract_yaml)

        assert wrapper["mcp"]["expose"] is True
        assert wrapper["mcp"]["tool_name"] == "node_sentiment_classifier"
        assert wrapper["mcp"]["description"] == "Classify customer review sentiment."

    def test_wrapper_routes_to_compute_handler(
        self, spec: ModelGeneratedComputeNodeSpec
    ) -> None:
        emission = ServiceMCPWrapperContractEmitter().emit(spec)
        wrapper = yaml.safe_load(emission.wrapper_contract_yaml)

        routing = wrapper["handler_routing"]
        assert routing["routing_strategy"] == "payload_type_match"
        handlers = routing["handlers"]
        assert len(handlers) == 1
        handler = handlers[0]["handler"]
        assert handler["module"] == (
            "generated.node_sentiment_classifier.handlers.handler_sentiment_classifier"
        )
        assert handler["name"] == "HandlerSentimentClassifier"

    def test_wrapper_input_model_matches_compute_invocation_input(
        self, spec: ModelGeneratedComputeNodeSpec
    ) -> None:
        emission = ServiceMCPWrapperContractEmitter().emit(spec)
        wrapper = yaml.safe_load(emission.wrapper_contract_yaml)

        assert wrapper["input_model"]["name"] == "ModelSentimentInput"
        assert wrapper["input_model"]["module"] == "generated.models"
        assert wrapper["output_model"]["name"] == "ModelSentimentOutput"

    def test_emit_is_deterministic(self, spec: ModelGeneratedComputeNodeSpec) -> None:
        first = ServiceMCPWrapperContractEmitter().emit(spec)
        second = ServiceMCPWrapperContractEmitter().emit(spec)

        assert first.wrapper_contract_yaml == second.wrapper_contract_yaml

    def test_empty_node_name_fails_fast(self) -> None:
        with pytest.raises(ValueError):
            ModelGeneratedComputeNodeSpec(
                node_name="",
                description="x",
                tool_name="x",
                compute_contract_yaml=_COMPUTE_CONTRACT_YAML,
                compute_handler_module="generated.handlers.h",
                compute_handler_class="H",
                invocation_input_model_module="generated.models",
                invocation_input_model_name="ModelIn",
                invocation_output_model_module="generated.models",
                invocation_output_model_name="ModelOut",
            )
