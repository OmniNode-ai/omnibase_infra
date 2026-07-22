# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RoutingGate wiring proof for node_model_router_compute (OMN-7404, Task 8).

RED->GREEN: on the pre-wiring tree, ``HandlerScoreModels.__init__`` takes no
``gate`` parameter and `RoutingGate` is not imported by the handler module,
so every test here fails (``TypeError: unexpected keyword argument 'gate'``
or ``ImportError``). After wiring lands, all tests pass.

These tests assert the plan's Task 8 "Clarification from review" invariant:
the gate AUDITS, it never selects a model or changes the routing decision —
shadow-mode only. `gate=None` (the default; no classifier has ever been
trained) must reproduce prior behavior exactly, which the sibling
``test_score_models_defb_omn14825.py`` golden-equivalence corpus already
locks in byte-for-byte.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.learning.routing.gate import RoutingGate
from omnibase_infra.nodes.node_model_router_compute.handlers.handler_score_models import (
    HandlerScoreModels,
)
from omnibase_infra.nodes.node_model_router_compute.models.enum_task_type import (
    EnumTaskType,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_registry_entry import (
    ModelRegistryEntry,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_routing_constraints import (
    ModelRoutingConstraints,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_scoring_input import (
    ModelScoringInput,
)


def _make_input(task_description: str = "write a function") -> ModelScoringInput:
    registry = (
        ModelRegistryEntry(
            model_key="qwen3-coder-30b",
            provider="local",
            transport="http",
            base_url_env="LLM_CODER_URL",
            capabilities=("code_generation",),
            context_window=65536,
            tier="local",
        ),
    )
    return ModelScoringInput(
        correlation_id=uuid4(),
        task_type=EnumTaskType.CODE_GENERATION,
        task_description=task_description,
        constraints=ModelRoutingConstraints(),
        registry=registry,
    )


@pytest.mark.unit
class TestHandlerScoreModelsRoutingGateWiring:
    def test_default_gate_is_none_graceful_degradation(self) -> None:
        """No gate injected == today's production path; must construct fine."""
        handler = HandlerScoreModels()
        decision = handler.handle(_make_input())
        assert decision.success is True
        assert decision.selected_model_key == "qwen3-coder-30b"

    def test_gate_is_optional_constructor_arg(self) -> None:
        gate = RoutingGate(classifier=None)
        handler = HandlerScoreModels(gate=gate)
        decision = handler.handle(_make_input())
        assert decision.selected_model_key == "qwen3-coder-30b"

    def test_gate_audit_never_alters_selected_model(self) -> None:
        """Shadow-mode invariant: even a low-confidence audit changes nothing."""
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = 0.1  # classifier flags as low-confidence
        gate = RoutingGate(classifier=mock_clf, confidence_threshold=0.75)
        handler_with_gate = HandlerScoreModels(gate=gate)
        handler_without_gate = HandlerScoreModels()

        inp = _make_input()
        decision_with_gate = handler_with_gate.handle(inp)
        decision_without_gate = handler_without_gate.handle(inp)

        # Routing decision is byte-identical regardless of audit confidence.
        assert decision_with_gate.model_dump(
            mode="json"
        ) == decision_without_gate.model_dump(mode="json")
        mock_clf.predict_proba.assert_called_once()

    def test_gate_audit_logs_shadow_mode_result(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_proba.return_value = 0.42
        gate = RoutingGate(classifier=mock_clf, confidence_threshold=0.75)
        handler = HandlerScoreModels(gate=gate)

        with caplog.at_level(
            logging.INFO,
            logger="omnibase_infra.nodes.node_model_router_compute.handlers.handler_score_models",
        ):
            handler.handle(_make_input())

        assert any("routing_audit" in record.message for record in caplog.records)

    def test_gate_classifier_failure_does_not_break_routing(self) -> None:
        """A broken classifier must degrade to no-op, never crash the handler."""
        mock_clf = MagicMock()
        mock_clf.predict_proba.side_effect = RuntimeError("model corrupt")
        gate = RoutingGate(classifier=mock_clf, confidence_threshold=0.75)
        handler = HandlerScoreModels(gate=gate)

        decision = handler.handle(_make_input())

        assert decision.success is True
        assert decision.selected_model_key == "qwen3-coder-30b"
