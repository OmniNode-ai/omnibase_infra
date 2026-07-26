# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical def-B flip proof for node_model_router_compute (OMN-14825).

RED->GREEN: on the pre-flip tree ``HandlerScoreModels`` exposed only the def-A
operation method ``score_candidates`` and NO ``handle`` entrypoint, so every
assertion here AttributeErrors / fails (RED). The canonical def-B flip renames
that method to ``handle(request) -> response`` (byte-identical body); these tests
then pass (GREEN).

``test_defb_handle_reproduces_legacy_goldens`` replays routing decisions recorded
against the git-BASE legacy handler through the live def-B ``handle`` and asserts
byte-equality (behaviour equivalence). ``test_golden_equivalence_is_non_vacuous``
proves the comparator would CATCH a divergence (RED-vs-exists-but-wrong).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_model_router_compute.handlers.handler_score_models import (
    HandlerScoreModels,
)
from omnibase_infra.nodes.node_model_router_compute.models.model_scoring_input import (
    ModelScoringInput,
)

_GOLDEN = (
    Path(__file__).resolve().parents[3]
    / "fixtures/golden/node_model_router_compute/legacy_equivalence_goldens.json"
)


def _load_cases() -> list[dict[str, object]]:
    data = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert isinstance(cases, list)
    return cases


@pytest.mark.unit
class TestModelRouterDefBFlip:
    """Proof that HandlerScoreModels is canonical definition B and behaviour-equivalent."""

    def test_handle_entrypoint_exists(self) -> None:
        handler = HandlerScoreModels()
        # Canonical def-B: the runtime adapts handle(request) -> response.
        assert hasattr(handler, "handle")
        assert callable(handler.handle)
        # The legacy def-A operation method must be gone (no retained shim).
        assert not hasattr(handler, "score_candidates")

    def test_defb_handle_reproduces_legacy_goldens(self) -> None:
        handler = HandlerScoreModels()
        cases = _load_cases()
        assert cases, "golden corpus must be non-empty"
        for case in cases:
            inp = ModelScoringInput.model_validate(case["input"])
            decision = handler.handle(inp)
            assert decision.model_dump(mode="json") == case["expected"], case[
                "scenario"
            ]

    def test_golden_equivalence_is_non_vacuous(self) -> None:
        handler = HandlerScoreModels()
        case = _load_cases()[0]
        inp = ModelScoringInput.model_validate(case["input"])
        decision = handler.handle(inp)
        perturbed = dict(case["expected"])
        perturbed["selected_model_key"] = (
            str(perturbed["selected_model_key"]) + "__WRONG"
        )
        assert decision.model_dump(mode="json") != perturbed
