# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""def-B equivalence proof for node_chain_replay_compute (OMN-14815).

Goldens were recorded against the git-BASE def-A handler
(``handle(cached_chain, new_prompt_text, correlation_id, new_context)``) at
origin/dev, one per covered branch class. This test replays each golden's INPUT
through the LIVE canonical def-B handler (``handle(request)``) and asserts the
output is byte-equivalent — proving the signature flip preserved behavior. The
selected input set is bound to the adequacy receipt
(``scripts/ci/adequacy_receipts/omnibase_infra.nodes.node_chain_replay_compute.json``)
and the equivalence artifact (``.equivalence.json``).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_chain_orchestrator.models import ModelChainReplayInput
from omnibase_infra.nodes.node_chain_replay_compute.handlers.handler_chain_replay import (
    HandlerChainReplay,
)

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[3]
    / "fixtures"
    / "golden"
    / "node_chain_replay_compute"
)


def _goldens() -> list[dict[str, object]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(_GOLDEN_DIR.glob("*.json"))
    ]


def _golden_ids() -> list[str]:
    return [p.stem for p in sorted(_GOLDEN_DIR.glob("*.json"))]


@pytest.mark.unit
class TestChainReplayGoldenEquivalence:
    def test_goldens_present(self) -> None:
        goldens = _goldens()
        assert goldens, "no golden fixtures found — equivalence proof would be vacuous"

    def test_handle_is_definition_b(self) -> None:
        """The flip target: a single typed ``request`` parameter (def-B)."""
        params = list(inspect.signature(HandlerChainReplay.handle).parameters)
        assert params == ["self", "request"], (
            f"handle is not canonical def-B (params={params}); a revert to the "
            f"multi-positional def-A signature must fail this test"
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("golden", _goldens(), ids=_golden_ids())
    async def test_def_b_reproduces_def_a_golden(
        self, golden: dict[str, object]
    ) -> None:
        request = ModelChainReplayInput.model_validate(golden["input"])
        result = await HandlerChainReplay().handle(request)
        assert result.model_dump(mode="json") == golden["output"], (
            f"def-B output diverged from the recorded def-A golden "
            f"{golden.get('name')!r} — behavior NOT preserved"
        )

    @pytest.mark.asyncio
    async def test_equivalence_is_non_vacuous(self) -> None:
        """RED-vs-exists-but-wrong: a perturbed golden MUST be flagged."""
        golden = _goldens()[0]
        request = ModelChainReplayInput.model_validate(golden["input"])
        result = await HandlerChainReplay().handle(request)
        actual = result.model_dump(mode="json")
        perturbed = json.loads(json.dumps(golden["output"]))
        perturbed["confidence"] = perturbed["confidence"] + 0.01
        assert actual != perturbed, "comparator failed to detect a perturbed golden"
        assert actual == golden["output"]
