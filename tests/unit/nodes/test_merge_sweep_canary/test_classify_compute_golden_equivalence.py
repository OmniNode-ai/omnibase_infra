# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-equivalence proof for the classify-compute def-B flip (OMN-14824).

RSD canonical-rewrite lane (parent OMN-14355; burns down 1 of the entrypoint-less
/ non-canonical handler set under OMN-14510). ``HandlerClassifyPRs`` moved from a
multi-positional ``handle(prs, correlation_id, require_approval)`` (non-adaptable
def-A splat) to the canonical ``handle(request: ModelClassifyInput) ->
ModelClassifyResult`` (definition B) — the single typed-request entrypoint the
shared runtime adapter drives, no ``ModelEventEnvelope`` in the core. The
per-PR decision helper ``_classify_single`` is preserved byte-for-byte.

The goldens under ``tests/fixtures/golden/node_merge_sweep_classify_compute/*.json``
were recorded by replaying each scenario through the PRE-FLIP def-A handler at the
independent proof-author lane base_ref_exec_sha (recorder not committed) and are
reproduced here through the LIVE def-B ``handle()``. The three scenarios exercise
every branch of ``_classify_single`` (draft / auto-merge / track-A / CI-fail /
review-changes / review-none / conflicting) plus the ``not require_approval`` path
and the empty-batch loop — the same selected input set the committed adequacy +
equivalence receipts bind to. This is a durable regression: a future edit to the
handler cannot silently change behavior without failing here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_merge_sweep_classify_compute.handlers.handler_classify_prs import (
    HandlerClassifyPRs,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_input import (
    ModelClassifyInput,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_result import (
    ModelClassifyResult,
)
from scripts.ci.compute_golden import compare_output

pytestmark = pytest.mark.unit

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "golden"
    / "node_merge_sweep_classify_compute"
)


def _golden_files() -> list[Path]:
    files = sorted(_GOLDEN_DIR.glob("*.json"))
    assert files, f"no golden fixtures found under {_GOLDEN_DIR}"
    return files


@pytest.mark.asyncio
@pytest.mark.parametrize("golden_path", _golden_files(), ids=lambda p: p.stem)
async def test_handle_reproduces_recorded_golden(golden_path: Path) -> None:
    """def-B ``handle(request)`` on the recorded input == the def-A recorded output."""
    golden: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    request = ModelClassifyInput.model_validate(golden["input"])
    fresh_output = await HandlerClassifyPRs().handle(request)
    assert isinstance(fresh_output, ModelClassifyResult)
    diffs = compare_output(golden, fresh_output)
    assert diffs == [], f"{golden_path.name}: handle() output diverged: {diffs}"


def test_golden_fixture_count_matches_expected_candidate_pool() -> None:
    """Regression guard: the recorded scenario pool has a known, reviewed size."""
    assert len(_golden_files()) == 3


@pytest.mark.asyncio
async def test_equivalence_comparator_is_non_vacuous() -> None:
    """RED-vs-exists-but-wrong: a perturbed golden output is caught by the comparator."""
    golden_path = _GOLDEN_DIR / "mixed_require_approval.json"
    golden: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    perturbed = json.loads(json.dumps(golden))
    perturbed["output"]["total_classified"] = 999
    request = ModelClassifyInput.model_validate(golden["input"])
    fresh_output = await HandlerClassifyPRs().handle(request)
    diffs = compare_output(perturbed, fresh_output)
    assert any("total_classified" in d for d in diffs), diffs
