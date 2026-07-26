# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-equivalence proof for the scope-extract def-B flip (OMN-14832).

RSD canonical-rewrite lane (parent OMN-14355; burns down 1 of the Class-B Tier-1
fan-out under OMN-14510). ``HandlerScopeExtract`` moved from a multi-positional
``handle(self, content, plan_file_path, correlation_id, output_path) ->
ModelScopeExtracted`` (non-canonical: the runtime adapter's
``_resolve_def_b_input_model_type`` returns ``None`` for it) to the canonical
definition-B ``handle(self, request: ModelScopeExtractInput) -> ModelScopeExtracted``
-- the entry signature is unpacked from the contract ``input_model``; the regex
extraction body is behaviorally identical.

The goldens under ``tests/fixtures/golden/node_scope_extract_compute/*.json`` were
recorded by replaying each scenario through the PRE-FLIP def-A ``handle`` body at
the git base (independent proof-author lane; recorder not committed) and are
reproduced here through the LIVE def-B ``handle()``. The selected scenarios are
the exact coverage-guided input set the committed adequacy + equivalence receipts
bind to (``scripts/ci/adequacy_receipts/omnibase_infra.nodes.node_scope_extract_compute*``).
This is a durable regression: a future edit to the handler cannot silently change
behavior without failing here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.nodes.node_scope_extract_compute.handlers import (
    HandlerScopeExtract,
)
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extract_input import (
    ModelScopeExtractInput,
)
from omnibase_infra.nodes.node_scope_extract_compute.models.model_scope_extracted import (
    ModelScopeExtracted,
)
from scripts.ci.compute_golden import compare_output

pytestmark = pytest.mark.unit

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "golden"
    / "node_scope_extract_compute"
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
    request = ModelScopeExtractInput.model_validate(golden["input"])
    fresh_output = await HandlerScopeExtract().handle(request)
    assert isinstance(fresh_output, ModelScopeExtracted)
    diffs = compare_output(golden, fresh_output)
    assert diffs == [], f"{golden_path.name}: handle() output diverged: {diffs}"


def test_golden_fixture_count_matches_expected_candidate_pool() -> None:
    """Regression guard: the recorded scenario pool has a known, reviewed size."""
    assert len(_golden_files()) == 3
