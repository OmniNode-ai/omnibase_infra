# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-equivalence proof for the contract-validate def-B flip (OMN-14810).

RSD canonical-rewrite lane (parent OMN-14355; burns down 1 of 44 under OMN-14510).
``HandlerContractValidate`` moved from a module-level ``handle_contract_validate(
input_data) -> ModelContractValidationResult`` (def-A) to
``HandlerContractValidate.handle(request) -> ModelContractValidationResult``
(def-B) — the validation body moved out of the free function and onto the handler
method, which now OWNS the behavior (no retained op-method shim, no
``ModelEventEnvelope``).

The goldens under
``tests/fixtures/golden/node_contract_validate_compute/*.json`` were recorded by
replaying each scenario through the PRE-FLIP def-A ``handle_contract_validate``
body at base_ref ``71ab1bf6`` (independent proof-author lane; recorder not
committed) and are reproduced here through the LIVE def-B ``handle()``. The three
scenarios exercise every branch of ``handle()`` (contract-YAML, model-compliance,
and file-path) — the same selected input set the committed adequacy +
equivalence receipts bind to. This is a durable regression: a future edit to the
handler cannot silently change behavior without failing here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.models.validation.model_contract_validation_result import (
    ModelContractValidationResult,
)
from omnibase_infra.nodes.node_contract_validate_compute.handlers import (
    HandlerContractValidate,
)
from omnibase_infra.nodes.node_contract_validate_compute.models import (
    ModelContractValidateInput,
)
from scripts.ci.compute_golden import compare_output

pytestmark = pytest.mark.unit

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "golden"
    / "node_contract_validate_compute"
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
    request = ModelContractValidateInput.model_validate(golden["input"])
    fresh_output = await HandlerContractValidate().handle(request)
    assert isinstance(fresh_output, ModelContractValidationResult)
    diffs = compare_output(golden, fresh_output)
    assert diffs == [], f"{golden_path.name}: handle() output diverged: {diffs}"


def test_golden_fixture_count_matches_expected_candidate_pool() -> None:
    """Regression guard: the recorded scenario pool has a known, reviewed size."""
    assert len(_golden_files()) == 3
