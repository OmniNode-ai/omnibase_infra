# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-equivalence proof for the workspace-compute def-B flip (OMN-14633).

RSD mechanical-wave, execution wave 1 (parent OMN-14355; baseline source
OMN-14587; playbook proven by OMN-14589's invoke-effect canary).
``HandlerWorkspaceValidate.handle()`` moved from ``handle(envelope) ->
ModelHandlerOutput[ModelWorkspaceValidateResult]`` (def-A) to
``handle(command) -> ModelWorkspaceValidateResult`` (def-B); the runtime
adapter (``omnibase_infra.runtime.auto_wiring.handler_wiring
._normalize_handler_result``) already folds a bare ``BaseModel`` return into
``output_events`` for a non-envelope-accepting handler, so no adapter change
was needed for this node (unlike the REDUCER family blocked by OMN-14598).

The goldens under
``tests/fixtures/golden/node_coding_agent_workspace_compute/*.json`` were
recorded by replaying each scenario through the PRE-FLIP envelope-wrapping
``handle()`` body (script: one-off recorder, not committed) and are reproduced
here through the LIVE def-B ``handle()``. ``validate_workspace()`` itself
(the pure fold) is byte-for-byte unchanged by this flip — only the dispatch
boundary shape moved — so this is a durable regression: a future edit to this
handler cannot silently change behavior without failing here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from omnibase_infra.models.coding_agent.model_workspace_validate_command import (
    ModelWorkspaceValidateCommand,
)
from omnibase_infra.models.coding_agent.model_workspace_validate_result import (
    ModelWorkspaceValidateResult,
)
from omnibase_infra.nodes.node_coding_agent_workspace_compute.handlers.handler_workspace_validate import (
    HandlerWorkspaceValidate,
)
from scripts.ci.compute_golden import compare_output

pytestmark = pytest.mark.unit

_GOLDEN_DIR = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "golden"
    / "node_coding_agent_workspace_compute"
)


def _golden_files() -> list[Path]:
    files = sorted(_GOLDEN_DIR.glob("*.json"))
    assert files, f"no golden fixtures found under {_GOLDEN_DIR}"
    return files


@pytest.mark.parametrize("golden_path", _golden_files(), ids=lambda p: p.stem)
async def test_handle_reproduces_recorded_golden(golden_path: Path) -> None:
    """def-B ``handle(command)`` on the recorded input == the recorded output."""
    golden: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    command = ModelWorkspaceValidateCommand.model_validate(golden["input"])
    handler = HandlerWorkspaceValidate()
    fresh_output = await handler.handle(command)
    assert isinstance(fresh_output, ModelWorkspaceValidateResult)
    diffs = compare_output(golden, fresh_output)
    assert diffs == [], f"{golden_path.name}: handle() output diverged: {diffs}"


def test_golden_fixture_count_matches_expected_candidate_pool() -> None:
    """Regression guard: the recorded scenario pool has a known, reviewed size."""
    assert len(_golden_files()) == 6
