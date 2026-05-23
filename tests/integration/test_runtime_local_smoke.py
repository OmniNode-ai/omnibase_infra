# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration smoke coverage for the infra-owned RuntimeLocal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_core.enums.enum_workflow_result import EnumWorkflowResult
from omnibase_infra.runtime.runtime_local import RuntimeLocal

pytestmark = pytest.mark.integration


def test_runtime_local_runs_contract_with_file_input(tmp_path: Path) -> None:
    """RuntimeLocal loads a contract, validates file input, and invokes a handler."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "---\n"
        "name: proof_noop\n"
        "node_type: compute\n"
        "terminal_event: onex.evt.proof.noop-completed.v1\n"
        "handler:\n"
        "  module: tests.fixtures.handler_proof_noop\n"
        "  class: HandlerProofNoop\n"
        "  input_model: tests.fixtures.handler_proof_noop.ModelProofNoopRequest\n"
        "handler_routing:\n"
        "  default_handler: tests.fixtures.handler_proof_noop:HandlerProofNoop\n",
        encoding="utf-8",
    )
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"name": "integration-smoke", "count": 7}), encoding="utf-8"
    )

    result = RuntimeLocal(
        workflow_path=contract_path,
        state_root=tmp_path / "state",
        input_path=input_path,
        timeout=10,
    ).run()

    assert result == EnumWorkflowResult.COMPLETED
    echo = json.loads((tmp_path / "state" / "echo.json").read_text(encoding="utf-8"))
    assert echo == {"name": "integration-smoke", "count": 7}
