# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How the delegate CLI uses the execution budget the task-class contract declares."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.cli.cli_delegate import _terminal_wait_seconds, _write_payload
from tests.helpers.cli_registry_stand_in import StandInExecutionBudget

# The budget itself -- its declared values, the 240-second ceiling bound and
# the refusal of a class with no declared budget -- is the task-class
# contract's, and is tested by its owner (omnimarket
# tests/unit/inference/test_task_class_resolution_omn19407.py). The CLI reads
# it through the registry (OMN-19407); what is tested here is only what the
# CLI does with the budget it is handed.


@pytest.mark.unit
def test_payload_omits_unrequested_timeout_and_preserves_explicit_request(
    tmp_path: Path,
) -> None:
    common = {
        "prompt": "document the router",
        "task_type": "document",
        "source": "claude-code",
        "max_tokens": None,
        "state_root": tmp_path,
        "run_id": uuid4(),
        "correlation_id": uuid4(),
    }
    omitted = json.loads(
        _write_payload(**common, requested_timeout_seconds=None).read_text("utf-8")
    )
    explicit = json.loads(
        _write_payload(**common, requested_timeout_seconds=120).read_text("utf-8")
    )

    assert "requested_timeout_seconds" not in omitted
    assert explicit["requested_timeout_seconds"] == 120


@pytest.mark.unit
def test_terminal_wait_uses_effective_requested_execution_plus_margin() -> None:
    budget = StandInExecutionBudget(
        task_class_timeout_ceiling_seconds=240,
        terminal_delivery_margin_seconds=60,
    )

    assert (
        _terminal_wait_seconds(requested_timeout_seconds=None, execution_budget=budget)
        == 300
    )
    assert (
        _terminal_wait_seconds(requested_timeout_seconds=120, execution_budget=budget)
        == 180
    )
