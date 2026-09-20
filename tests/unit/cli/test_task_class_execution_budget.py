# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Task-class execution budget resolution for the delegate CLI."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.cli.cli_delegate import _terminal_wait_seconds, _write_payload
from omnibase_infra.cli.model_task_class_execution_budget import (
    ModelTaskClassExecutionBudget,
)
from omnibase_infra.cli.task_class_selection import (
    TaskClassContractError,
    resolve_task_class_execution_budget,
)


def _contract(path: Path, *, budget: str = "") -> Path:
    declared = budget or (
        "    task_class_timeout_ceiling_seconds: 240\n"
        "    terminal_delivery_margin_seconds: 60"
    )
    path.write_text(
        f"execution_budgets:\n  document:\n{declared}\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_resolves_declared_execution_and_delivery_windows(tmp_path: Path) -> None:
    budget = resolve_task_class_execution_budget(
        _contract(tmp_path / "task_classes.yaml"), task_type="document"
    )

    assert budget.task_class_timeout_ceiling_seconds == 240
    assert budget.terminal_delivery_margin_seconds == 60


@pytest.mark.unit
def test_refuses_an_execution_ceiling_that_reaches_port_wait(tmp_path: Path) -> None:
    with pytest.raises(TaskClassContractError, match="less_than_equal"):
        resolve_task_class_execution_budget(
            _contract(
                tmp_path / "task_classes.yaml",
                budget=(
                    "    task_class_timeout_ceiling_seconds: 300\n"
                    "    terminal_delivery_margin_seconds: 60"
                ),
            ),
            task_type="document",
        )


@pytest.mark.unit
def test_refuses_a_selected_class_without_declared_budget(tmp_path: Path) -> None:
    with pytest.raises(
        TaskClassContractError, match="'research' declares no execution budget"
    ):
        resolve_task_class_execution_budget(
            _contract(tmp_path / "task_classes.yaml"), task_type="research"
        )


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
    budget = ModelTaskClassExecutionBudget(
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
