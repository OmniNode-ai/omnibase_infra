# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A lane-selection refusal prints its reason; a driver refusal prints none."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from omnibase_infra.cli.cli_ledger import ledger_group
from omnibase_infra.cli.delegate_lane import DelegateLaneSelectionError
from omnibase_infra.errors.error_infra import WorkLedgerQueryError
from omnibase_infra.models.model_work_ledger_read_refusal import (
    ModelWorkLedgerReadRefusal,
)

pytestmark = pytest.mark.unit


def test_lane_selection_refusal_prints_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refused(lane: str) -> object:
        raise DelegateLaneSelectionError(
            "the lane declaration x/config/lab/ci_bus_lanes.yaml does not exist "
            "(set ONEX_WORKSPACE_CONFIG_ROOT)"
        )

    monkeypatch.setattr(
        "omnibase_infra.cli.cli_ledger.resolve_work_ledger_query_transport",
        refused,
    )
    result = CliRunner().invoke(ledger_group, ["query"])
    assert result.exit_code == 1
    assert not result.stdout
    assert "refused (DelegateLaneSelectionError): the lane declaration" in result.stderr
    assert "ONEX_WORKSPACE_CONFIG_ROOT" in result.stderr


def test_driver_refusal_prints_no_message(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(lane: str) -> object:
        raise RuntimeError("postgresql://user:synthetic-secret@invalid/db")

    monkeypatch.setattr(
        "omnibase_infra.cli.cli_ledger.resolve_work_ledger_query_transport",
        refused,
    )
    result = CliRunner().invoke(ledger_group, ["query"])
    assert result.exit_code == 1
    assert not result.stdout
    assert "refused (RuntimeError)" in result.stderr
    assert "synthetic-secret" not in result.output
    line = next(
        line for line in result.stderr.splitlines() if "refused (RuntimeError)" in line
    )
    assert line.endswith("refused (RuntimeError)")


def test_model_of_lane_selection_keeps_message_and_other_types_do_not() -> None:
    lane_message = "the lane declaration x/config/lab/ci_bus_lanes.yaml does not exist"
    lane_refusal = ModelWorkLedgerReadRefusal.of(
        DelegateLaneSelectionError(lane_message)
    )
    assert str(lane_refusal) == f": {lane_message}"

    query_message = "the node's answer is an error"
    query_refusal = ModelWorkLedgerReadRefusal.of(WorkLedgerQueryError(query_message))
    assert str(query_refusal) == f": {query_message}"

    value_refusal = ModelWorkLedgerReadRefusal.of(ValueError("a credential"))
    assert str(value_refusal) == ""
