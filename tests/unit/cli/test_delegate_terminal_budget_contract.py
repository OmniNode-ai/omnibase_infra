# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The CLI receipt preserves budget outcomes without fabricating execution."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.cli.cli_delegate import (
    _budget_outcome_receipt_block,
    _require_completed_terminal_evidence,
    _response_contract_receipt_block,
)
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal


def _terminal(**budget_outcome: object) -> ModelDelegateTerminal:
    return ModelDelegateTerminal.model_validate(
        {
            "attempts": [],
            "status": "failed",
            **budget_outcome,
        }
    )


@pytest.mark.unit
def test_terminal_preserves_omitted_requested_timeout_as_null() -> None:
    terminal = _terminal(
        budget_evidence={
            "requested_timeout_seconds": None,
            "task_class_timeout_ceiling_seconds": 240,
            "execution_timeout_seconds": 240,
            "terminal_delivery_margin_seconds": 60,
        }
    )

    assert terminal.budget_evidence is not None
    assert terminal.budget_evidence.requested_timeout_seconds is None
    assert terminal.budget_refusal is None


@pytest.mark.unit
def test_terminal_preserves_typed_predispatch_refusal() -> None:
    terminal = _terminal(
        budget_refusal={
            "reason": "timeout_exceeds_task_class_ceiling",
            "task_type": "document",
            "requested_timeout_seconds": 241,
            "task_class_timeout_ceiling_seconds": 240,
        }
    )

    assert terminal.budget_refusal is not None
    assert terminal.budget_refusal.reason == "timeout_exceeds_task_class_ceiling"
    assert terminal.budget_evidence is None


@pytest.mark.unit
def test_terminal_refuses_mixed_execution_and_predispatch_evidence() -> None:
    with pytest.raises(ValidationError, match="cannot carry both"):
        _terminal(
            budget_evidence={
                "requested_timeout_seconds": 120,
                "task_class_timeout_ceiling_seconds": 240,
                "execution_timeout_seconds": 120,
                "terminal_delivery_margin_seconds": 60,
            },
            budget_refusal={
                "reason": "timeout_exceeds_task_class_ceiling",
                "task_type": "document",
                "requested_timeout_seconds": 241,
                "task_class_timeout_ceiling_seconds": 240,
            },
        )


@pytest.mark.unit
def test_receipt_block_does_not_invent_budget_evidence_for_legacy_terminal() -> None:
    assert _budget_outcome_receipt_block(_terminal()) == {}


@pytest.mark.unit
def test_receipt_block_carries_only_the_terminal_declared_refusal() -> None:
    block = _budget_outcome_receipt_block(
        _terminal(
            budget_refusal={
                "reason": "timeout_exceeds_task_class_ceiling",
                "task_type": "document",
                "requested_timeout_seconds": 241,
                "task_class_timeout_ceiling_seconds": 240,
            }
        )
    )

    assert block == {
        "budget_refusal": {
            "reason": "timeout_exceeds_task_class_ceiling",
            "task_type": "document",
            "requested_timeout_seconds": 241,
            "task_class_timeout_ceiling_seconds": 240,
        }
    }


@pytest.mark.unit
def test_receipt_block_carries_declared_contract_evidence_and_preamble() -> None:
    block = _response_contract_receipt_block(
        _terminal(
            response_contract_evidence={
                "conveyed": True,
                "validated": True,
                "output_shape": "json",
                "contract_sha256": "0" * 64,
                "channel": "system",
            },
            preamble_chars=12,
        )
    )

    assert block["preamble_chars"] == 12
    assert block["response_contract_evidence"] == {
        "conveyed": True,
        "validated": True,
        "output_shape": "json",
        "contract_sha256": "0" * 64,
        "channel": "system",
    }


@pytest.mark.unit
def test_receipt_block_preserves_typed_output_refusal() -> None:
    block = _response_contract_receipt_block(
        _terminal(
            output_refusal={
                "reason": "no_schema_conforming_json",
                "output_shape": "json",
                "contract_failure_reasons": ("required property 'answer' is missing",),
            }
        )
    )

    assert block["output_refusal"] == {
        "reason": "no_schema_conforming_json",
        "output_shape": "json",
        "contract_failure_reasons": ["required property 'answer' is missing"],
    }


@pytest.mark.unit
def test_completed_terminal_without_budget_evidence_is_refused() -> None:
    completed = _terminal().model_copy(update={"status": "completed"})

    with pytest.raises(DelegateTerminalUnresolvedError, match="budget_evidence"):
        _require_completed_terminal_evidence(
            completed,
            require_budget_evidence=True,
            require_contract_evidence=False,
        )


@pytest.mark.unit
def test_completed_declared_contract_without_contract_evidence_is_refused() -> None:
    completed = _terminal(
        budget_evidence={
            "requested_timeout_seconds": None,
            "task_class_timeout_ceiling_seconds": 240,
            "execution_timeout_seconds": 240,
            "terminal_delivery_margin_seconds": 60,
        }
    ).model_copy(update={"status": "completed"})

    with pytest.raises(
        DelegateTerminalUnresolvedError, match="response_contract_evidence"
    ):
        _require_completed_terminal_evidence(
            completed,
            require_budget_evidence=True,
            require_contract_evidence=True,
        )
