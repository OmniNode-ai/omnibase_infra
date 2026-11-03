# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A boundary failure terminal is a delegation terminal the CLI reports (OMN-22216).

When the deployed lane's consume boundary fails a delegate request for good it
publishes a ``ModelBoundaryFailureTerminal`` (``status="failed"``,
``failure_class``, ``failure_code``, ``failure_reason``) inside an event
envelope. That payload has no ``attempts``, so before OMN-22216 neither carrier
shape validated and ``onex delegate`` printed "delegate receipt carries no
resolvable delegation terminal", dropping the ``DatabaseError`` text the
terminal carried.

* AC1 -> ``test_failure_terminal_resolved_*``
* AC2 -> ``test_no_terminal_sentence_kept_*``

Every receipt here is an in-memory dict shaped as ``run_receipt_mode`` builds
one; no bus, broker or database is touched.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import _delegation_result
from omnibase_infra.cli.delegate_human_output import render_delegate_outcome
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
    resolve_delegate_terminal,
)

pytestmark = pytest.mark.unit

_NO_TERMINAL_SENTENCE = "no resolvable delegation terminal"
_FAILURE_CLASS = "DatabaseError"
_FAILURE_CODE = "ONEX_CORE_DATABASE_ERROR"
_FAILURE_REASON = "file is not a database"
_CONTRACT = (
    "/site-packages/omnimarket/nodes/node_delegate_skill_orchestrator/contract.yaml"
)


def _boundary_failure_payload(*, failure_code: str | None) -> dict[str, object]:
    """The boundary's terminal payload as ``model_dump(mode="json")`` writes it."""
    return {
        "correlation_id": str(uuid.uuid4()),
        "status": "failed",
        "failure_class": _FAILURE_CLASS,
        "failure_code": failure_code,
        "retryable": False,
        "failure_reason": _FAILURE_REASON,
        "origin_topic": "onex.cmd.omnimarket.delegate-skill.v1",
        "terminalized_at": "2026-10-23T06:55:37Z",
    }


def _envelope_carrier(*, failure_code: str | None) -> dict[str, object]:
    return {
        "envelope_id": str(uuid.uuid4()),
        "correlation_id": str(uuid.uuid4()),
        "payload": _boundary_failure_payload(failure_code=failure_code),
    }


def _receipt(terminal_payload: object) -> dict[str, object]:
    """A dispatched-run receipt: the summary shape, the terminal in a carrier."""
    return {
        "run_id": str(uuid.uuid4()),
        "correlation_id": str(uuid.uuid4()),
        "status": "error",
        "exit_code": 1,
        "result_model": (
            "omnibase_infra.cli.model_receipt_runtime_summary."
            "ModelReceiptRuntimeSummary"
        ),
        "result": {
            "workflow_result": "failed",
            "exit_code": 1,
            "workflow": _CONTRACT,
            "terminal_payload": terminal_payload,
            "handler_result": None,
            "wire_correlation_id": str(uuid.uuid4()),
        },
    }


@pytest.mark.parametrize("failure_code", [_FAILURE_CODE, None])
def test_failure_terminal_resolved_by_the_resolver(failure_code: str | None) -> None:
    terminal = resolve_delegate_terminal(_envelope_carrier(failure_code=failure_code))

    assert terminal.status == "failed"
    assert terminal.attempts == ()
    assert terminal.accepted_attempt is None
    assert _FAILURE_CLASS in (terminal.terminal_failure_cause or "")
    if failure_code is not None:
        assert failure_code in (terminal.terminal_failure_cause or "")
    assert terminal.terminal_failure_reason == _FAILURE_REASON


def test_failure_terminal_resolved_bare() -> None:
    terminal = resolve_delegate_terminal(
        _boundary_failure_payload(failure_code=_FAILURE_CODE)
    )

    assert terminal.status == "failed"
    assert terminal.terminal_failure_reason == _FAILURE_REASON


def test_failure_terminal_resolved_from_the_receipt() -> None:
    terminal = _delegation_result(
        _receipt(_envelope_carrier(failure_code=_FAILURE_CODE))
    )

    assert terminal is not None
    assert terminal.terminal_failure_reason == _FAILURE_REASON


def test_failure_terminal_resolved_cli_prints_code_and_text(tmp_path: Path) -> None:
    outcome = render_delegate_outcome(
        _receipt(_envelope_carrier(failure_code=_FAILURE_CODE)), state_root=tmp_path
    )

    assert outcome is not None
    assert not outcome.succeeded
    line = " ".join(outcome.stderr)
    assert _FAILURE_CLASS in line
    assert _FAILURE_CODE in line
    assert _FAILURE_REASON in line
    assert _NO_TERMINAL_SENTENCE not in line


def test_failure_terminal_resolved_validator_reports_no_sentence() -> None:
    from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
    from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
    from omnibase_infra.cli.model_receipt_runtime_summary import (
        ModelReceiptRuntimeSummary,
    )
    from omnibase_infra.runtime_identity import collect_runtime_identity

    receipt = ModelSkillResult[ModelReceiptRuntimeSummary](
        skill_name=cli_delegate.DELEGATE_NODE_NAME,
        node_name=cli_delegate.DELEGATE_NODE_NAME,
        status=EnumSkillResultStatus.ERROR,
        correlation_id=uuid.uuid4(),
        run_id=uuid.uuid4(),
        exit_code=1,
        duration_ms=10,
        result=ModelReceiptRuntimeSummary(
            workflow_result="failed",
            exit_code=1,
            workflow=_CONTRACT,
            terminal_payload=_envelope_carrier(failure_code=_FAILURE_CODE),
            handler_result=None,
        ),
        result_model=(
            "omnibase_infra.cli.model_receipt_runtime_summary."
            "ModelReceiptRuntimeSummary"
        ),
        runtime_identity=collect_runtime_identity(config_source="test"),
    )

    defect = cli_delegate._delegate_receipt_evidence_error(receipt)

    assert defect is None or _NO_TERMINAL_SENTENCE not in defect


def test_no_terminal_sentence_kept_when_no_carrier_holds_one() -> None:
    with pytest.raises(DelegateTerminalUnresolvedError, match=_NO_TERMINAL_SENTENCE):
        _delegation_result(_receipt(None))


def test_no_terminal_sentence_kept_for_an_unrelated_payload() -> None:
    with pytest.raises(DelegateTerminalUnresolvedError, match=_NO_TERMINAL_SENTENCE):
        _delegation_result(_receipt({"envelope_id": str(uuid.uuid4()), "payload": {}}))


def test_no_terminal_sentence_kept_for_a_success_status_payload() -> None:
    """Only a ``failed`` boundary terminal resolves; a look-alike does not."""
    payload = _boundary_failure_payload(failure_code=_FAILURE_CODE)
    payload["status"] = "completed"
    with pytest.raises(DelegateTerminalUnresolvedError, match=_NO_TERMINAL_SENTENCE):
        _delegation_result(_receipt(payload))
