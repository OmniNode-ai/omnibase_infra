# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` reports a boundary failure terminal's own code and text (OMN-22216).

Drives the receipt renderer the command ends in with a dispatched-run receipt
whose only terminal is the consume boundary's failure terminal, and asserts the
human failure line carries the failure class, ONEX code and failure text instead
of the no-resolvable-terminal sentence. No bus, broker or database is touched.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from omnibase_infra.cli.cli_delegate import _delegation_result
from omnibase_infra.cli.delegate_human_output import render_delegate_outcome

_FAILURE_CLASS = "DatabaseError"
_FAILURE_CODE = "ONEX_CORE_DATABASE_ERROR"
_FAILURE_REASON = "file is not a database"
_NO_TERMINAL_SENTENCE = "no resolvable delegation terminal"


def _receipt(terminal_payload: object) -> dict[str, object]:
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
            "workflow": (
                "/site-packages/omnimarket/nodes/"
                "node_delegate_skill_orchestrator/contract.yaml"
            ),
            "terminal_payload": terminal_payload,
            "handler_result": None,
            "wire_correlation_id": str(uuid.uuid4()),
        },
    }


def test_boundary_failure_terminal_reaches_the_failure_line(tmp_path: Path) -> None:
    carrier = {
        "envelope_id": str(uuid.uuid4()),
        "correlation_id": str(uuid.uuid4()),
        "payload": {
            "correlation_id": str(uuid.uuid4()),
            "status": "failed",
            "failure_class": _FAILURE_CLASS,
            "failure_code": _FAILURE_CODE,
            "retryable": False,
            "failure_reason": _FAILURE_REASON,
            "origin_topic": "onex.cmd.omnimarket.delegate-skill.v1",
            "terminalized_at": "2026-10-23T06:55:37Z",
        },
    }

    outcome = render_delegate_outcome(_receipt(carrier), state_root=tmp_path)

    assert outcome is not None
    assert not outcome.succeeded
    line = " ".join(outcome.stderr)
    assert _FAILURE_CLASS in line
    assert _FAILURE_CODE in line
    assert _FAILURE_REASON in line
    assert _NO_TERMINAL_SENTENCE not in line


@pytest.mark.parametrize("payload", [None, {"payload": {}}])
def test_no_terminal_receipt_keeps_the_sentence(payload: object) -> None:
    from omnibase_infra.cli.delegate_terminal_resolver import (
        DelegateTerminalUnresolvedError,
    )

    with pytest.raises(DelegateTerminalUnresolvedError, match=_NO_TERMINAL_SENTENCE):
        _delegation_result(_receipt(payload))
