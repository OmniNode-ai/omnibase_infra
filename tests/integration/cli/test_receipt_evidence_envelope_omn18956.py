# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The whole receipt-validator chain accepts the terminal the lane produced (OMN-18956).

WHY THIS IS SEPARATE FROM THE UNIT MODULE. The unit module next door calls
`_require_completed_terminal_evidence` directly, so it proves the predicate
and nothing about how a real receipt reaches it. The failure being fixed
happened one layer out: `_delegate_receipt_evidence_error` takes a RECEIPT,
walks it to the terminal through `_delegation_result` -- which has two receipt
shapes and, inside one of them, two carrier shapes -- and only then asks the
question. A fix proven at the leaf can still be wrong at the entry the CLI
actually wires as `receipt_validator`.

So this drives the entry point over a FROZEN CAPTURE of the real receipt:
`tests/fixtures/delegation/omn18956/completed_terminal_no_evidence.json`, the
verbatim envelope from correlation `0b21b5f0-fb56-4ab6-9219-24bbfd841f35` on
the dev lane, which completed, passed its quality gate at 1.0, wrote its
projection row, and was still reported as a failure.

The capture is deliberate rather than convenient. The first draft of this
module hand-built what I believed the envelope looked like and the validator
answered "delegate receipt carries no delegation terminal" -- my shape was
wrong. A test written against a believed shape proves nothing about the
artifact, which is the same class of error as the outage this ticket sits
under.

The three cases are the whole contract in one place: the lane's own shape is
accepted, the same shape is still refused when a contract was requested and
its evidence is absent, and a request that asked and was answered passes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.cli.cli_delegate import (
    _delegate_receipt_evidence_error,
    _receipt_evidence_requirements,
    _write_local_run_files,
)
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.integration


class _Receipt:
    """Stands in for the typed receipt only in being serializable.

    The validator's first act is `getattr(receipt, "model_dump", None)`, so
    what matters is that it dumps to the envelope below. Using the real
    receipt class would drag in a runtime the assertion does not need and
    would test `run_receipt_mode`, not this.
    """

    def __init__(self, envelope: dict[str, object]) -> None:
        self._envelope = envelope

    def model_dump(self, mode: str = "python") -> dict[str, object]:
        return self._envelope


#: A valid contract-evidence object, every field the model declares required.
#: Written out rather than minimised, because a partial one is refused by the
#: terminal resolver and would make the case below pass for the wrong reason.
_CONTRACT_EVIDENCE: dict[str, object] = {
    "conveyed": True,
    "validated": True,
    "output_shape": "plain_text",
    "contract_sha256": "0" * 64,
    "channel": "response_contract",
}

_CAPTURE = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18956"
    / "completed_terminal_no_evidence.json"
)


def _captured_envelope(
    *, response_contract_evidence: dict[str, object] | None = None
) -> dict[str, object]:
    """Return the frozen capture, optionally with contract evidence added.

    Fails loudly rather than skipping when the capture is missing: a fixture
    that vanishes must not turn this into a green no-op.
    """
    if not _CAPTURE.is_file():
        raise AssertionError(f"frozen receipt capture missing: {_CAPTURE}")
    envelope: dict[str, object] = json.loads(_CAPTURE.read_text(encoding="utf-8"))
    if response_contract_evidence is not None:
        result = envelope["result"]
        assert isinstance(result, dict)
        terminal = result["terminal_payload"]["payload"]  # type: ignore[index]
        terminal["response_contract_evidence"] = response_contract_evidence
    return envelope


@pytest.mark.integration
def test_the_capture_is_the_shape_this_module_claims() -> None:
    """Positive control: the capture really is a completed terminal with no evidence.

    Without this, every assertion below would also pass against a capture
    that had quietly become something else.
    """
    envelope = _captured_envelope()
    terminal = envelope["result"]["terminal_payload"]["payload"]  # type: ignore[index]
    assert terminal["status"] == "completed"
    assert terminal["quality_gate_passed"] is True
    assert terminal.get("budget_evidence") is None
    assert terminal.get("response_contract_evidence") is None


@pytest.mark.integration
def test_the_lanes_own_terminal_is_accepted_when_no_contract_was_requested() -> None:
    """AC1 through the real entry point, on the real envelope shape."""
    require_budget, require_contract = _receipt_evidence_requirements(
        response_contract=None
    )
    error = _delegate_receipt_evidence_error(
        _Receipt(_captured_envelope()),
        require_budget_evidence=require_budget,
        require_contract_evidence=require_contract,
    )
    assert error is None, error


@pytest.mark.integration
def test_a_requested_contract_with_no_evidence_is_still_refused() -> None:
    """AC3. The positive control, and the reason this is not a deletion.

    Same envelope, same terminal, one difference: the request carried a
    response contract. If this returned None the change would have turned a
    false failure into a blind spot.
    """
    require_budget, require_contract = _receipt_evidence_requirements(
        response_contract={"type": "object"}
    )
    error = _delegate_receipt_evidence_error(
        _Receipt(_captured_envelope()),
        require_budget_evidence=require_budget,
        require_contract_evidence=require_contract,
    )
    assert error is not None
    assert "response_contract_evidence" in error


@pytest.mark.integration
def test_a_requested_contract_that_was_answered_passes() -> None:
    """The requirement is satisfiable, not merely enforceable."""
    require_budget, require_contract = _receipt_evidence_requirements(
        response_contract={"type": "object"}
    )
    error = _delegate_receipt_evidence_error(
        _Receipt(_captured_envelope(response_contract_evidence=_CONTRACT_EVIDENCE)),
        require_budget_evidence=require_budget,
        require_contract_evidence=require_contract,
    )
    assert error is None, error


@pytest.mark.integration
def test_the_receipt_callback_also_accepts_the_lanes_own_terminal(
    tmp_path: Path,
) -> None:
    """The residual: the CALLBACK arms the same refusal as the validator.

    The first OMN-18956 fix moved the validator's flags and left literals on
    this path, so the merge changed nothing observable and the next lane run
    failed with the message it always had. The validator case above passed
    throughout, which is exactly why this case exists: two entry points share
    one refusal, and covering one of them is not coverage.
    """
    require_budget, require_contract = _receipt_evidence_requirements(
        response_contract=None
    )
    _write_local_run_files(
        receipt=_Receipt(_captured_envelope()),
        state_root=tmp_path,
        prompt="summarize the following",
        task_type="document",
        task_type_resolution="fallback",
        addressing=ModelDelegateRunAddressing(
            locus=EnumDelegateLocus.DEPLOYED_LANE,
            bus="kafka",
            lane="dev",
        ),
        require_budget_evidence=require_budget,
        require_contract_evidence=require_contract,
    )


@pytest.mark.integration
def test_the_receipt_callback_still_refuses_a_requested_contract(
    tmp_path: Path,
) -> None:
    """Positive control on the callback path, matching the validator's."""
    from omnibase_infra.cli.delegate_terminal_resolver import (
        DelegateTerminalUnresolvedError,
    )

    require_budget, require_contract = _receipt_evidence_requirements(
        response_contract={"type": "object"}
    )
    with pytest.raises(
        DelegateTerminalUnresolvedError, match="response_contract_evidence"
    ):
        _write_local_run_files(
            receipt=_Receipt(_captured_envelope()),
            state_root=tmp_path,
            prompt="summarize the following",
            task_type="document",
            task_type_resolution="fallback",
            addressing=ModelDelegateRunAddressing(
                locus=EnumDelegateLocus.DEPLOYED_LANE,
                bus="kafka",
                lane="dev",
            ),
            require_budget_evidence=require_budget,
            require_contract_evidence=require_contract,
        )
