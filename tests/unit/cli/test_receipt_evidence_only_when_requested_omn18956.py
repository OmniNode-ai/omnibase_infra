# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Receipt evidence is required only when the request asked for it (OMN-18956).

THE DEFECT. `_require_completed_terminal_evidence` documents itself as
refusing "a completed receipt that lacks evidence ITS REQUEST REQUIRED", and
its only caller passed both flags as literal `True`. So every completed
delegation was required to carry `budget_evidence` and
`response_contract_evidence` regardless of what was asked for, and the
deployed producer emits neither.

The consequence was measured on the dev lane at 2026-09-21T02:25:54Z,
correlation `0b21b5f0-fb56-4ab6-9219-24bbfd841f35`: the terminal was
`completed`, the quality gate passed at 1.0, the projection row was written
-- and the CLI still exited non-zero on `completed delegation terminal omits
required evidence: budget_evidence, response_contract_evidence`. Nothing about
the delegation failed; the receipt assertion did. Every lane brief in this
session cites a delegate receipt, so an exit status that disagrees with the
terminal is not cosmetic: a caller branching on it reads success as failure.

Third surface of `omnibase_infra#3882` (OMN-15504), after the absent
`execution_budgets` map and the two required dispatch keyword arguments, both
fixed under OMN-18924. Same producer-before-consumer inversion each time.

WHAT IS NOT RELAXED, and this is the half worth reading. The check itself is
untouched and still refuses. What changes is who turns it on: a response
contract in the request arms the contract half, and nothing arms it when no
contract was asked for. `TestTheCheckStillRefusesWhenItIsAsked` is the
positive control that keeps this from quietly becoming a no-op, which would
trade a false failure for a blind spot.
"""

from __future__ import annotations

import pytest

from omnibase_infra.cli.cli_delegate import (
    DelegateTerminalUnresolvedError,
    _require_completed_terminal_evidence,
)
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal

pytestmark = pytest.mark.unit


def _terminal(status: str = "completed") -> ModelDelegateTerminal:
    """A terminal shaped like the one the lane actually produced.

    Deliberately carries NO budget or contract evidence, because that is the
    shape the deployed producer emits and the shape that was being refused.
    """
    return ModelDelegateTerminal.model_validate(
        {
            # `attempts` is the model's only required field, by design: it is
            # what identifies the object as a delegation terminal at all.
            "attempts": [],
            "status": status,
            "quality_gate_passed": status == "completed",
        }
    )


class TestAnUnaskedCompletedTerminalIsAccepted:
    """AC1. The reproduction, at the function boundary."""

    def test_neither_kind_asked_for_means_no_refusal(self) -> None:
        _require_completed_terminal_evidence(
            _terminal(),
            require_budget_evidence=False,
            require_contract_evidence=False,
        )

    def test_a_non_completed_terminal_is_never_refused_here(self) -> None:
        """Unchanged behaviour, pinned so the guard below cannot widen onto it."""
        _require_completed_terminal_evidence(
            _terminal(status="failed"),
            require_budget_evidence=True,
            require_contract_evidence=True,
        )


class TestTheCheckStillRefusesWhenItIsAsked:
    """AC3. Tolerating the un-asked case must not disarm the asked case.

    Without these, the change is indistinguishable from deleting the check,
    which trades a false failure for a blind spot.
    """

    def test_a_requested_contract_with_no_evidence_is_refused(self) -> None:
        with pytest.raises(
            DelegateTerminalUnresolvedError, match="response_contract_evidence"
        ):
            _require_completed_terminal_evidence(
                _terminal(),
                require_budget_evidence=False,
                require_contract_evidence=True,
            )

    def test_a_requested_budget_with_no_evidence_is_refused(self) -> None:
        with pytest.raises(DelegateTerminalUnresolvedError, match="budget_evidence"):
            _require_completed_terminal_evidence(
                _terminal(),
                require_budget_evidence=True,
                require_contract_evidence=False,
            )

    def test_both_missing_are_named_in_one_refusal(self) -> None:
        with pytest.raises(DelegateTerminalUnresolvedError) as caught:
            _require_completed_terminal_evidence(
                _terminal(),
                require_budget_evidence=True,
                require_contract_evidence=True,
            )
        message = str(caught.value)
        assert "budget_evidence" in message
        assert "response_contract_evidence" in message


class TestTheCallSiteAsksOnlyForWhatTheRequestCarried:
    """AC2. The docstring and the call site must agree.

    The flags are derived from the request rather than written as literals,
    and this reads the derivation rather than the behaviour so a future edit
    back to `True` is a red test rather than another silent outage.
    """

    @staticmethod
    def _derive(response_contract: dict[str, object] | None) -> tuple[bool, bool]:
        from omnibase_infra.cli.cli_delegate import (
            _receipt_evidence_requirements,
        )

        return _receipt_evidence_requirements(response_contract=response_contract)

    def test_no_contract_in_the_request_asks_for_no_contract_evidence(self) -> None:
        _, require_contract = self._derive(None)
        assert require_contract is False

    def test_a_contract_in_the_request_asks_for_contract_evidence(self) -> None:
        _, require_contract = self._derive({"type": "object"})
        assert require_contract is True

    def test_budget_evidence_is_never_demanded_from_the_request_today(self) -> None:
        """Stated as a fact about the request surface, not as an opinion.

        Nothing a caller can pass expresses "I require budget evidence": the
        budget comparison happens only when the BACKEND declares
        `max_grounded_input_tokens`, which the CLI cannot know when it builds
        the request. So the honest derivation is False, and the parameter
        stays wired for the day a request-side demand exists. The check
        itself remains armed and is proven refusing in the class above.
        """
        for contract in (None, {"type": "object"}):
            require_budget, _ = self._derive(contract)
            assert require_budget is False
