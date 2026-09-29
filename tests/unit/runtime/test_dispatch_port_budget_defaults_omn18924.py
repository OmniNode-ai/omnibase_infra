# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The dispatch port accepts a caller that passes no budget (OMN-18924).

THE SECOND FACE OF ONE SKEW. OMN-15504 added `execution_timeout_seconds` and
`terminal_delivery_margin_seconds` to `dispatch()` as REQUIRED keyword-only
arguments. The caller deployed on the dev lane passes neither, so every
delegation there terminalized `provider_error` carrying a TypeError naming
both names -- measured on the 19:44Z chain canary, correlation
`b267d3bd-0f60-466e-8c8a-e7c60446e1f0`. Same change, same
producer-before-consumer inversion and same day as the absent
`execution_budgets` map the sibling module covers; the map refused the CLI
before dispatch, this refused the lane after it.

WHY DEFAULTED RATHER THAN REQUIRED. A new argument lands consumer-first or is
excluded when unset. Requiring it of a caller that does not yet pass it makes
the failure the caller's, and here the caller is a different repository on a
different release cadence. The defaults are the contract budget, so a caller
that does pass them is unaffected.

THIS EXACT SHAPE HAS HAPPENED BEFORE ON THIS SIGNATURE. The OMN-18321 comment
in the protocol records a kwarg added on one side alone, a TypeError on the
deployed bus path swallowed by the consumer's `except Exception` into a failed
terminal, and the dev-lane chain dying silently for a day. That is this
incident with the direction reversed, which is why the assertion below is on
the CALL SHAPE the deployed caller actually uses rather than on the two
defaults alone.
"""

from __future__ import annotations

import inspect

import pytest

from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_TERMINAL_DELIVERY_MARGIN_SECONDS,
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.unit

#: The keyword names the deployed caller does NOT pass.
_BUDGET_KWARGS = (
    "execution_timeout_seconds",
    "terminal_delivery_margin_seconds",
)

#: Every keyword the deployed omnimarket caller DOES pass, read off
#: `handler_delegate_skill.py`'s call site. Binding this set is the real
#: assertion: two defaults prove nothing if the caller is short a third
#: argument nobody noticed.
_DEPLOYED_CALLER_KWARGS = {
    "prompt": "a prompt",
    "task_type": "document",
    "correlation_id": None,
    "max_tokens": None,
    "source_file_path": None,
    "source_session_id": None,
    "wait": True,
    "quality_contract_mode": "extend_task_class",
    "acceptance_criteria": (),
    "tenant_id": None,
    "provenance": None,
}


@pytest.mark.parametrize(
    "dispatch",
    [
        RuntimeDelegationDispatchPort.dispatch,
        ProtocolDelegationDispatchPort.dispatch,
    ],
    ids=["the runtime port", "the protocol it satisfies"],
)
class TestTheBudgetArgumentsAreOptional:
    def test_both_declare_a_default(self, dispatch: object) -> None:
        parameters = inspect.signature(dispatch).parameters  # type: ignore[arg-type]
        for name in _BUDGET_KWARGS:
            assert name in parameters, name
            assert parameters[name].default is not inspect.Parameter.empty, name

    def test_the_deployed_call_shape_binds(self, dispatch: object) -> None:
        """The reproduction: the caller's own kwargs, and nothing else.

        Before this change `bind` raised `TypeError: missing 2 required
        keyword-only arguments`, which is verbatim what the lane recorded.
        """
        signature = inspect.signature(dispatch)  # type: ignore[arg-type]
        signature.bind(object(), **_DEPLOYED_CALLER_KWARGS)

    def test_a_caller_that_passes_them_is_unaffected(self, dispatch: object) -> None:
        """Positive control: defaulting must not stop an explicit value binding."""
        signature = inspect.signature(dispatch)  # type: ignore[arg-type]
        bound = signature.bind(
            object(),
            **_DEPLOYED_CALLER_KWARGS,
            execution_timeout_seconds=11,
            terminal_delivery_margin_seconds=7,
        )
        assert bound.arguments["execution_timeout_seconds"] == 11
        assert bound.arguments["terminal_delivery_margin_seconds"] == 7


class TestTheDefaultsAgree:
    """The port and its protocol declare one default, not two.

    The CLI no longer carries a default budget of its own (OMN-19407): it
    reads every class's budget from the task-class contract, which declares
    one for every class. What is left to pin is that the runtime port and its
    protocol cannot disagree with each other.
    """

    def test_the_port_and_the_protocol_agree(self) -> None:
        """A default on one and a different default on the other is worse than none."""
        runtime = inspect.signature(RuntimeDelegationDispatchPort.dispatch).parameters
        protocol = inspect.signature(ProtocolDelegationDispatchPort.dispatch).parameters
        for name in _BUDGET_KWARGS:
            assert runtime[name].default == protocol[name].default, name
