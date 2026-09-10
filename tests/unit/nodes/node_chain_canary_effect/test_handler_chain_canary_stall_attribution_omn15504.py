# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-15504 -- a downstream stall must not be reported as an ingress fault.

An ingress that consumes its ENTIRE budget and then answers ``ok=false`` with a
dispatch-timeout has not failed. It accepted the request, published it, and
waited for a terminal nobody produced -- it is the component REPORTING the
silence, not causing it.

The ``terminal_missing`` verdict used to render that as "ingress also returned
ok=false (...)", which reads as an accusation and was taken as one. On
2026-09-10 the .201 dev lane's chain canary returned exactly that string while
the real defect was a rebalance livelock in the delegate-skill consumer one hop
downstream: its committed offset had been frozen since 11:56:43Z, so nothing
had terminalized for two hours. The ingress was the only component behaving
correctly and the only one the verdict named.

The discriminator is evidence the probe already collects: ``elapsed_ms``
against ``budget_ms``. An ingress that answers INSIDE its budget refused the
request, and that is a genuinely different finding, so it keeps its own wording.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
)
from tests.unit.nodes.node_chain_canary_effect.test_handler_chain_canary_readback import (
    _handler,
    _Ingress,
    _link,
    _request,
    _TerminalReadback,
)

_BUDGET_MS = 5_000

_DISPATCH_TIMEOUT_RESPONSE: dict[str, object] = {
    "ok": False,
    "error": {
        "code": "dispatch_timeout",
        "message": "Local runtime ingress timed out after 5000 ms",
        "retryable": True,
    },
}

_REFUSAL_RESPONSE: dict[str, object] = {
    "ok": False,
    "error": {
        "code": "invalid_task_type",
        "message": "task_type 'nonsense' is not routable",
        "retryable": False,
    },
}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_exhausted_ingress_is_not_blamed_for_the_missing_terminal() -> (
    None
):
    """The live 2026-09-10 shape: full budget spent, no terminal on the bus."""
    handler = _handler(
        _Ingress(
            response=_DISPATCH_TIMEOUT_RESPONSE,
            elapsed_ms=_BUDGET_MS + 126,
        ),
        terminal_readback=_TerminalReadback(found=""),
    )

    result = await handler.handle(_request(budget_ms=_BUDGET_MS))

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    detail = result.detail
    assert "NOT an ingress fault" in detail
    assert "accepted the request" in detail
    assert "look at the consumer" in detail
    # The accusatory phrasing that misdirected the investigation.
    assert "ingress also returned ok=false" not in detail


@pytest.mark.unit
@pytest.mark.asyncio
async def test_link_one_still_passes_when_the_ingress_answered() -> None:
    """Link 1 is reachability, and a budget-exhausted answer is reachable.

    Recorded as a test rather than assumed: the whole point of the wording fix
    is that link 1 was ALREADY honest here, and the per-link table disagreed
    with the prose verdict beside it.
    """
    handler = _handler(
        _Ingress(response=_DISPATCH_TIMEOUT_RESPONSE, elapsed_ms=_BUDGET_MS + 126),
        terminal_readback=_TerminalReadback(found=""),
    )

    result = await handler.handle(_request(budget_ms=_BUDGET_MS))

    assert _link(result, EnumChainLink.INGRESS_ACCEPTED) is EnumChainLinkStatus.PASS


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_ingress_that_refuses_inside_its_budget_keeps_its_own_wording() -> (
    None
):
    """The positive control: a real ingress refusal must still read as one.

    Without this, the fix above could be satisfied by never mentioning the
    ingress again, which would lose a finding rather than correct one.
    """
    handler = _handler(
        _Ingress(response=_REFUSAL_RESPONSE, elapsed_ms=37),
        terminal_readback=_TerminalReadback(found=""),
    )

    result = await handler.handle(_request(budget_ms=_BUDGET_MS))

    assert result.verdict is EnumChainCanaryVerdict.TERMINAL_MISSING
    detail = result.detail
    assert "refused this request" in detail
    assert "invalid_task_type" in detail
    assert "NOT an ingress fault" not in detail
