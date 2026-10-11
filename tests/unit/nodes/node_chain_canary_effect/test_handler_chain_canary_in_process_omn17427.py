# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canary reports terminal-only evidence immediately without claiming a pass."""

import inspect
from unittest.mock import AsyncMock

import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    _replay_ledger_chain_via_asyncpg,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_ledger_replay_status import (
    EnumLedgerReplayStatus,
)
from tests.unit.nodes.node_chain_canary_effect.test_handler_chain_canary_ledger import (
    _FULL_CHAIN,
    _handler,
    _link,
    _request,
)

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("state", "hops", "green", "status", "verdict", "link"),
    [
        (
            "in_process_terminal_only",
            ("terminal",),
            False,
            EnumLedgerReplayStatus.IN_PROCESS_TERMINAL_ONLY,
            EnumChainCanaryVerdict.LEDGER_CHAIN_IN_PROCESS_TERMINAL_ONLY,
            EnumChainLinkStatus.NOT_EVALUATED,
        ),
        (
            "incomplete",
            _FULL_CHAIN[:-1],
            False,
            EnumLedgerReplayStatus.CHAIN_INCOMPLETE,
            EnumChainCanaryVerdict.LEDGER_CHAIN_INCOMPLETE,
            EnumChainLinkStatus.FAIL,
        ),
        (
            "complete",
            _FULL_CHAIN,
            True,
            EnumLedgerReplayStatus.VERIFIED,
            EnumChainCanaryVerdict.GREEN,
            EnumChainLinkStatus.PASS,
        ),
        (
            "",
            _FULL_CHAIN,
            True,
            EnumLedgerReplayStatus.VERIFIED,
            EnumChainCanaryVerdict.GREEN,
            EnumChainLinkStatus.PASS,
        ),
    ],
)
async def test_labels(state, hops, green, status, verdict, link, monkeypatch):
    monkeypatch.setenv(
        "CHAIN_CANARY_PROJECTION_DSN", "postgresql://probe@db.invalid/test"
    )
    replay = AsyncMock(return_value=(hops, green, "pass", state, ""))
    handler = _handler(replay)
    request = _request()
    actual, detail = await handler._replay_ledger(
        request, str(request.correlation_id), 0.01 if state == "incomplete" else 30
    )
    assert actual is status
    assert replay.await_count == 1
    if state == "in_process_terminal_only":
        assert "terminal" in detail
    result = await handler.handle(request)
    assert result.verdict is verdict
    assert result.success is (verdict is EnumChainCanaryVerdict.GREEN)
    assert _link(result, EnumChainLink.LEDGER_REPLAY) is link


def test_reader_selects_chain_state():
    source = inspect.getsource(_replay_ledger_chain_via_asyncpg)
    assert (
        "SELECT hop, replay_green, verifier_verdict, chain_state FROM public.ledger_chain"
        in source
    )


@pytest.mark.asyncio
async def test_asyncpg_returns_last_chain_state(monkeypatch):
    import asyncpg

    connection = AsyncMock()
    connection.fetch.return_value = [
        {
            "hop": "terminal",
            "replay_green": False,
            "verifier_verdict": "pass",
            "chain_state": "in_process_terminal_only",
        }
    ]
    monkeypatch.setattr(asyncpg, "connect", AsyncMock(return_value=connection))
    assert await _replay_ledger_chain_via_asyncpg("unused", "correlation", 10) == (
        ("terminal",),
        False,
        "pass",
        "in_process_terminal_only",
        "",
    )
    assert "chain_state" in connection.fetch.await_args.args[0]


def test_reader_token_is_the_writers_state() -> None:
    """The canary's literal is the token the chain writer persists."""
    from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
        _CHAIN_STATE_IN_PROCESS_TERMINAL_ONLY,
    )
    from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
        EnumLedgerChainState,
    )

    assert (
        EnumLedgerChainState.IN_PROCESS_TERMINAL_ONLY.value
        == _CHAIN_STATE_IN_PROCESS_TERMINAL_ONLY
    )
