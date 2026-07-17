# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression: terminal-event correlator daemon threads must not leak (OMN-14708).

``LongLivedTerminalCorrelator`` starts a daemon thread named
``terminal-correlator-<handler>`` running a dedicated asyncio loop for its whole
lifetime. Tests that construct the consumer (directly, or via the auto-wiring
path that builds a ``TerminalEventConsumer``) and never close it leak that thread
into later tests in the shared single-process slice. Under CI load the leaked
loop-owning thread starves the event loop that
``test_kernel.py::TestBootstrap::test_bootstrap_uses_config_grace_period``
depends on, pushing it past its 60s timeout -- a nondeterministic hang that
greens on a clean dev slice.

These tests pin the two guarantees that make the leak class impossible to
reintroduce silently:

  1. ``close()`` deterministically joins the worker thread (no survivor), so the
     autouse ``reap_terminal_correlator_threads`` fixture in ``conftest.py`` can
     reliably reap any un-closed consumer; and
  2. ``close()`` is idempotent (the fixture double-closes a consumer a test
     already closed).
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from omnibase_infra.runtime.service_terminal_event_consumer import (
    LongLivedTerminalCorrelator,
    make_terminal_event_consumer,
)

pytestmark = [pytest.mark.unit]

_THREAD_PREFIX = "terminal-correlator-"


class _MinimalKafkaLikeBus:
    """Bus exposing only the attributes the correlator reads before first use."""

    config = SimpleNamespace(
        session_timeout_ms=45000,
        heartbeat_interval_ms=15000,
        max_poll_interval_ms=1800000,
        reconnect_backoff_ms=2000,
    )
    _bootstrap_servers = "omn-14708-teardown-test-broker"

    def _build_auth_kwargs(self) -> dict[str, object]:
        return {}


def _live_correlator_threads(handler_name: str) -> list[str]:
    return sorted(
        thread.name
        for thread in threading.enumerate()
        if thread.name == f"{_THREAD_PREFIX}{handler_name}"
    )


@pytest.mark.timeout(30)
def test_correlator_close_joins_worker_thread() -> None:
    """Constructing the correlator starts one worker thread; close() joins it."""
    handler_name = "HandlerOmn14708Direct"
    correlator = LongLivedTerminalCorrelator(
        event_bus=_MinimalKafkaLikeBus(), handler_name=handler_name
    )
    try:
        assert _live_correlator_threads(handler_name) == [
            f"{_THREAD_PREFIX}{handler_name}"
        ], "correlator did not start its named worker thread"
    finally:
        correlator.close()

    assert _live_correlator_threads(handler_name) == [], (
        "close() did not join the worker thread -- a leaked terminal-correlator "
        "daemon thread survives into later tests (OMN-14708)"
    )


@pytest.mark.timeout(30)
def test_correlator_close_is_idempotent() -> None:
    """A second close() is a no-op and never re-raises or re-leaks a thread."""
    handler_name = "HandlerOmn14708Idempotent"
    correlator = LongLivedTerminalCorrelator(
        event_bus=_MinimalKafkaLikeBus(), handler_name=handler_name
    )
    correlator.close()
    correlator.close()  # must not raise
    assert _live_correlator_threads(handler_name) == []


@pytest.mark.timeout(30)
def test_terminal_event_consumer_close_joins_correlator_thread() -> None:
    """The injected ``TerminalEventConsumer`` closes its owned correlator thread."""
    handler_name = "HandlerOmn14708Consumer"
    consumer = make_terminal_event_consumer(
        event_bus=_MinimalKafkaLikeBus(), handler_name=handler_name
    )
    try:
        assert _live_correlator_threads(handler_name) == [
            f"{_THREAD_PREFIX}{handler_name}"
        ]
    finally:
        consumer.close()

    assert _live_correlator_threads(handler_name) == []
