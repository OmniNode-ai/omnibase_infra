# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression coverage for OMN-13058: no thread/loop leak when ``open()`` fails.

``TerminalConsumerSession.__init__`` starts its dedicated worker loop thread
immediately. Before the fix, ``TerminalEventConsumer.open`` did
``session = TerminalConsumerSession(...); return session.open()`` with no
cleanup: any failure inside ``session.open()`` (consumer start timeout,
partition-assign timeout, broker auth error, missing event-bus attributes)
propagated out of the raising expression, the session reference was lost, and
the daemon worker thread plus its never-closed asyncio event loop leaked — one
pair per failed open. In the long-lived effects container the motivating caller
(``HandlerContextRoiRunner``) opens a session per trial, so a 160-560-trial
battery against a degraded broker accumulated hundreds of leaked threads.

The fix wraps ``session.open()`` in ``TerminalEventConsumer.open`` so the
session is closed (loop stopped, thread joined) before the failure re-raises.
This covers both the two-phase ``.open(topic)`` path and the legacy single-call
``__call__`` path, which opens through the same method.

Failure injection: an event bus exposing no ``_bootstrap_servers`` makes
``service_pattern_b_broker._direct_terminal_bootstrap_servers`` raise
``RuntimeError`` inside the submitted ``_open_positioned_consumer`` coroutine —
a real open-failure through the production code path, no Kafka required.
"""

from __future__ import annotations

import threading

import pytest

from omnibase_infra.runtime.service_terminal_event_consumer import (
    make_terminal_event_consumer,
)

pytestmark = pytest.mark.unit

_HANDLER_NAME = "Omn13058LeakProbe"
_THREAD_NAME = f"terminal-consumer-{_HANDLER_NAME}"
_TOPIC = "onex.evt.omninode.node-generation-completed.v1"


def _leaked_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name == _THREAD_NAME and thread.is_alive()
    ]


def test_open_failure_does_not_leak_worker_thread() -> None:
    """A failing two-phase open must close the session (thread joined)."""
    consumer = make_terminal_event_consumer(
        event_bus=object(),  # no _bootstrap_servers -> open fails fast
        handler_name=_HANDLER_NAME,
    )

    assert _leaked_threads() == []

    with pytest.raises(RuntimeError, match="_bootstrap_servers"):
        consumer.open(_TOPIC)

    assert _leaked_threads() == [], (
        "TerminalEventConsumer.open leaked its session worker thread after a "
        "failed open; the session must be closed before the failure re-raises "
        "(OMN-13058)."
    )


def test_single_call_open_failure_does_not_leak_worker_thread() -> None:
    """The legacy single-call form opens through the same guarded path."""
    consumer = make_terminal_event_consumer(
        event_bus=object(),
        handler_name=_HANDLER_NAME,
    )

    with pytest.raises(RuntimeError, match="_bootstrap_servers"):
        consumer(_TOPIC, "cid-omn-13058", 0.1)

    assert _leaked_threads() == []
