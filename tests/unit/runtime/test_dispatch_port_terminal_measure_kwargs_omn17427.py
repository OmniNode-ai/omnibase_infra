# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The dispatch port accepts the terminal-measure keywords omnimarket passes (OMN-17427).

omnimarket#4252 (OMN-19448) made the delegate-skill handler pass ``trace_id`` and
``queue_wait_ms`` to ``dispatch()`` whenever the port declares them. The
consumer-kwarg-parity hook caught the missing declaration on every omnibase_infra
commit. Both keywords are evidence only: the canonical delegation request wire
(``ModelDelegationRequest``, ``extra="forbid"``) carries neither, so the port
accepts them and does not publish them -- the same boundary as ``attribution``.
Both default to ``None`` and the protocol and the port must agree on the type
and the default, because the two repositories deploy independently.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from uuid import UUID

import pytest

from omnibase_infra.runtime.protocols.protocol_delegation_dispatch_port import (
    ProtocolDelegationDispatchPort,
)
from omnibase_infra.runtime.service_delegation_dispatch_port import (
    RuntimeDelegationDispatchPort,
)

pytestmark = pytest.mark.unit

_TERMINAL_MEASURE_KWARGS = ("trace_id", "queue_wait_ms")

_DEPLOYED_CALLER_KWARGS: dict[str, object] = {
    "prompt": "a prompt",
    "task_type": "document",
    "correlation_id": None,
    "max_tokens": None,
    "source_file_path": None,
    "source_session_id": None,
    "wait": True,
    "quality_contract_mode": "extend_task_class",
    "acceptance_criteria": (),
}


@pytest.mark.parametrize(
    "dispatch",
    [
        RuntimeDelegationDispatchPort.dispatch,
        ProtocolDelegationDispatchPort.dispatch,
    ],
    ids=["the runtime port", "the protocol it satisfies"],
)
class TestTheTerminalMeasureKeywordsAreAccepted:
    def test_both_keywords_are_declared_and_default_to_none(
        self, dispatch: Callable[..., object]
    ) -> None:
        parameters = inspect.signature(dispatch).parameters
        for name in _TERMINAL_MEASURE_KWARGS:
            assert name in parameters, name
            assert parameters[name].default is None, name

    def test_a_caller_that_passes_them_binds(
        self, dispatch: Callable[..., object]
    ) -> None:
        signature = inspect.signature(dispatch)
        trace_id = UUID("11111111-1111-4111-8111-111111111111")
        bound = signature.bind(
            object(), **_DEPLOYED_CALLER_KWARGS, trace_id=trace_id, queue_wait_ms=42
        )
        assert bound.arguments["trace_id"] == trace_id
        assert bound.arguments["queue_wait_ms"] == 42

    def test_a_caller_that_passes_neither_still_binds(
        self, dispatch: Callable[..., object]
    ) -> None:
        signature = inspect.signature(dispatch)
        signature.bind(object(), **_DEPLOYED_CALLER_KWARGS)


def test_the_port_and_the_protocol_agree_on_annotation_and_default() -> None:
    runtime = inspect.signature(RuntimeDelegationDispatchPort.dispatch).parameters
    protocol = inspect.signature(ProtocolDelegationDispatchPort.dispatch).parameters
    for name in _TERMINAL_MEASURE_KWARGS:
        assert runtime[name].default == protocol[name].default, name
        assert str(runtime[name].annotation) == str(protocol[name].annotation), name
