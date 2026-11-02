# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The timing wrapper advertises only a usable ungrouped reader."""

from collections.abc import Awaitable, Callable
from typing import cast, get_type_hints

import pytest

from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    ProtocolLocalRuntimeBus,
    UnsubscribeCallback,
)
from omnibase_core.protocols.runtime.protocol_local_runtime_message import (
    ProtocolLocalRuntimeMessage,
)
from omnibase_infra.cli.receipt_mode import DelegatePhaseStopwatch
from omnibase_infra.models.delegation.model_delegate_phase_timed_bus import (
    DelegatePhaseTimedBus,
    DelegatePhaseTimedUngroupedBus,
    wrap_delegate_bus,
)
from omnibase_infra.protocols.protocol_delegate_phase_stopwatch import (
    ProtocolDelegatePhaseStopwatch,
)

pytestmark = pytest.mark.unit


def _sync_reader(topic: object, callback: object) -> None:
    raise AssertionError("Capability inspection must not invoke the reader")


async def _missing_callback(topic: object) -> None:
    raise AssertionError("Capability inspection must not invoke the reader")


async def _extra_required(topic: object, callback: object, required: object) -> None:
    raise AssertionError("Capability inspection must not invoke the reader")


async def _keyword_only(topic: object, *, callback: object) -> None:
    raise AssertionError("Capability inspection must not invoke the reader")


async def _valid_reader(topic: object, callback: object) -> None:
    raise AssertionError("Capability inspection must not invoke the reader")


class _ReaderBus:
    def __init__(self, reader: object) -> None:
        self.subscribe_ungrouped = reader

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def publish(self, topic: str, key: object, value: bytes) -> object:
        return None

    async def subscribe(
        self,
        topic: str,
        *,
        on_message: Callable[[ProtocolLocalRuntimeMessage], Awaitable[None]],
        group_id: str,
    ) -> UnsubscribeCallback:
        async def unsubscribe() -> None:
            return None

        return unsubscribe


def _bus(reader: object) -> ProtocolLocalRuntimeBus:
    return _ReaderBus(reader)


@pytest.mark.parametrize(
    "reader",
    [42, _sync_reader, _missing_callback, _extra_required, _keyword_only],
    ids=[
        "noncallable",
        "synchronous",
        "missing-callback",
        "extra-required",
        "keyword-only",
    ],
)
def test_invalid_reader_keeps_grouped_wrapper(reader: object) -> None:
    wrapped = wrap_delegate_bus(_bus(reader), DelegatePhaseStopwatch())

    assert type(wrapped) is DelegatePhaseTimedBus


def test_uninspectable_reader_keeps_grouped_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _uninspectable(reader: object) -> None:
        raise ValueError("No signature available")

    monkeypatch.setattr(
        "omnibase_infra.models.delegation.model_delegate_phase_timed_bus.inspect.signature",
        _uninspectable,
    )

    assert type(wrap_delegate_bus(_bus(_valid_reader), DelegatePhaseStopwatch())) is (
        DelegatePhaseTimedBus
    )


def test_valid_async_reader_is_advertised_without_calling_it() -> None:
    assert isinstance(
        wrap_delegate_bus(_bus(_valid_reader), DelegatePhaseStopwatch()),
        DelegatePhaseTimedUngroupedBus,
    )


@pytest.mark.asyncio
async def test_async_reader_receives_original_arguments_and_unsubscribe() -> None:
    calls: list[tuple[str, object]] = []

    async def unsubscribe() -> None:
        return None

    async def reader(topic: str, callback: object) -> Callable[[], Awaitable[None]]:
        calls.append((topic, callback))
        return unsubscribe

    wrapped = wrap_delegate_bus(_bus(reader), DelegatePhaseStopwatch())
    assert isinstance(wrapped, DelegatePhaseTimedUngroupedBus)
    callback = cast("Callable[..., Awaitable[None]]", unsubscribe)

    assert await wrapped.subscribe_ungrouped("topic", callback) is unsubscribe
    assert calls == [("topic", callback)]


def test_wrapper_type_hints_resolve_without_cli_stopwatch() -> None:
    for target in (DelegatePhaseTimedBus.__init__, wrap_delegate_bus):
        assert get_type_hints(target)["stopwatch"] is ProtocolDelegatePhaseStopwatch
