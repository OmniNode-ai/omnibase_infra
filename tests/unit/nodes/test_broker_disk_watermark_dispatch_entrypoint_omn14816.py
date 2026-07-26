# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""OMN-14816 — HandlerBrokerDiskWatermark is REACHABLE through the real dispatch path.

This is the RED-against-EXISTS-but-WRONG proof for the canonical def-B flip.

Before OMN-14816 the handler was contract-declared, wired, ingress-valid and CI-green
while exposing only the op-method ``probe_disk_watermark()`` and NO ``handle``.
Auto-wiring's ``_make_dispatch_callback`` looks for ``handle_async`` then ``handle``;
finding neither it binds ``_missing_handle``, which raises::

    ModelOnexError: Auto-wired handler HandlerBrokerDiskWatermark does not expose a
                    callable handle() or handle_async() dispatch entrypoint.

...on the FIRST dispatch. These tests drive the REAL production dispatch callback over
the REAL handler class (no fake handler, no patched entrypoint), so they FAIL against
the entrypoint-less pre-flip handler and PASS only once the canonical def-B ``handle``
exists. The node uses ``operation_match`` (no contract ``event_model``), so the callback
signature-introspects ``handle(inp: ModelBrokerDiskWatermarkInput)`` and coerces the
payload into that model at the adapter boundary (OMN-14716 parity with runtime_local).

Ticket: OMN-14816
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_broker_disk_watermark_compute.handlers.handler_broker_disk_watermark import (
    HandlerBrokerDiskWatermark,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.enum_disk_severity import (
    EnumDiskSeverity,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_watermark_input import (
    ModelBrokerDiskWatermarkInput,
)
from omnibase_infra.nodes.node_broker_disk_watermark_compute.models.model_broker_disk_watermark_output import (
    ModelBrokerDiskWatermarkOutput,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import _make_dispatch_callback

_GiB = 1024**3
_DOCKER_INFO = {"DockerRootDir": "/data"}


def _p0_disk_usage(path: str) -> tuple[int, int, int]:
    """96% used — above the p0 watermark (deterministic; no real docker/shutil)."""
    total = 100 * _GiB
    used = int(0.96 * total)
    return total, used, total - used


def _healthy_disk_usage(path: str) -> tuple[int, int, int]:
    """70% used — well below warn (deterministic)."""
    total = 100 * _GiB
    used = 70 * _GiB
    return total, used, total - used


def _input(cid: object, disk_usage: object) -> ModelBrokerDiskWatermarkInput:
    return ModelBrokerDiskWatermarkInput(
        correlation_id=cid,  # type: ignore[arg-type]
        docker_info_runner=lambda: _DOCKER_INFO,
        disk_usage_runner=disk_usage,  # type: ignore[arg-type]
    )


@pytest.mark.unit
def test_handler_exposes_handle_entrypoint() -> None:
    """The bare invariant: auto-wiring can only bind handle/handle_async.

    RED against the pre-OMN-14816 handler, which exposed only
    ``probe_disk_watermark``.
    """
    assert callable(getattr(HandlerBrokerDiskWatermark, "handle", None)), (
        "HandlerBrokerDiskWatermark exposes no handle(); auto-wiring binds "
        "_missing_handle and every dispatch raises ModelOnexError."
    )
    assert callable(getattr(HandlerBrokerDiskWatermark, "handle_async", None)), (
        "HandlerBrokerDiskWatermark must expose handle_async(); auto-wiring runs "
        "the synchronous docker/disk probes through the event-loop dispatch path."
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_dispatch_callback_returns_p0_output() -> None:
    """A real payload dispatched through the REAL callback reaches handle() and
    returns a P0 classification.

    Against the entrypoint-less handler this raises ModelOnexError (_missing_handle)
    rather than returning a result — that raise IS the bug, caught here.
    """
    cid = uuid4()
    callback = _make_dispatch_callback(HandlerBrokerDiskWatermark())
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=_input(cid, _p0_disk_usage),
        correlation_id=cid,
        event_type="ModelBrokerDiskWatermarkInput",
    )

    result = await callback(envelope)

    assert result is not None, "Dispatch produced no result — handle() never ran."
    assert isinstance(result, ModelDispatchResult)
    assert result.status is EnumDispatchStatus.SUCCESS
    assert len(result.output_events) == 1, (
        f"Expected exactly one ModelBrokerDiskWatermarkOutput; got "
        f"{result.output_events!r}"
    )
    out = result.output_events[0]
    assert isinstance(out, ModelBrokerDiskWatermarkOutput)
    assert out.correlation_id == cid
    assert out.max_severity is EnumDiskSeverity.P0
    assert "docker-data-root" in out.p0_labels
    assert out.docker_root_dir == "/data"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_dispatch_callback_returns_clean_output() -> None:
    """A healthy filesystem dispatched through the REAL callback returns CLEAN."""
    cid = uuid4()
    callback = _make_dispatch_callback(HandlerBrokerDiskWatermark())
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=_input(cid, _healthy_disk_usage),
        correlation_id=cid,
        event_type="ModelBrokerDiskWatermarkInput",
    )

    result = await callback(envelope)

    assert result is not None
    assert isinstance(result, ModelDispatchResult)
    out = result.output_events[0]
    assert isinstance(out, ModelBrokerDiskWatermarkOutput)
    assert out.correlation_id == cid
    assert out.max_severity is EnumDiskSeverity.CLEAN
    assert out.p0_labels == ()
    assert out.warn_labels == ()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_real_dispatch_callback_runs_blocking_probe_in_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runtime callback binds handle_async, which delegates handle via to_thread."""
    cid = uuid4()
    delegated: list[object] = []

    async def _record_to_thread(
        func: object, /, *args: object, **kwargs: object
    ) -> object:
        delegated.append(func)
        return func(*args, **kwargs)  # type: ignore[misc]

    monkeypatch.setattr(asyncio, "to_thread", _record_to_thread)

    handler = HandlerBrokerDiskWatermark()
    callback = _make_dispatch_callback(handler)
    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
        payload=_input(cid, _healthy_disk_usage),
        correlation_id=cid,
        event_type="ModelBrokerDiskWatermarkInput",
    )

    result = await callback(envelope)

    assert result is not None
    assert delegated == [handler.handle]
