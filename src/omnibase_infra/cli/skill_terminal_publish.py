# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Publish a skill's returned terminal result to a declared lane (OMN-19152).

``onex skill`` dispatches on the in-memory bus. That is deliberate: the
verification a skill performs must never depend on a broker. It also meant
that what the node returned -- for ``dod_verify``, the definition-of-done
verdict -- died with the process, so the durable verdict table (OMN-18900)
stayed empty on the path production actually runs.

A skill mapping may now name a lane (``publish_terminal_to_lane``). After the
in-memory dispatch has returned and its receipt has been printed, the node's
returned result is published to the contract's own ``terminal_event`` topic on
that lane's broker, wrapped by :func:`envelope_terminal_payload` so it has
field parity with the runtime's success-path applier.

Fail-soft by construction: :func:`publish_skill_terminal_event` never raises.
A missing lane declaration, a missing identity, an unreachable broker and a
timeout each come back as a ``FAILED`` report, and the caller keeps the exit
code the dispatch produced.

The broker address and the SASL identity are resolved exactly as
``onex delegate --lane <lane>`` resolves them; nothing here reads an ambient
``KAFKA_BOOTSTRAP_SERVERS`` (OMN-16871).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Coroutine
from contextlib import nullcontext
from pathlib import Path
from uuid import UUID

import yaml
from pydantic import JsonValue

from omnibase_infra.cli.delegate_lane import resolve_lane_target
from omnibase_infra.cli.delegate_lane_credentials import (
    resolve_lane_client_transport_for,
)
from omnibase_infra.cli.enum_skill_terminal_publish_outcome import (
    EnumSkillTerminalPublishOutcome,
)
from omnibase_infra.cli.model_skill_terminal_publish_report import (
    ModelSkillTerminalPublishReport,
)
from omnibase_infra.cli.model_skill_terminal_publish_target import (
    ModelSkillTerminalPublishTarget,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bind_lane_client_transport,
)
from omnibase_infra.runtime.contract_terminal_events import envelope_terminal_payload
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

__all__ = [
    "SkillTerminalPublisher",
    "publish_skill_terminal_event",
    "resolve_skill_terminal_target",
]


SkillTerminalPublisher = Callable[
    [ModelSkillTerminalPublishTarget, str, bytes | None, bytes, float],
    Coroutine[object, object, None],
]


def resolve_skill_terminal_target(
    lane: str, *, omni_home: Path | None
) -> ModelSkillTerminalPublishTarget:
    """Resolve a lane to its declared broker and this machine's identity for it.

    ``omni_home`` is the workspace root the lane declaration is read from
    (``--omnibase-path`` / ``$OMNIBASE_PATH``, as ``onex delegate`` reads it).
    Raises when the lane cannot be resolved (no root, no declaration, an
    unreadable identity); the caller reports that as a failed publish.
    """
    selection = resolve_lane_target(
        bus="kafka",
        lane=lane,
        kafka_bootstrap=None,
        omni_home=omni_home,
    )
    if selection is None:
        raise ValueError(f"lane {lane!r} resolved to no broker")
    transport = resolve_lane_client_transport_for(
        lane_target=selection,
        onex_home=Path.home() / ".onex",
        environ=os.environ,
    )
    return ModelSkillTerminalPublishTarget(
        lane=lane,
        bootstrap_servers=selection.bootstrap_servers,
        transport=transport,
    )


async def _publish_to_kafka(
    target: ModelSkillTerminalPublishTarget,
    topic: str,
    key: bytes | None,
    value: bytes,
    timeout_seconds: float,
) -> None:
    """Publish one message, bounded by ``timeout_seconds`` per broker call."""
    binding = (
        bind_lane_client_transport(target.transport)
        if target.transport is not None
        else nullcontext()
    )
    with binding:
        bus = EventBusKafka.from_bootstrap(target.bootstrap_servers)
        try:
            await asyncio.wait_for(bus.start(), timeout_seconds)
            await asyncio.wait_for(bus.publish(topic, key, value), timeout_seconds)
        finally:
            await asyncio.wait_for(bus.close(), timeout_seconds)


def _handler_result(receipt: object, result_model: str) -> dict[str, JsonValue] | None:
    """The node's returned result, from either receipt shape.

    A success-like run carries it as the receipt's ``result``. A run that
    exited non-zero -- which a failed verdict does -- carries it inside the
    failure summary's ``handler_result``. Both are published: a failure is as
    durable as a pass.
    """
    result = getattr(receipt, "result", None)
    if getattr(receipt, "result_model", None) == result_model and isinstance(
        result, dict
    ):
        return result
    handler_result = getattr(result, "handler_result", None)
    if isinstance(handler_result, dict):
        return handler_result
    return None


def _terminal_topic(contract_path: Path) -> str:
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    topic = contract.get("terminal_event") if isinstance(contract, dict) else None
    return topic.strip() if isinstance(topic, str) else ""


def publish_skill_terminal_event(
    *,
    receipt: object,
    result_model: str,
    contract_path: Path,
    lane: str,
    resolve_target: Callable[[str], ModelSkillTerminalPublishTarget],
    publisher: SkillTerminalPublisher | None = None,
    timeout_seconds: float = 5.0,
) -> ModelSkillTerminalPublishReport:
    """Publish the receipt's handler result to its terminal topic. Never raises."""
    topic = ""
    correlation_id: UUID | None = None
    try:
        result = _handler_result(receipt, result_model)
        if result is None:
            return ModelSkillTerminalPublishReport(
                outcome=EnumSkillTerminalPublishOutcome.SKIPPED_NO_RESULT, lane=lane
            )
        topic = _terminal_topic(contract_path)
        if not topic:
            return ModelSkillTerminalPublishReport(
                outcome=EnumSkillTerminalPublishOutcome.SKIPPED_NO_TERMINAL_TOPIC,
                lane=lane,
            )
        correlation_id = UUID(str(result["correlation_id"]))
        value = envelope_terminal_payload(
            topic=topic,
            payload=json.dumps(result).encode("utf-8"),
            terminal_topics={topic},
        )
        target = resolve_target(lane)
        publish = publisher or _publish_to_kafka
        asyncio.run(
            publish(target, topic, str(correlation_id).encode(), value, timeout_seconds)
        )
    except Exception as exc:  # noqa: BLE001 -- boundary: a publish never fails the verification
        return ModelSkillTerminalPublishReport(
            outcome=EnumSkillTerminalPublishOutcome.FAILED,
            lane=lane,
            topic=topic,
            correlation_id=correlation_id,
            detail=f"{type(exc).__name__}: {sanitize_error_message(exc)}",
        )
    return ModelSkillTerminalPublishReport(
        outcome=EnumSkillTerminalPublishOutcome.PUBLISHED,
        lane=lane,
        topic=topic,
        correlation_id=correlation_id,
    )
