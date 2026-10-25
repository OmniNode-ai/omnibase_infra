# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read the dev work ledger through its node, using the issued machine identity.

The broker enforces the principal's ACL. This client never infers authority
from a lane name or a caller-supplied principal. It subscribes only to the
completed topic under the group's granted prefix, establishes its starting
offset before publishing, and returns the whole answer (including freshness
and parity). Domain request/answer models remain owned by omnimarket.
"""

from __future__ import annotations

import asyncio
import math
from importlib.resources import files
from types import TracebackType
from typing import Self
from uuid import UUID, uuid4

import yaml
from pydantic import JsonValue, TypeAdapter

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumKafkaAcks,
)
from omnibase_infra.errors.error_infra import WorkLedgerQueryError
from omnibase_infra.event_bus.kafka_transport import KafkaTransport
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.model_work_ledger_query_client_config import (
    ModelWorkLedgerQueryClientConfig,
)

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


def load_work_ledger_query_client_config() -> ModelWorkLedgerQueryClientConfig:
    """Load the packaged declaration, also checked against the derived ACLs."""
    raw = yaml.safe_load(
        files("omnibase_infra")
        .joinpath("contracts/work_ledger_query_client.contract.yaml")
        .read_text(encoding="utf-8")
    )
    return ModelWorkLedgerQueryClientConfig(
        **raw["client"],
        command_topic=raw["event_bus"]["publish_topics"][0],
        completed_topic=raw["event_bus"]["subscribe_topics"][0],
    )


class HandlerWorkLedgerQuery:
    """Bounded reads with a private reply consumer group per connection.

    The CLI resolves the dev transport and credential before construction.
    Kafka's topic/group authorization errors propagate; they never produce an
    empty ledger answer. No offsets are committed and no topics are created.
    Use ``async with client`` to reuse one connection for multiple queries.
    Queries in a session are serialized so they cannot consume each other's
    replies. A standalone query owns and closes its connection as before.
    """

    def __init__(self, transport: ModelKafkaEventBusConfig) -> None:
        self._transport = transport
        self._contract = load_work_ledger_query_client_config()
        self._client: KafkaTransport | None = None
        self._session_open = False
        self._query_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        """Open a reusable session; connect only after a valid first request."""
        async with self._query_lock:
            if self._session_open:
                raise RuntimeError("Work-ledger query session is already open")
            self._session_open = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Finish an in-flight read and release the session's clients."""
        async with self._query_lock:
            self._session_open = False
            await self._close_transport()

    async def _close_transport(self) -> None:
        transport, self._client = self._client, None
        if transport is not None:
            await transport.close()

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure transport for a work-ledger read."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Side-effecting command/reply I/O."""
        return EnumHandlerTypeCategory.EFFECT

    async def query(
        self, request: dict[str, JsonValue], *, timeout_s: float = 30.0
    ) -> dict[str, JsonValue]:
        """Preserve the reader's correlation and await its answer within the deadline."""
        supplied_id = request.get("correlation_id")
        if supplied_id is not None and not isinstance(supplied_id, str):
            raise WorkLedgerQueryError("Request correlation_id must be a UUID string")
        try:
            correlation_id = UUID(supplied_id) if supplied_id is not None else uuid4()
        except ValueError as exc:
            raise WorkLedgerQueryError(
                "Request correlation_id must be a UUID string"
            ) from exc
        kind = request.get("query")
        if kind not in self._contract.queries:
            raise WorkLedgerQueryError(
                "Unsupported work-ledger query", correlation_id=correlation_id
            )
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        payload = {**request, "correlation_id": str(correlation_id)}
        command = self._contract.command_topic
        completed = self._contract.completed_topic
        parts = self._contract.command_topic.split(".")
        envelope = ModelEventEnvelope[dict[str, JsonValue]](
            payload=payload,
            correlation_id=correlation_id,
            event_type=f"{parts[2]}.{parts[3]}",
            tenant_id=None,
        )
        async with asyncio.timeout(timeout_s):
            async with self._query_lock:
                try:
                    if self._client is None:
                        self._client = KafkaTransport(
                            config=self._transport.model_copy(
                                update={
                                    "acks": EnumKafkaAcks.ALL,
                                    "enable_idempotence": False,
                                }
                            ),
                            group=self._contract.consumer_group_prefix + uuid4().hex,
                            topics=(completed,),
                            auto_offset_reset="latest",
                        )
                        # Establish reply offsets before the first publish.
                        await self._client.start()
                    transport = self._client
                    await transport.send(
                        command,
                        key=str(correlation_id).encode("utf-8"),
                        value=envelope.model_dump_json().encode("utf-8"),
                        headers={},
                    )
                    while True:
                        messages = await transport.poll(max_messages=1, timeout_ms=1000)
                        if not messages:
                            continue
                        message = messages[0]
                        try:
                            raw = _JSON_OBJECT.validate_json(message.value)
                        except ValueError:
                            # A shared terminal can carry unrelated malformed records.
                            # They cannot answer our fresh request.
                            continue
                        if raw.get("correlation_id") != str(correlation_id):
                            continue
                        answer = raw.get("payload")
                        if not isinstance(answer, dict):
                            raise WorkLedgerQueryError(
                                "Ledger reply has no object payload",
                                correlation_id=correlation_id,
                            )
                        if answer.get("correlation_id") != str(correlation_id):
                            raise WorkLedgerQueryError(
                                "Ledger reply correlation mismatch",
                                correlation_id=correlation_id,
                            )
                        if answer.get("query") != kind:
                            raise WorkLedgerQueryError(
                                "Ledger reply query mismatch",
                                correlation_id=correlation_id,
                            )
                        if answer.get("error"):
                            # The domain error can contain database driver details.
                            # Keep those off stderr; callers get a named refusal.
                            raise WorkLedgerQueryError(
                                "Ledger node reported a database error",
                                correlation_id=correlation_id,
                            )
                        if not isinstance(
                            answer.get("freshness"), dict
                        ) or not isinstance(answer.get("parity"), dict):
                            raise WorkLedgerQueryError(
                                "Ledger reply lacks freshness or parity",
                                correlation_id=correlation_id,
                            )
                        return answer
                except BaseException:
                    # A failed read must not leave a broken session connection.
                    await self._close_transport()
                    raise
                finally:
                    if not self._session_open:
                        await self._close_transport()
