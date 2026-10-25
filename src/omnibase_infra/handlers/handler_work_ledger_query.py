# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Correlated database-ledger reads using the developer principal's derived ACL.

Authorization belongs to the broker's authenticated principal. A payload cannot
assert a principal or substitute the runtime's credential. Callers retain the
node's freshness and parity metadata; a database error is never an empty answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from types import TracebackType
from uuid import UUID, uuid4

from pydantic import BaseModel, ValidationError

from omnibase_core.event_bus.util_consumer_group import compute_consumer_group_id
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.cli.contract_registry import RegistryUnresolvedError, node_contract
from omnibase_infra.cli.delegate_lane import resolve_lane_selection_at
from omnibase_infra.cli.delegate_lane_credentials import (
    resolve_lane_client_transport_for,
)
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.lane_client_transport_binding import (
    bind_lane_client_transport,
)
from omnibase_infra.event_bus.models.model_event_message import ModelEventMessage
from omnibase_infra.models.model_work_ledger_query_contract import (
    ModelWorkLedgerQueryContract,
)


class WorkLedgerQueryError(RuntimeError):
    """The read node could not provide a usable answer."""


class HandlerWorkLedgerQuery:
    """One subscription per client, with a unique group inside the granted prefix.

    The supplied bus must already be started using the caller's lane credential.
    This client owns its subscription and pending calls, but never owns the bus.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        """Infrastructure transport handler for the existing query node."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """The query transport performs external I/O."""
        return EnumHandlerTypeCategory.EFFECT

    def __init__(
        self,
        bus: EventBusKafka,
        *,
        contract: ModelWorkLedgerQueryContract | None = None,
    ) -> None:
        self._bus = bus
        self.contract = contract or load_work_ledger_query_contract()
        prefix = compute_consumer_group_id(
            env="local",
            service="omnimarket",
            node_name="work_ledger_query_client",
            version="v1",
        )
        self.group_id = f"{prefix}.{uuid4().hex}"
        self._unsubscribe: Callable[[], Awaitable[None]] | None = None
        self._pending: dict[UUID, asyncio.Future[BaseModel]] = {}

    async def __aenter__(self) -> HandlerWorkLedgerQuery:
        if self._unsubscribe is not None:
            raise WorkLedgerQueryError("query client is already open")
        self._unsubscribe = await self._bus.subscribe(
            self.contract.completed_topic,
            on_message=self._terminal,
            group_id=self.group_id,
            auto_offset_reset="latest",
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self._unsubscribe is not None:
                await self._unsubscribe()
        finally:
            self._unsubscribe = None
            for future in self._pending.values():
                if not future.done():
                    future.cancel()
            self._pending.clear()

    async def _terminal(self, message: ModelEventMessage) -> None:
        try:
            raw = json.loads(message.value)
            if not isinstance(raw, dict):
                return
            correlation = UUID(str(raw.get("correlation_id")))
        except (ValueError, TypeError, UnicodeDecodeError):
            return
        future = self._pending.get(correlation)
        if future is None or future.done():
            return
        try:
            answer = self.contract.result_model.model_validate(raw.get("payload"))
            if getattr(answer, "correlation_id", None) != correlation:
                raise WorkLedgerQueryError(
                    "reply payload correlation_id differs from envelope"
                )
            if error := getattr(answer, "error", None):
                raise WorkLedgerQueryError(str(error))
        except (ValidationError, WorkLedgerQueryError) as exc:
            future.set_exception(WorkLedgerQueryError(str(exc)))
        else:
            future.set_result(answer)

    async def query(
        self, payload: Mapping[str, object], *, timeout_s: float = 60.0
    ) -> BaseModel:
        if self._unsubscribe is None:
            raise WorkLedgerQueryError(
                "open the client with async with before querying"
            )
        request = self.contract.request_model.model_validate(
            {"correlation_id": uuid4(), **payload}
        )
        correlation = UUID(str(request.model_dump(mode="json")["correlation_id"]))
        if correlation in self._pending:
            raise WorkLedgerQueryError("correlation_id already has a pending query")
        envelope = ModelEventEnvelope[dict[str, object]](
            payload=request.model_dump(mode="json"),
            correlation_id=correlation,
            # This operator work ledger is not tenant-scoped.
            tenant_id=None,
            event_type=".".join(self.contract.command_topic.split(".")[2:4]),
        )
        future: asyncio.Future[BaseModel] = asyncio.get_running_loop().create_future()
        self._pending[correlation] = future
        try:
            async with asyncio.timeout(timeout_s):
                await self._bus.publish(
                    self.contract.command_topic,
                    str(correlation).encode(),
                    envelope.model_dump_json().encode(),
                )
                return await future
        finally:
            self._pending.pop(correlation, None)


def load_work_ledger_query_contract() -> ModelWorkLedgerQueryContract:
    """Refuse a missing or malformed installed node contract before connecting."""
    contract = node_contract("node_work_ledger_query_effect")
    try:
        dispatch = contract["runtime_dispatch"]
        if not isinstance(dispatch, dict):
            raise ValueError("runtime_dispatch must be a mapping")
        terminals = dispatch["terminal_events"]
        if not isinstance(terminals, dict):
            raise ValueError("terminal_events must be a mapping")

        def model(field: str) -> type[BaseModel]:
            declared = contract[field]
            if isinstance(declared, str):
                module, _, name = declared.rpartition(".")
            elif isinstance(declared, dict):
                module, name = declared["module"], declared["class"]
            else:
                raise ValueError(f"{field} must declare a model")
            result = getattr(importlib.import_module(module), name)
            if not isinstance(result, type) or not issubclass(result, BaseModel):
                raise ValueError(f"{field} must be a pydantic model")
            return result

        return ModelWorkLedgerQueryContract(
            command_topic=dispatch["command_topic"],
            completed_topic=terminals["success"],
            request_model=model("input_model"),
            result_model=model("output_model"),
        )
    except (KeyError, TypeError, ValueError, ImportError, AttributeError) as exc:
        raise RegistryUnresolvedError(
            f"work-ledger query contract is unreadable: {exc}"
        ) from exc


@contextlib.asynccontextmanager
async def open_work_ledger_query_client(
    *, lane: str, workspace_root: Path | None
) -> AsyncIterator[HandlerWorkLedgerQuery]:
    """Use the caller's selected dev credential; refuse other lanes before I/O."""
    if lane != "dev":
        raise WorkLedgerQueryError("work-ledger query is enabled only for the dev lane")
    selection = resolve_lane_selection_at(lane, workspace_root)
    if not selection.security_protocol.upper().startswith("SASL_"):
        raise WorkLedgerQueryError(
            "work-ledger reads require an authenticated dev lane"
        )
    transport = resolve_lane_client_transport_for(
        lane_target=selection, onex_home=Path.home() / ".onex", environ=os.environ
    )
    binding = (
        bind_lane_client_transport(transport)
        if transport is not None
        else contextlib.nullcontext()
    )
    with binding:
        bus = EventBusKafka.from_bootstrap(selection.bootstrap_servers)
        try:
            client = HandlerWorkLedgerQuery(bus)
            await bus.start()
            async with client:
                yield client
        finally:
            await bus.close()
