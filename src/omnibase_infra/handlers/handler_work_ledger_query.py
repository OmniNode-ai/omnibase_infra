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
import json
from datetime import UTC
from importlib.resources import files
from uuid import UUID, uuid4

import click
import yaml
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import JsonValue, TypeAdapter, ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.cli.delegate_lane_credentials import resolve_lane_binding
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumInfraTransportType,
)
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.event_bus.kafka_auth import build_aiokafka_auth_kwargs
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models.model_work_ledger_query_cli_options import (
    ModelWorkLedgerQueryCliOptions,
)
from omnibase_infra.models.model_work_ledger_query_client_config import (
    ModelWorkLedgerQueryClientConfig,
)
from omnibase_infra.topics.topic_namespace import apply_topic_namespace

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])


class WorkLedgerQueryError(RuntimeHostError):
    """The node's answer is an error or violates the read contract."""

    def __init__(self, message: str, *, correlation_id: UUID | None = None) -> None:
        super().__init__(
            message,
            context=ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="work_ledger_query",
                correlation_id=correlation_id,
            ),
        )


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
    """One bounded request per connection, with a private reply consumer group.

    The CLI resolves the dev transport and credential before construction.
    Kafka's topic/group authorization errors propagate; they never produce an
    empty ledger answer. No offsets are committed and no topics are created.
    """

    def __init__(self, transport: ModelKafkaEventBusConfig) -> None:
        self._transport = transport
        self._contract = load_work_ledger_query_client_config()

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
        correlation_id = UUID(supplied_id) if supplied_id is not None else uuid4()
        kind = request.get("query")
        if kind not in self._contract.queries:
            raise WorkLedgerQueryError(
                "Unsupported work-ledger query", correlation_id=correlation_id
            )
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        payload = {**request, "correlation_id": str(correlation_id)}
        command = apply_topic_namespace(self._contract.command_topic)
        completed = apply_topic_namespace(self._contract.completed_topic)
        parts = self._contract.command_topic.split(".")
        envelope = ModelEventEnvelope[dict[str, JsonValue]](
            payload=payload,
            correlation_id=correlation_id,
            event_type=f"{parts[2]}.{parts[3]}",
            tenant_id=None,
        )
        auth = build_aiokafka_auth_kwargs(self._transport)
        consumer = AIOKafkaConsumer(
            completed,
            bootstrap_servers=self._transport.bootstrap_servers,
            group_id=self._contract.consumer_group_prefix + uuid4().hex,
            enable_auto_commit=False,
            auto_offset_reset="latest",
            **auth,
        )
        producer = AIOKafkaProducer(
            bootstrap_servers=self._transport.bootstrap_servers, acks="all", **auth
        )
        try:
            async with asyncio.timeout(timeout_s):
                await consumer.start()
                # start() joins the group. position() resolves the initial
                # latest offset BEFORE send, so a fast reply cannot be skipped.
                for partition in consumer.assignment():
                    await consumer.position(partition)
                await producer.start()
                await producer.send_and_wait(
                    command,
                    envelope.model_dump_json().encode("utf-8"),
                    key=str(correlation_id).encode("utf-8"),
                )
                while True:
                    message = await consumer.getone()
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
                            "Ledger reply query mismatch", correlation_id=correlation_id
                        )
                    if answer.get("error"):
                        # The domain error can contain database driver details.
                        # Keep those off stderr; callers get a named refusal.
                        raise WorkLedgerQueryError(
                            "Ledger node reported a database error",
                            correlation_id=correlation_id,
                        )
                    if not isinstance(answer.get("freshness"), dict) or not isinstance(
                        answer.get("parity"), dict
                    ):
                        raise WorkLedgerQueryError(
                            "Ledger reply lacks freshness or parity",
                            correlation_id=correlation_id,
                        )
                    return answer
        finally:
            # Both may have partially started, even if startup raised.
            try:
                await producer.stop()
            finally:
                await consumer.stop()


def resolve_work_ledger_query_transport(lane: str) -> ModelKafkaEventBusConfig:
    """Select dev's declared address/protocol and the machine's stored identity."""
    if lane != "dev":
        raise ValueError("Work-ledger queries are enabled for the dev lane only")
    selection, transport = resolve_lane_binding(lane)
    config = ModelKafkaEventBusConfig.default()
    overrides = {
        "bootstrap_servers": selection.bootstrap_servers,
        "security_protocol": selection.security_protocol,
        "sasl_mechanism": selection.sasl_mechanism,
    }
    if transport is not None:
        overrides.update(transport.as_client_config_overrides())
    return config.model_copy(update=overrides)


async def run_work_ledger_read(
    request: dict[str, JsonValue], *, bus_lane: str, timeout_s: float
) -> dict[str, JsonValue]:
    client = HandlerWorkLedgerQuery(resolve_work_ledger_query_transport(bus_lane))
    return await client.query(request, timeout_s=timeout_s)


def _read(request: dict[str, JsonValue], bus_lane: str, timeout_s: float) -> None:
    try:
        answer = asyncio.run(
            run_work_ledger_read(request, bus_lane=bus_lane, timeout_s=timeout_s)
        )
    except Exception as exc:  # boundary: broker and driver errors must never look empty
        raise click.ClickException(
            f"Work-ledger read refused ({type(exc).__name__})"
        ) from exc
    click.echo(json.dumps(answer))


@click.command("query")
@click.option(
    "--bus-lane", type=click.Choice(["dev"]), default="dev", show_default=True
)
@click.option(
    "--query",
    "kind",
    type=click.Choice(load_work_ledger_query_client_config().queries),
    default="rows",
)
@click.option("--since", type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%SZ"]))
@click.option("--until", type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%SZ"]))
@click.option("--lane", help="Filter rows to a ledger lane.")
@click.option("--ticket")
@click.option("--pr")
@click.option("--term", help="Ledger read node's lexical term or expression.")
@click.option("--kind", "kinds", multiple=True, help="Row kind; may be repeated.")
@click.option("--limit", type=click.IntRange(min=1))
@click.option(
    "--stdin",
    "read_stdin",
    is_flag=True,
    help="Read one typed query JSON object from stdin (ONEX_LEDGER_BUS_READ_COMMAND).",
)
@click.option(
    "--timeout",
    "timeout_s",
    type=click.FloatRange(min=0, min_open=True),
    default=60.0,
    envvar="ONEX_LEDGER_BUS_TIMEOUT_S",
)
@click.pass_context
def ledger_query(context: click.Context, **options: object) -> None:
    """Query work_ledger_rows through the lab read node; print its full JSON answer."""
    args = ModelWorkLedgerQueryCliOptions.model_validate(options)
    if args.read_stdin:
        # The existing ledger-ask/catchup source adapter owns the request and
        # verifies its correlation and row count. Never rewrite its filters.
        filter_options = (
            "kind",
            "since",
            "until",
            "lane",
            "ticket",
            "pr",
            "term",
            "kinds",
            "limit",
        )
        if any(
            context.get_parameter_source(name) is click.core.ParameterSource.COMMANDLINE
            for name in filter_options
        ):
            raise click.UsageError(
                "--stdin cannot be combined with query filter options"
            )
        try:
            stdin_request = TypeAdapter(dict[str, JsonValue]).validate_json(
                click.get_text_stream("stdin").read()
            )
        except ValidationError as exc:
            raise click.UsageError("stdin must contain one query JSON object") from exc
        _read(stdin_request, args.bus_lane, args.timeout_s)
        return
    request: dict[str, JsonValue] = {"query": args.kind}
    for name, value in (
        ("lane", args.lane),
        ("ticket", args.ticket),
        ("pr", args.pr),
        ("term", args.term),
        ("limit", args.limit),
    ):
        if value is not None:
            request[name] = value
    for name, stamp in (("since", args.since), ("until", args.until)):
        if stamp is not None:
            request[name] = stamp.replace(tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.kinds:
        request["kinds"] = list(args.kinds)
    if args.kind == "inbox":
        if not args.lane:
            raise click.UsageError("--query inbox requires --lane")
        request.pop("lane", None)
        request["lanes"] = [args.lane]
    _read(request, args.bus_lane, args.timeout_s)
