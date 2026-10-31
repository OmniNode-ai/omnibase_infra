# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex ledger``: cloud hook receipts and dev database work-ledger reads.

``read --correlation-id`` queries the cloud hook projection. It prints one
JSON document and exits 0 for found, 1 for not found, 2 for absent projection,
3 for unauthenticated and 4 for unavailable. Its HTTP credential and base URL
come by reference from the machine's ``~/.onex`` store.

``query --bus-lane dev`` calls the database work-ledger read node over Kafka.
``query --stdin`` implements ``ONEX_LEDGER_BUS_READ_COMMAND`` for ledger-ask
and catchup: one typed query in, one correlated answer out. The machine's
issued dev credential and the lane's declared transport are resolved by the
query command. Every failed read is nonzero, with no answer on stdout.

Neither command accepts a credential value on its command line. See
the root README for work-ledger ACLs and reader wiring.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import click
from pydantic import JsonValue, TypeAdapter, ValidationError

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.delegate_lane_credentials import resolve_lane_binding
from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.gateway.client.cloud_ledger_reader import CloudLedgerReader
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_cloud_ledger_read import ModelCloudLedgerRead
from omnibase_infra.handlers.container.container_ledger_query import bind_ledger_query
from omnibase_infra.handlers.handler_work_ledger_query import (
    load_work_ledger_query_client_config,
)
from omnibase_infra.models.model_work_ledger_query_cli_options import (
    ModelWorkLedgerQueryCliOptions,
)

__all__ = ["ledger_group"]

_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_MAX_STDIN_CHARS = 1_048_576

#: Exit status for "there is no credential to present". Deliberately the same
#: code as a credential the plane refused: from the caller's side both mean
#: "this read was never authenticated", and splitting them would ask a shell
#: to care about a distinction the JSON already carries in ``detail``.
_UNAUTHENTICATED_EXIT = 3


def _emit(result: ModelCloudLedgerRead) -> NoReturn:
    """Print the one JSON document and exit with the verdict's status."""
    document = result.model_dump(mode="json")
    document["exit_code"] = result.exit_code
    click.echo(json.dumps(document, indent=2))
    sys.exit(result.exit_code)


@click.group("ledger")
def ledger_group() -> None:  # stub-ok
    """Read the cloud hook ledger and the dev work ledger."""


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
    client = bind_ledger_query(resolve_work_ledger_query_transport(bus_lane))
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
    "--bus-lane",
    type=click.Choice((load_work_ledger_query_client_config().lane,)),
    default="dev",
    show_default=True,
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
def ledger_query(context: click.Context, /, **options: object) -> None:
    """Query work_ledger_rows through the lab read node; print its full JSON answer."""
    timeout_s = options["timeout_s"]
    if not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s):
        raise click.UsageError("--timeout must be finite and positive")
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
            stdin_text = click.get_text_stream("stdin").read(_MAX_STDIN_CHARS + 1)
            if len(stdin_text) > _MAX_STDIN_CHARS:
                raise click.UsageError(
                    "stdin query JSON object exceeds 1048576 characters"
                )
            stdin_request = _JSON_OBJECT.validate_json(stdin_text)
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


ledger_group.add_command(ledger_query)


@ledger_group.command("read")
@click.option(
    "--correlation-id",
    required=True,
    help="Correlation id stamped on the emitted hook event.",
)
@click.option(
    "--limit",
    default=10,
    show_default=True,
    type=click.IntRange(1, 100),
    help="Maximum rows to request.",
)
@click.option(
    "--include-payload",
    is_flag=True,
    default=False,
    help=(
        "Ask for the verbatim event bodies. Off by default -- a hook payload is "
        "raw captured text and the default read must be safe to paste."
    ),
)
def ledger_read(correlation_id: str, limit: int, include_payload: bool) -> None:
    """Read cloud hook-ledger rows for one correlation id.

    Prints one JSON document on stdout and exits 0 only when a row came back.
    """
    store = StoreGatewayCredential(onex_home=Path.home() / ".onex")
    try:
        credential = store.load_read_credential()
    except ModelOnexError as exc:
        # Fail-closed and NAMED: the remediation is the command that fixes it,
        # not a stack trace. An unreadable credential must never degrade into
        # an anonymous call the operator believes was authenticated.
        click.echo(f"Error: {exc}", err=True)
        sys.exit(_UNAUTHENTICATED_EXIT)

    reader = CloudLedgerReader(
        transport=GatewayTransportHttpx(),
        credential=credential,
    )
    result = asyncio.run(
        reader.read(
            correlation_id=correlation_id,
            now=datetime.now(UTC),
            limit=limit,
            include_payload=include_payload,
        )
    )
    if result.verdict is not EnumCloudLedgerVerdict.FOUND:
        click.echo(f"{result.verdict.value}: {result.detail}", err=True)
    _emit(result)
