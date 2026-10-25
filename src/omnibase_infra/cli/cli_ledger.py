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
work-ledger handler. Every failed read is nonzero, with no answer on stdout.

Neither command accepts a credential value on its command line. See
``docs/operations/work_ledger_query.md`` for work-ledger ACLs and reader wiring.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

import click

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict
from omnibase_infra.gateway.client.cloud_ledger_reader import CloudLedgerReader
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_cloud_ledger_read import ModelCloudLedgerRead
from omnibase_infra.handlers.handler_work_ledger_query import ledger_query

__all__ = ["ledger_group"]

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
