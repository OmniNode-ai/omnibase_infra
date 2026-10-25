# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex ledger`` -- read the cloud ledger from the operator's own machine (OMN-17205).

    onex ledger read --correlation-id <id>

That single line is the terminal read in ``beta/GOAL.md`` row 0b. Before this
command existed the row's own probe could not be run by the person who set it:
the staging Kubernetes API is unreachable from the operator Mac, and the
deployed onex-api served no projection route, so every attempt came back 401 --
the same answer it gives for a path it does not serve. A goal row whose probe
cannot be executed is indistinguishable from a goal row with no probe: neither
can fail, so neither can catch a drop.

CONTRACT OF THIS COMMAND
    stdout is exactly one JSON document and nothing else, so the probe composes
    into a shell pipeline and a goal row can cite the process's exit status.
    Every diagnostic goes to stderr. The exit code is the verdict:

        0  found              a row carries this correlation id
        1  not_found          the projection is there and holds no such row
        2  projection_absent  the projection does not exist on that plane
        3  unauthenticated    no credential, or the plane refused it
        4  unavailable        the plane could not answer, or was not reached

    "No row" is a NON-ZERO exit on purpose. A probe that succeeds when the
    ledger is empty is a probe that passes while the pipeline is dead.

SECRETS
    There is no ``--client-secret``, no ``--token`` and no ``--api-key`` option,
    and a test asserts their absence: a secret passed on the command line lands
    in the process table, the shell history and every exec log. The credential
    is resolved by reference from ``~/.onex`` (mode 0600), which is also where
    the base URL comes from -- there is no host literal and no environment
    variable on this path.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn, TextIO

import click
from pydantic import ValidationError

from omnibase_core.errors.model_onex_error import ModelOnexError
from omnibase_infra.cli.contract_registry import RegistryUnresolvedError
from omnibase_infra.cli.delegate_lane import DelegateLaneSelectionError
from omnibase_infra.cli.delegate_lane_credentials import DelegateLaneCredentialError
from omnibase_infra.enums.enum_cloud_ledger_verdict import EnumCloudLedgerVerdict
from omnibase_infra.gateway.client.cloud_ledger_reader import CloudLedgerReader
from omnibase_infra.gateway.client.gateway_transport_httpx import (
    GatewayTransportHttpx,
)
from omnibase_infra.gateway.client.store_gateway_credential import (
    StoreGatewayCredential,
)
from omnibase_infra.gateway.models.model_cloud_ledger_read import ModelCloudLedgerRead
from omnibase_infra.handlers.handler_work_ledger_query import (
    WorkLedgerQueryError,
    open_work_ledger_query_client,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

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
    """Read cloud hook projections and the dev database work ledger."""


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


def _bus_options[Command: Callable[..., object]](command: Command) -> Command:
    # cli-own-vocabulary: this infrastructure client is intentionally dev-only.
    command = click.option("--lane", required=True, type=click.Choice(["dev"]))(command)
    command = click.option(
        "--omnibase-path",
        envvar="OMNIBASE_PATH",
        type=click.Path(file_okay=False, path_type=Path),
    )(command)
    return click.option(
        "--timeout",
        "timeout_s",
        type=click.FloatRange(min=0, min_open=True),
        default=60.0,
        show_default=True,
    )(command)


def _run(coroutine: Coroutine[None, None, dict[str, object]]) -> None:
    try:
        result = asyncio.run(coroutine)
    except (
        RegistryUnresolvedError,
        DelegateLaneSelectionError,
        DelegateLaneCredentialError,
        WorkLedgerQueryError,
        ValidationError,
        TimeoutError,
        ModelOnexError,
    ) as exc:
        raise click.ClickException(sanitize_error_message(exc)) from exc
    click.echo(json.dumps(result, indent=2))


@ledger_group.command("query")
@_bus_options
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.File("r"),
    help="The read node's typed request JSON; use - for stdin. correlation_id is optional.",
)
def work_ledger_query_command(
    lane: str, omnibase_path: Path | None, timeout_s: float, input_file: TextIO
) -> None:
    """Query work_ledger_rows for ledger-ask; return rows, freshness and parity."""
    try:
        payload = json.load(input_file)
    except ValueError as exc:
        raise click.BadParameter(
            "input must be a JSON object", param_hint="--input"
        ) from exc
    if not isinstance(payload, dict):
        raise click.BadParameter("input must be a JSON object", param_hint="--input")

    async def query() -> dict[str, object]:
        async with open_work_ledger_query_client(
            lane=lane, workspace_root=omnibase_path
        ) as client:
            return (await client.query(payload, timeout_s=timeout_s)).model_dump(
                mode="json"
            )

    _run(query())


@ledger_group.command("catchup")
@_bus_options
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%dT%H:%M:%SZ"]),
    help="UTC window start; defaults to 12 hours before now.",
)
@click.option("--to", "recipient", help="Lane whose unanswered inbox is included.")
def work_ledger_catchup_command(
    lane: str,
    omnibase_path: Path | None,
    timeout_s: float,
    since: datetime | None,
    recipient: str | None,
) -> None:
    """Read a bounded catchup from the database, retaining each answer's metadata."""
    now = datetime.now(UTC)
    start = (
        since.replace(tzinfo=UTC) if since is not None else now - timedelta(hours=12)
    )

    async def catchup() -> dict[str, object]:
        async with open_work_ledger_query_client(
            lane=lane, workspace_root=omnibase_path
        ) as client:
            sections: dict[str, object] = {}
            for kind in ("open-claims", "holds", "rulings", "rows", "inbox"):
                if kind == "inbox" and recipient is None:
                    continue
                request: dict[str, object] = {
                    "query": kind,
                    "since": start,
                    "until": now,
                    "now": now,
                }
                if kind == "rows":
                    request["kinds"] = ("FRICTION", "CORRECTION", "STATUS")
                if kind == "inbox":
                    request["lane"] = recipient
                sections[kind] = (
                    await client.query(request, timeout_s=timeout_s)
                ).model_dump(mode="json")
            return sections

    _run(catchup())
