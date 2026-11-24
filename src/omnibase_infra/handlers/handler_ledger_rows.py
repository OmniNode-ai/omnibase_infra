# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ledger rows from the reader node and one row to the bus append command (OMN-26507).

WHY. A ledger on the event bus with one writer and many readers of a file copy is two ledgers
(operator ruling 2026-11-05). ``scripts/runtime_build/prepr_runtime_pool.py`` read the rolling
ledger file for the HOLD rows that lease a lab host surface and the RELEASE rows that decide the
INCONCLUSIVE streak alarm, and wrote the alarm row through ``onex-ledger <ledger file> --append``.
It now asks ``omnimarket.nodes.node_work_ledger_reader_effect`` and appends over the bus.

READ. ``read_rows`` is the reader's ``rows`` query for the kinds a caller names, through the same
``call_reader`` the open-claims read uses. A reader that exits 3 (not in the overlays'
``moved_readers``), 1 (stale), 69 (unreadable) or cannot be run is a refusal
(``LedgerReadRefusedError``): nothing is read from the file in its place.

WRITE. ``bus_row_appender`` binds the bus append command named by ``ONEX_LEDGER_BUS_APPEND_COMMAND``:
the row on stdin with ``--lane``, ``--host``, ``--request-id`` and ``--rows-file -``, the route of
``handler_deploy_job_reading.append_ledger_row``. ``onex-ledger`` is not used: it takes the ledger
file path (``ONEX_LEDGER_PATH``) as its target even in bus mode.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.ledger_bus_calls import (
    LedgerReadRefusedError,
    append_row_over_bus,
    call_reader,
)
from omnibase_infra.models.lab_proof.model_proof_ledger import (
    ALARM_LANE,
    ModelProofLedger,
)
from omnibase_infra.models.model_ledger_rows_request import ModelLedgerRowsRequest
from omnibase_infra.models.model_ledger_rows_result import ModelLedgerRowsResult
from omnibase_infra.models.operational_timeouts import TIMEOUTS

READER_COMMAND_ENV = "ONEX_LEDGER_READER_COMMAND"
APPEND_COMMAND_ENV = "ONEX_LEDGER_BUS_APPEND_COMMAND"
ATTRIBUTION_ENV = "ONEX_LEDGER_HOST_NAME"


def read_rows(request: ModelLedgerRowsRequest) -> ModelLedgerRowsResult:
    """Every row of the request's kinds in its window, as the reader node answered."""
    extra = tuple(arg for kind in request.kinds for arg in ("--kind", kind))
    lines = call_reader(
        request.reader_command, request.reader, "rows", request.window_hours, extra
    )
    return ModelLedgerRowsResult(rows=tuple(lines))


class HandlerLedgerRows:
    """Fail closed: a read that is not whole is a refusal, never a partial answer."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def handle(self, request: ModelLedgerRowsRequest) -> ModelLedgerRowsResult:
        """Read the rows of the request's window, or raise LedgerReadRefusedError."""
        return read_rows(request)


def rows_from_environment(
    environ: Mapping[str, str], reader: str, window_hours: int, kinds: tuple[str, ...]
) -> list[str]:
    """The rows of the kinds, read through the command ``ONEX_LEDGER_READER_COMMAND`` names.

    No command configured reads no row (a probe on a host with no ledger reader, as a run with no
    ledger file path did before); a command that cannot answer raises LedgerReadRefusedError.
    """
    command = environ.get(READER_COMMAND_ENV, "").strip()
    if not command:
        return []
    request = ModelLedgerRowsRequest(
        reader_command=command, reader=reader, window_hours=window_hours, kinds=kinds
    )
    return list(read_rows(request).rows)


def bus_row_appender(
    environ: Mapping[str, str], lane: str = ALARM_LANE
) -> Callable[[str], None]:
    """Bind the bus append command to one row at a time; ``ModelProofLedger.no_append`` when unset.

    The row is the command's stdin, never a shell string. A refusal raises RuntimeError, so the
    proof text says the row was not written.
    """
    if not environ.get(APPEND_COMMAND_ENV, "").strip():
        return ModelProofLedger.no_append

    def append(row: str) -> None:
        command = environ[APPEND_COMMAND_ENV].strip()
        host = environ.get(ATTRIBUTION_ENV, "").strip()
        if not host:
            raise RuntimeError(
                f"{APPEND_COMMAND_ENV} and {ATTRIBUTION_ENV} must both be set"
            )
        why = append_row_over_bus(
            command,
            lane=lane,
            host=host,
            row=row,
            request_id=str(uuid.uuid4()),
            env=environ,
            timeout=TIMEOUTS.dev_lane_ledger_seconds,
        )
        if why:
            raise RuntimeError(why)

    return append


__all__ = [
    "HandlerLedgerRows",
    "LedgerReadRefusedError",
    "bus_row_appender",
    "read_rows",
    "rows_from_environment",
]
