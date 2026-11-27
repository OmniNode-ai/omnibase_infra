# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two calls a host-side caller makes to the work ledger over the bus (OMN-20547, OMN-26507).

Not a handler: the handlers that read the ledger rows, read the open claims and append a row all
call through here, so the reader command line and the append command line are built in one place.

READ. ``call_reader`` runs the ledger reader node
(``python -m omnimarket.nodes.node_work_ledger_reader_effect``) for one named reader and query. A
reader that exits 3 (not in the overlays' ``moved_readers``), 1 (stale), 69 (unreadable) or cannot
be run is a refusal (:class:`LedgerReadRefusedError`): nothing is read from the ledger file in its
place.

WRITE. ``append_row_over_bus`` runs the bus append command (omnimarket's ``work-ledger append``)
with the row on stdin and ``--lane``, ``--host``, ``--request-id`` and ``--rows-file -`` on its
command line. ``onex-ledger`` is not used for it: that tool takes the ledger file path as its target
even in bus mode.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Mapping

from omnibase_infra.models.operational_timeouts import TIMEOUTS


class LedgerReadRefusedError(OSError):
    """The ledger was not read; the message names why (an OSError, so a caller that
    treats an unreadable ledger file as a refusal treats this the same way)."""


def call_reader(
    reader_command: str,
    reader: str,
    query: str,
    window_hours: int,
    extra: tuple[str, ...] = (),
) -> list[str]:
    """One reader call; the lines it printed, or a refusal."""
    argv = [
        *shlex.split(reader_command),
        "--reader",
        reader,
        "--query",
        query,
        "--window-hours",
        str(window_hours),
        *extra,
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=TIMEOUTS.ledger_reader_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LedgerReadRefusedError(f"ledger reader {query} not run: {exc}") from exc
    if proc.returncode != 0:
        summary = (proc.stderr or "").strip().splitlines()[-1:] or ["no output"]
        raise LedgerReadRefusedError(
            f"ledger reader exit {proc.returncode} ({query}): {summary[0]}"
        )
    return (proc.stdout or "").splitlines()


def append_row_over_bus(
    command: str,
    *,
    lane: str,
    host: str,
    row: str,
    request_id: str,
    env: Mapping[str, str],
    timeout: float | None,
) -> str:
    """Append ``row`` through the bus append command; ``""`` when it took it, else why not."""
    argv = [
        *shlex.split(command),
        "--lane",
        lane,
        "--host",
        host,
        "--request-id",
        request_id,
        "--rows-file",
        "-",
    ]
    try:
        done = subprocess.run(
            argv,
            input=f"{row} | req={request_id}\n",
            capture_output=True,
            text=True,
            env=dict(env),
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"append-command-failed:{type(exc).__name__}"
    if done.returncode != 0:
        tail = done.stderr.strip()[-300:].replace("\n", " | ")
        return f"append-command-exit-{done.returncode}: {tail}".rstrip(": ")
    return ""


__all__ = ["LedgerReadRefusedError", "append_row_over_bus", "call_reader"]
