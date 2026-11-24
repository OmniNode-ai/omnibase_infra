# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A row appended over the bus command is read back through the reader node route (OMN-26507).

Two real subprocess boundaries stand in for the bus: the append command writes the row it
receives on stdin to a stand-in bus log, and the reader command answers the ``rows`` query from
that log. No ledger file exists and none is named to either command, so a read or a write of a
ledger file would fail the test.
"""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_ledger_rows import (
    HandlerLedgerRows,
    bus_row_appender,
)
from omnibase_infra.models.model_ledger_rows_request import ModelLedgerRowsRequest

pytestmark = pytest.mark.integration

_APPEND = """\
import sys
open(sys.argv[1], "a", encoding="utf-8").write(sys.stdin.read())
"""

_READER = """\
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    print(line.rstrip("\\n"))
print("READER reader=bus query=rows status=ok", file=sys.stderr)
"""


def _script(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_row_appended_over_the_bus_command_is_read_through_the_reader_without_the_file(
    tmp_path: Path,
) -> None:
    bus_log = tmp_path / "bus.log"
    bus_log.write_text("", encoding="utf-8")
    ledger_file = tmp_path / "ledger.md"
    append_script = _script(tmp_path, "append.py", _APPEND)
    reader_script = _script(tmp_path, "reader.py", _READER)

    append = bus_row_appender(
        {
            "ONEX_LEDGER_BUS_APPEND_COMMAND": f"{sys.executable} {append_script} {bus_log}",
            "ONEX_LEDGER_HOST_NAME": "h-test",
        }
    )
    row = "2026-11-24T01:00:00Z | HOLD | lane=peer | id=h1 | surface=s | until=2026-11-24T04:00:00Z | x"
    append(row)

    result = HandlerLedgerRows().handle(
        ModelLedgerRowsRequest(
            reader_command=f"{sys.executable} {reader_script} {bus_log}",
            reader="prepr-runtime-pool",
            window_hours=24,
            kinds=("HOLD",),
        )
    )

    assert len(result.rows) == 1
    assert result.rows[0].startswith(row)
    assert not ledger_file.exists()
