# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ledger rows through the reader node, and one row through the bus append command (OMN-26507).

The rows read is the ``rows`` query of ``node_work_ledger_reader_effect`` for the kinds a caller
names; a reader that cannot answer is a refusal, never an empty answer and never a read of the
ledger file. The append is the bus append command with the row on stdin (the route the deploy job
reading takes), never ``onex-ledger <ledger file> --append``.
"""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

from omnibase_infra.handlers.handler_ledger_open_claims import LedgerReadRefusedError
from omnibase_infra.handlers.handler_ledger_rows import (
    HandlerLedgerRows,
    bus_row_appender,
    rows_from_environment,
)
from omnibase_infra.models.lab_proof.model_proof_ledger import ModelProofLedger
from omnibase_infra.models.model_ledger_rows_request import ModelLedgerRowsRequest
from tests.scripts.ledger_reader_stub import LedgerReaderStub

pytestmark = pytest.mark.unit

ROW = "2026-11-23T05:00:00Z | HOLD | lane=peer | id=h1 | surface=s | until=2026-11-23T08:00:00Z | x"


def test_ledger_bus_handler_returns_the_rows_of_the_named_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = LedgerReaderStub(tmp_path / "reader")
    for name, value in stub.env().items():
        monkeypatch.setenv(name, value)
    (stub.directory / "rows.txt").write_text(f"{ROW}\n", encoding="utf-8")
    result = HandlerLedgerRows().handle(
        ModelLedgerRowsRequest(
            reader_command=f"{stub.python} -m omnimarket.nodes.node_work_ledger_reader_effect",
            reader="prepr-runtime-pool",
            window_hours=720,
            kinds=("HOLD", "RELEASE"),
        )
    )
    assert result.rows == (ROW,)
    (call,) = stub.calls()
    assert "--kind HOLD --kind RELEASE" in call


def test_ledger_bus_handler_refuses_when_the_reader_does_not_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = LedgerReaderStub(tmp_path / "reader")
    for name, value in stub.env().items():
        monkeypatch.setenv(name, value)
    stub.refuse_with(3)
    request = ModelLedgerRowsRequest(
        reader_command=f"{stub.python} -m omnimarket.nodes.node_work_ledger_reader_effect",
        reader="not-moved",
        window_hours=1,
        kinds=("HOLD",),
    )
    with pytest.raises(LedgerReadRefusedError, match="exit 3"):
        HandlerLedgerRows().handle(request)


def test_ledger_bus_rows_from_an_environment_without_a_reader_command_is_empty() -> (
    None
):
    assert rows_from_environment({}, "r", 1, ("HOLD",)) == []


def _append_script(tmp_path: Path) -> tuple[Path, Path]:
    sink = tmp_path / "seen.json"
    script = tmp_path / "append.py"
    script.write_text(
        "import json, sys\n"
        f"json.dump({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}, open({str(sink)!r}, 'w'))\n"
        "print('{\"ok\": true}')\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return script, sink


def test_ledger_bus_append_sends_the_row_on_stdin_as_one_row(tmp_path: Path) -> None:
    script, sink = _append_script(tmp_path)
    environ = {
        "ONEX_LEDGER_BUS_APPEND_COMMAND": f"{sys.executable} {script} append",
        "ONEX_LEDGER_HOST_NAME": "h-test",
    }
    append = bus_row_appender(environ)
    append("2026-11-23T05:00:00Z | STATUS | lane=x | a row; with 'quotes' and $vars")
    seen = json.loads(sink.read_text(encoding="utf-8"))
    argv = seen["argv"]
    assert argv[:1] == ["append"]
    assert argv[argv.index("--lane") + 1] == "lab-proof-alarm"
    assert argv[argv.index("--host") + 1] == "h-test"
    assert argv[argv.index("--rows-file") + 1] == "-"
    assert "--request-id" in argv
    assert seen["stdin"].startswith("2026-11-23T05:00:00Z | STATUS | lane=x | a row;")
    assert "with 'quotes' and $vars | req=" in seen["stdin"]
    assert seen["stdin"].count("\n") == 1


def test_ledger_bus_append_without_a_command_is_the_refusing_append() -> None:
    assert bus_row_appender({}) is ModelProofLedger.no_append


def test_ledger_bus_append_without_a_host_name_says_so(tmp_path: Path) -> None:
    script, _ = _append_script(tmp_path)
    append = bus_row_appender(
        {"ONEX_LEDGER_BUS_APPEND_COMMAND": f"{sys.executable} {script}"}
    )
    with pytest.raises(RuntimeError, match="ONEX_LEDGER_HOST_NAME"):
        append("2026-11-23T05:00:00Z | STATUS | lane=x | row")
