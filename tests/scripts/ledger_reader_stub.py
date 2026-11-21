# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A stand-in interpreter that answers the ledger reader node from files (OMN-20547).

Scripts that read the rolling ledger reach it through
``<interp> -m omnimarket.nodes.node_work_ledger_reader_effect``, where ``<interp>`` is
the first word of ``ONEX_LEDGER_BUS_APPEND_COMMAND``. This module writes an
executable named ``ledger-python`` that answers exactly that invocation from
canned files, logs its argv, and runs the real interpreter for anything else, so
the scripts under test are the real ones and only the bus read is replaced.
"""

from __future__ import annotations

import sys
from pathlib import Path

_STUB = """#!/usr/bin/env bash
if [ "${1:-}" = "-m" ] && [ "${2:-}" = "omnimarket.nodes.node_work_ledger_reader_effect" ]; then
  echo "$*" >> "$STUB_DIR/reader.log"
  query=""
  prev=""
  for arg in "$@"; do
    [ "$prev" = "--query" ] && query="$arg"
    prev="$arg"
  done
  if [ -f "$STUB_DIR/exit" ]; then
    echo "READER reader=stub query=$query status=refused" >&2
    exit "$(cat "$STUB_DIR/exit")"
  fi
  echo "READER reader=stub query=$query status=ok" >&2
  case "$query" in
    rows) cat "$STUB_DIR/rows.txt" ;;
    open-claims) cat "$STUB_DIR/open.txt" ;;
  esac
  exit 0
fi
exec "$STUB_REAL_PYTHON" "$@"
"""


class LedgerReaderStub:
    """One stub interpreter and the claims it reports."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.python = directory / "ledger-python"
        self.python.write_text(_STUB, encoding="utf-8")
        self.python.chmod(0o755)
        self._claims: list[tuple[str, str, str, str, bool]] = []
        self.set_claims()

    def env(self) -> dict[str, str]:
        """The environment a script needs to reach this stub."""
        return {
            "STUB_DIR": str(self.directory),
            "STUB_REAL_PYTHON": sys.executable,
            "ONEX_LEDGER_BUS_APPEND_COMMAND": f"{self.python} -m x append",
            "ONEX_LEDGER_READER_COMMAND": f"{self.python} -m omnimarket.nodes.node_work_ledger_reader_effect",
        }

    def set_claims(self, *claims: tuple[str, str, str, str, bool]) -> None:
        """Each claim is (stamp, lane, ticket, text, open). A closed claim is
        still a CLAIM row of the window; the node leaves it out of open-claims."""
        self._claims = list(claims)
        rows = [
            f"{stamp} | CLAIM | lane={lane} | ticket={ticket} | {text}"
            for stamp, lane, ticket, text, _ in self._claims
        ]
        opened = [
            f"OPEN-CLAIM lane={lane} claim={stamp} db:{n} ticket={ticket} "
            f"newest-row={stamp} kind=CLAIM age=0h01m rows=1 host=h1 run=none "
            "leases=none liveness=unknown"
            for n, (stamp, lane, ticket, _, is_open) in enumerate(self._claims)
            if is_open
        ]
        head = f"open-claims since=2026-09-21T00:00:00Z claims={len(rows)} open={len(opened)}"
        (self.directory / "rows.txt").write_text(
            "".join(f"{r}\n" for r in rows), encoding="utf-8"
        )
        (self.directory / "open.txt").write_text(
            "".join(f"{line}\n" for line in [head, *opened]), encoding="utf-8"
        )

    def refuse_with(self, code: int) -> None:
        """Every later read exits with ``code`` and prints no row."""
        (self.directory / "exit").write_text(str(code), encoding="utf-8")

    def calls(self) -> list[str]:
        """The argv of every reader call, one string per call."""
        log = self.directory / "reader.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []
