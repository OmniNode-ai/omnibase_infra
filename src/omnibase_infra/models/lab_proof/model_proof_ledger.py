# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""The rolling work ledger as the lab proof driver reads it (OMN-19998).

Two reads of the same rows live here, apart from the driver script:

* the HOLD rows still in force on a lab host surface (moved out of
  ``prepr_runtime_pool.py``, which only shrinks);
* the verdict of each repo's last proof, from the ``RELEASE`` rows a prover
  writes as ``LAB PROOF <PASS|FAIL|INCONCLUSIVE> ... <repo>#<pr> ...``, which
  decides the alarm after two INCONCLUSIVE proofs in a row for one repo.

Nothing here writes: the driver hands :meth:`alarm_on_streak` the ledger's own
append, and a failed append is reported in the proof text, never swallowed.
"""

from __future__ import annotations

import datetime as dt
import re
import shlex
import subprocess
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

ALARM_LANE = "lab-proof-alarm"
ALARM_TICKET = "OMN-19998"
ALARM_EVENT = "lab-proof-inconclusive-streak"

_ROW = re.compile(r"^(\S+Z) \| (HOLD|RELEASE) \|")
_VERDICT = re.compile(r"LAB PROOF (PASS|FAIL|INCONCLUSIVE)\b:?\s+(.*)$")
_PR_REF = re.compile(r"\b([A-Za-z0-9_.-]+)#\d+")


def _field(row: str, key: str) -> str | None:
    m = re.search(rf"(?:^|\| |\s){key}=([^\s|]+)", row)
    return m.group(1) if m else None


def _row_id(row: str, stamp: str) -> str:
    return _field(row, "id") or f"{stamp}-{_field(row, 'lane')}"


class ModelProofLedger(BaseModel):
    """Ledger rows, oldest first, as the driver read them at the start of a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lines: tuple[str, ...] = ()

    def live_surface_holds(self, now: dt.datetime) -> dict[str, tuple[str, str, str]]:
        """Map surface -> (hold id, lane, until) for every HOLD still in force.

        A HOLD is in force until its ``until=`` passes or a RELEASE cites its id
        (``re=<id>``). Rows without ``surface=`` are not host leases and are ignored.
        """
        holds: dict[str, tuple[str, str, str, dt.datetime]] = {}
        released: set[str] = set()
        for line in self.lines:
            m = _ROW.match(line)
            if not m:
                continue
            if m.group(2) == "RELEASE":
                ref = _field(line, "re")
                if ref:
                    released.add(ref)
                continue
            surface, until = _field(line, "surface"), _field(line, "until")
            if not surface or not until:
                continue
            try:
                until_ts = dt.datetime.fromisoformat(until.replace("Z", "+00:00"))
            except ValueError:
                continue
            holds[_row_id(line, m.group(1))] = (
                surface,
                _field(line, "lane") or "?",
                until,
                until_ts,
            )
        return {
            surface: (hold_id, lane, until)
            for hold_id, (surface, lane, until, until_ts) in holds.items()
            if hold_id not in released and until_ts > now
        }

    def last_proof_verdicts(self) -> dict[str, tuple[str, str]]:
        """Map repo -> (verdict, proof id) of the newest proof row that named it.

        The proof id is the HOLD id the row releases (``re=``). The repos a row
        names are the ``<repo>#<pr>`` references before its first `` on ``, so a
        later mention of another pull request in the free text is not read.
        """
        last: dict[str, tuple[str, str]] = {}
        for line in self.lines:
            m = _ROW.match(line)
            verdict = _VERDICT.search(line)
            if not m or m.group(2) != "RELEASE" or verdict is None:
                continue
            proof_id = _field(line, "re")
            if not proof_id:
                continue
            subject = verdict.group(2).split(" on ", 1)[0]
            for repo in _PR_REF.findall(subject):
                last[repo] = (verdict.group(1), proof_id)
        return last

    @staticmethod
    def no_append(_row: str) -> None:
        """The append of a run with no ledger tool configured: it refuses, so the proof says so."""
        raise RuntimeError("no ledger append configured")

    @staticmethod
    def appender(command: str, ledger: str | None) -> Callable[[str], None]:
        """Bind the ledger tool's own append (``<command> <ledger> --timeout 60s --append <row>``).

        The row is one argv element, never a shell string. :meth:`no_append` when
        either the command or the ledger path is unset, so the proof says so.
        """
        argv = shlex.split(command)
        if not argv or not ledger:
            return ModelProofLedger.no_append

        def append(row: str) -> None:
            done = subprocess.run(
                [*argv, ledger, "--timeout", "60s", "--append", row],
                capture_output=True,
                text=True,
                check=False,
            )
            if done.returncode != 0:
                why = (done.stderr or done.stdout).strip()[:200]
                raise RuntimeError(f"onex-ledger exited {done.returncode}: {why}")

        return append

    @staticmethod
    def streak_row(repo: str, earlier: str, current: str, stamp: str) -> str:
        """The STATUS row (a kind the ledger grammar accepts) that raises the alarm."""
        cells = [
            stamp,
            "STATUS",
            f"lane={ALARM_LANE}",
            f"ticket={ALARM_TICKET}",
            "actor=script:prepr-runtime-pool",
            "model=none",
            f"event={ALARM_EVENT}",
            f"repo={repo}",
            f"proofs={earlier},{current}",
            f"two lab proofs of {repo} in a row ended INCONCLUSIVE ({earlier}, "
            f"{current}); the proof logs are kept beside each receipt",
        ]
        return " | ".join(cells)

    def alarm_on_streak(
        self,
        repos: Sequence[str],
        current: str,
        stamp: str,
        append: Callable[[str], None],
    ) -> str:
        """Write one alarm row per repo whose last proof and this one are both INCONCLUSIVE.

        The caller asks only when this proof ended INCONCLUSIVE.

        Returns the lines to append to the proof text, each led by a newline: one per
        row written, and one when a row could not be written or no append was
        configured. Empty when nothing alarms.
        """
        last = self.last_proof_verdicts()
        notes: list[str] = []
        for repo in sorted(set(repos)):
            verdict, earlier = last.get(repo, ("", ""))
            if verdict != "INCONCLUSIVE":
                continue
            row = self.streak_row(repo, earlier, current, stamp)
            try:
                append(row)
            except (OSError, RuntimeError) as exc:
                notes.append(f"  alarm NOT written ({exc}): {row}")
                continue
            notes.append(
                f"  alarm written: {ALARM_EVENT} repo={repo} proofs={earlier},{current}"
            )
        return "".join(f"\n{note}" for note in notes)
