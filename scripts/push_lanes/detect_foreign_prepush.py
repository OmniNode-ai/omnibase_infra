# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-side DETECTION for foreign (queue-invisible) heavy prepush runs on the
`.201` gate-runner host (OMN-16968, host-side guard half).

WHAT THIS IS
------------
The enqueue-only mechanism (``scripts/hooks/prepush_201_enqueue.py``) closes
the ROOT CAUSE by giving the pre-push hook an executable path onto
``~/push-lanes/QUEUE`` instead of leaving a human/agent to improvise a
detached run. This module is the MECHANICAL BACKSTOP on the `.201` host
itself: a periodic scan (run by cron or a `queue-runner.sh` tick) that finds
any ``prepush_smart_tests.sh`` process NOT descended from the live,
flock-holding ``queue-runner.sh`` loop, and reports it LOUDLY.

DETECTION ONLY -- NO MUTATION
------------------------------
This module never signals, renices, or otherwise touches a process it finds.
It only classifies and logs. That is a deliberate, scoped decision for this
ticket (OMN-16968), not an oversight: killing or reprioritizing a running
heavy suite mid-flight destroys whatever partial evidence it was producing
and could itself strand the lane that launched it. See the OMN-16968 ticket
body's own constraint ("no kills, no reordering") and the dispatch brief's
"detection first; automatic deprioritization only if it can be done safely
and reversibly" -- this PR ships detection; deprioritization (e.g. `renice`)
is explicitly NOT implemented here and remains open follow-on work.

HOW "FOREIGN" IS DECIDED
-------------------------
``queue-runner.sh`` holds ``~/push-lanes/.runner.lock`` via ``flock`` for its
entire lifetime (see that script: ``exec 9>"$LOCK"; flock -n 9 || exit 0``),
and every governed push it runs is a DESCENDANT of that one process (it does
``cd "$WORKTREE" && git push ...`` in a subshell, which execs the git
pre-push hook, which execs ``prepush_smart_tests.sh``). So:

  1. Find the live ``queue-runner.sh`` loop process (there is at most one --
     its own flock guarantees that; see ``find_queue_runner_pid``).
  2. Find every live ``prepush_smart_tests.sh`` process.
  3. A ``prepush_smart_tests.sh`` process is LEGIT iff the queue-runner pid
     appears somewhere in its ancestor (``ppid``) chain. Everything else is
     FOREIGN -- including the case where NO queue-runner is running at all
     (any prepush process found then is foreign by construction, since
     nothing could have launched it through the queue).

This is exactly the shape of the real incident evidence (OMN-16968): the
foreign PIDs measured there (``docker exec omninode-gate-runner bash
scripts/hooks/prepush_smart_tests.sh``) have a `bash -c` ancestor chain
rooted at a `docker exec` invocation, never at the queue-runner loop.

TESTABILITY
-----------
The classification logic (``find_queue_runner_pid``, ``is_descendant``,
``scan``) is a pure function of a process table -- no subprocess, no I/O --
so it is fully covered against a FAKE process table
(``tests/unit/scripts/test_detect_foreign_prepush.py``), including the exact
ancestry shape measured live during the OMN-16968 investigation. Only
``collect_processes`` (a thin ``ps -eo pid,ppid,args`` wrapper) and
``main``/``write_report`` (I/O) are untested by design -- OS wrappers, not
logic.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_LOG_PATH = "~/push-lanes/foreign-prepush-detector.log"
QUEUE_RUNNER_PATTERN = re.compile(r"queue-runner\.sh\b")
PREPUSH_PATTERN = re.compile(r"prepush_smart_tests\.sh\b")
MAX_ANCESTRY_DEPTH = 128  # defensive bound against a corrupt/cyclic ps table


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    cmd: str


@dataclass(frozen=True)
class ScanResult:
    queue_runner_pid: int | None
    legit_pids: tuple[int, ...] = field(default_factory=tuple)
    foreign_pids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def all_prepush_pids(self) -> tuple[int, ...]:
        return tuple(sorted((*self.legit_pids, *self.foreign_pids)))


def parse_ps_output(text: str) -> list[ProcessInfo]:
    """Parse `ps -eo pid,ppid,args` output (header line + one row per process).

    Tolerant of extra leading/trailing whitespace; skips any line that does
    not start with two integers (defensive against a malformed/truncated
    read rather than crashing the whole scan on one bad line).
    """
    processes: list[ProcessInfo] = []
    lines = text.splitlines()
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        if len(parts) < 2:
            continue
        pid_s, ppid_s = parts[0], parts[1]
        cmd = parts[2] if len(parts) > 2 else ""
        if not (pid_s.isdigit() and ppid_s.isdigit()):
            continue  # header row ("PID PPID COMMAND") or garbage
        processes.append(ProcessInfo(pid=int(pid_s), ppid=int(ppid_s), cmd=cmd))
    return processes


def find_queue_runner_pid(processes: Sequence[ProcessInfo]) -> int | None:
    """The live queue-runner.sh loop's pid, or None if it is not running.

    `queue-runner.sh`'s own `flock -n` guarantees at most one live instance,
    so the first match is unambiguous. If somehow more than one line matches
    (a transient double-launch race before the second one's flock fails),
    the lowest pid wins deterministically -- callers must not depend on which
    one, only that a stable choice is made.
    """
    matches = sorted(p.pid for p in processes if QUEUE_RUNNER_PATTERN.search(p.cmd))
    return matches[0] if matches else None


def is_descendant(pid: int, ancestor_pid: int, by_pid: dict[int, ProcessInfo]) -> bool:
    """True if ancestor_pid appears in pid's ppid chain, within a bounded walk."""
    current = pid
    seen: set[int] = set()
    for _ in range(MAX_ANCESTRY_DEPTH):
        if current in seen:
            return False  # cyclic ps table -- defensive, never loops forever
        seen.add(current)
        proc = by_pid.get(current)
        if proc is None:
            return False
        if proc.ppid == ancestor_pid:
            return True
        if proc.ppid == current or proc.ppid <= 1:
            return False
        current = proc.ppid
    return False


def scan(processes: Sequence[ProcessInfo]) -> ScanResult:
    """Pure classification: which prepush_smart_tests.sh pids are foreign."""
    by_pid = {p.pid: p for p in processes}
    queue_runner_pid = find_queue_runner_pid(processes)
    prepush_procs = [p for p in processes if PREPUSH_PATTERN.search(p.cmd)]

    legit: list[int] = []
    foreign: list[int] = []
    for proc in prepush_procs:
        if queue_runner_pid is not None and is_descendant(
            proc.pid, queue_runner_pid, by_pid
        ):
            legit.append(proc.pid)
        else:
            foreign.append(proc.pid)

    return ScanResult(
        queue_runner_pid=queue_runner_pid,
        legit_pids=tuple(sorted(legit)),
        foreign_pids=tuple(sorted(foreign)),
    )


def collect_processes() -> list[ProcessInfo]:
    """Live `ps -eo pid,ppid,args` read. Thin OS wrapper -- not unit tested."""
    proc = subprocess.run(
        ["ps", "-eo", "pid,ppid,args"],
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_ps_output(proc.stdout)


def format_report(result: ScanResult, *, now: str) -> str:
    if result.foreign_pids:
        status = (
            f"[{now}] FOREIGN PREPUSH DETECTED: {len(result.foreign_pids)} process(es) "
            f"running prepush_smart_tests.sh OUTSIDE queue-runner.sh's control "
            f"(pids: {' '.join(str(p) for p in result.foreign_pids)}). "
            f"queue_runner_pid={result.queue_runner_pid if result.queue_runner_pid is not None else 'NOT RUNNING'}. "
            f"legit (queue-descended) pids: {' '.join(str(p) for p in result.legit_pids) or 'none'}. "
            "NOT KILLED, NOT RENICED -- detection only (OMN-16968). "
            "See docs/runbooks/201-gate-runner-queue.md."
        )
    else:
        status = (
            f"[{now}] clean: {len(result.legit_pids)} queue-descended prepush process(es), "
            f"0 foreign. queue_runner_pid={result.queue_runner_pid if result.queue_runner_pid is not None else 'NOT RUNNING'}."
        )
    return status


def write_report(line: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write(line + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect_foreign_prepush.py",
        description=(
            "Detect (never kill) prepush_smart_tests.sh processes on the .201 "
            "gate-runner host that are not descendants of the live queue-runner.sh "
            "loop -- the class of foreign run that stranded 4 branches on "
            "2026-08-29 (OMN-16968)."
        ),
    )
    parser.add_argument(
        "--log-file",
        default=DEFAULT_LOG_PATH,
        help=f"append the report line here (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not also print the report line to stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    processes = collect_processes()
    result = scan(processes)
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = format_report(result, now=now)
    write_report(line, Path(args.log_file).expanduser())
    if not args.quiet:
        print(line, file=sys.stderr if result.foreign_pids else sys.stdout)
    # Detection-only: a foreign process found is not this SCRIPT failing, so
    # exit 0 regardless -- a cron job that "fails" on a normal detection would
    # train an operator to ignore its own non-zero exits. The log line is the
    # signal, not the exit code.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
