#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Host-level exclusive lane lock, one per compose project (OMN-16729).

Why this exists
---------------
``onex-runtime-deploy``, ``scripts/runtime_build/refresh_dev_lane.sh``
and ``scripts/runtime_build/refresh_stability_lane.sh`` all mutate ONE lane --
a compose project -- and until this module none of them took a host-level lock
over the whole build/gate/readback critical section. ``onex-runtime-deploy``'s
own ``.deploy.lock`` is host-WIDE (it serialises unrelated lanes against each
other) and, decisively, it is scoped to that script alone: a refresh wrapper
holds nothing while it captures pre-state, health-gates, and reads back, which
is precisely the window in which two sanctioned dev-lane refreshes collided on
2026-09-08 at 13:51Z (third recorded occurrence of the class).

Design
------
* **One lock per compose project.** ``<lock dir>/<compose_project>.lock``.
  Two different lanes never block each other; two touches of the SAME lane
  always serialise.
* **fcntl, not a lock directory.** ``flock(2)`` is released by the kernel when
  the last file descriptor on the open file description closes, so a killed
  holder cannot leave a lock behind. That removes the stale-lock branch
  entirely -- and with it the *stealing* that a mkdir/pid lock needs in order
  to recover. This lock is never stolen.
* **``flock(1)`` is not used** because macOS ships none (memory
  ``reference_macos_no_flock_use_fcntl_shim``); the lab host is Linux but the
  tests run on both, so one Python shim is used consistently everywhere.
* **The caller owns the fd.** The shell opens the lock file on a numbered fd
  (``exec 9>"$lock"``) and hands the NUMBER here; this process flocks the
  inherited descriptor and exits. The lock survives because the parent shell
  still holds the same open file description -- and it is released
  automatically when the shell exits, however it exits.
* **A holder sidecar next to the lock** records pid / start time / lane / ref /
  argv, so a contended acquisition names WHO holds the lane instead of hanging
  anonymously.

Residual, stated rather than implied
------------------------------------
An fcntl lock lives on the open file description, and a child process inherits
that description. So a lane stays locked for as long as ANY process in the
holder's tree still holds the descriptor -- normally the right answer (a
still-running ``docker compose up`` belonging to a killed deploy is exactly when
a second refresh must not start), but it means an orphaned child can outlive its
shell. That case is diagnosable rather than silent: the holder sidecar names the
shell's pid, and ``describe`` reports it as ``NOT RUNNING`` while the lock is
still held, which is the signal to look for the surviving child. It is still not
stolen -- stealing a lock on a liveness heuristic is what produced the
mkdir/pid-file lock's recovery branch, and a wrong guess there mutates a live
lane.

Reading the lock (OMN-19424)
----------------------------
``status --compose-project <p>`` prints one JSON object (``held``, ``holder``,
``holder_alive``, ``lock_path``) and exits 0. ``held`` is read from the flock
itself, not from the sidecar, so a reader that must WAIT while a reconcile holds
the lane (the runtime-train pre-merge check) does not mistake a stale sidecar
for a live holder.

Exit codes
----------
    0  acquired (or nothing to do)
    2  contended -- the bounded wait expired; the holder is named on stderr
    3  usage / precondition error
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import re
import socket
import sys
import time
from datetime import UTC, datetime, timezone
from pathlib import Path

EXIT_OK = 0
EXIT_CONTENDED = 2
EXIT_USAGE = 3

DEFAULT_TIMEOUT_SECONDS = 900.0
POLL_SECONDS = 0.5

# Same character class onex-runtime-deploy's resolve_compose_project() enforces.
# Enforced here too so a compose-project string can never traverse out of the
# lock directory.
PROJECT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def lock_dir() -> Path:
    """Directory holding one lock file per compose project.

    ``ONEX_LANE_LOCK_DIR`` exists for tests; the operational path is fixed so
    two lanes on the same host can never disagree about where the lock lives.
    """
    override = os.environ.get("ONEX_LANE_LOCK_DIR", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".omnibase" / "state" / "lane-locks"


def validate_project(project: str) -> str:
    if not PROJECT_RE.match(project):
        print(
            f"lane_lock: invalid compose project name {project!r} -- expected "
            "only alphanumerics, hyphens and underscores.",
            file=sys.stderr,
        )
        raise SystemExit(EXIT_USAGE)
    return project


def lock_path(project: str) -> Path:
    return lock_dir() / f"{validate_project(project)}.lock"


def holder_path(project: str) -> Path:
    return lock_dir() / f"{validate_project(project)}.lock.holder"


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def describe_holder(project: str) -> str:
    """Human-readable description of the current holder, for a contention message."""
    try:
        data = json.loads(holder_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "holder unknown (no readable holder sidecar next to the lock)"
    pid = data.get("pid")
    try:
        alive = "alive" if pid is not None and pid_alive(int(pid)) else "NOT RUNNING"
    except (TypeError, ValueError):
        alive = "unknown"
    return (
        f"pid={pid} ({alive}) started={data.get('started_at')} "
        f"host={data.get('host')} lane={data.get('lane')} "
        f"compose_project={data.get('compose_project')} ref={data.get('ref')} "
        f"argv={data.get('argv')}"
    )


def write_holder(project: str, lane: str, ref: str, argv: str) -> None:
    path = holder_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getppid(),
        "acquirer_pid": os.getpid(),
        "started_at": utc_now(),
        "host": socket.gethostname(),
        "lane": lane,
        "compose_project": project,
        "ref": ref,
        "argv": argv,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def cmd_path(args: argparse.Namespace) -> int:
    print(lock_path(args.compose_project))
    return EXIT_OK


def cmd_describe(args: argparse.Namespace) -> int:
    print(describe_holder(args.compose_project))
    return EXIT_OK


def lock_is_held(project: str) -> bool:
    """True while some open file description holds the lane's flock.

    Probes with a non-blocking flock on a fresh open of the lock file and drops
    it at once, so it never disturbs a holder. A lane whose lock file does not
    exist has never been locked. The lock file is opened read-only: flock does
    not need write access, and a probe must not create the file.
    """
    try:
        fd = os.open(lock_path(project), os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EACCES):
                return True
            raise
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def cmd_status(args: argparse.Namespace) -> int:
    """Print one JSON object: is the lane locked right now, and by whom.

    ``held`` comes from the flock itself; ``holder`` is the sidecar as written
    by ``acquire`` (any extra keys a holder recorded ride through), or null when
    there is none. After a killed holder the sidecar can outlive the lock, so
    ``held`` false with a non-null holder means the sidecar is stale.
    """
    project = validate_project(args.compose_project)
    holder: dict[str, object] | None
    try:
        holder = json.loads(holder_path(project).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        holder = None
    holder_alive: bool | None = None
    if holder is not None:
        try:
            holder_alive = pid_alive(int(str(holder.get("pid"))))
        except ValueError:
            holder_alive = None
    report = {
        "compose_project": project,
        "lock_path": str(lock_path(project)),
        "held": lock_is_held(project),
        "holder": holder,
        "holder_alive": holder_alive,
    }
    print(json.dumps(report, sort_keys=True))
    return EXIT_OK


def cmd_acquire(args: argparse.Namespace) -> int:
    """flock an fd the CALLER opened, so the lock outlives this process."""
    project = validate_project(args.compose_project)
    fd = args.fd
    try:
        os.fstat(fd)
    except OSError as exc:
        if exc.errno == errno.EBADF:
            print(
                f"lane_lock: fd {fd} is not open in this process -- the caller must "
                f'open the lock file on that descriptor first (exec {fd}>"$lock").',
                file=sys.stderr,
            )
            return EXIT_USAGE
        raise

    deadline = time.monotonic() + max(args.timeout, 0.0)
    announced = False
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if not announced:
                print(
                    f"[lane-lock] waiting for lane '{project}' "
                    f"({lock_path(project)}) -- held by {describe_holder(project)}",
                    file=sys.stderr,
                )
                announced = True
            if time.monotonic() >= deadline:
                print(
                    f"[lane-lock] CONTENDED: lane '{project}' is held by another "
                    f"process and did not free within {args.timeout:g}s.",
                    file=sys.stderr,
                )
                print(f"[lane-lock]   lock:   {lock_path(project)}", file=sys.stderr)
                print(
                    f"[lane-lock]   holder: {describe_holder(project)}",
                    file=sys.stderr,
                )
                print(
                    "[lane-lock]   The lock is NEVER stolen. Wait for the holder to "
                    "finish, or stop that process yourself by its own pid.",
                    file=sys.stderr,
                )
                return EXIT_CONTENDED
            time.sleep(POLL_SECONDS)

    write_holder(project, args.lane, args.ref, args.argv)
    print(
        f"[lane-lock] acquired lane '{project}' ({lock_path(project)}) "
        f"for lane={args.lane or 'unknown'} ref={args.ref or 'unknown'}",
        file=sys.stderr,
    )
    return EXIT_OK


def cmd_release(args: argparse.Namespace) -> int:
    """Remove the holder sidecar.

    The LOCK itself is released by the calling shell closing its fd (or by the
    kernel when that shell dies) -- there is deliberately nothing here that can
    release another process's lock.
    """
    try:
        holder_path(args.compose_project).unlink()
    except OSError:
        pass
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_path = sub.add_parser(
        "path", help="print the lock file path for a compose project"
    )
    p_path.add_argument("--compose-project", required=True)
    p_path.set_defaults(func=cmd_path)

    p_desc = sub.add_parser("describe", help="print the current holder of a lane lock")
    p_desc.add_argument("--compose-project", required=True)
    p_desc.set_defaults(func=cmd_describe)

    p_status = sub.add_parser(
        "status", help="print JSON: is the lane locked now, and its holder"
    )
    p_status.add_argument("--compose-project", required=True)
    p_status.set_defaults(func=cmd_status)

    p_acq = sub.add_parser("acquire", help="flock a caller-opened fd, bounded wait")
    p_acq.add_argument("--compose-project", required=True)
    p_acq.add_argument("--fd", type=int, required=True)
    p_acq.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    p_acq.add_argument("--lane", default="")
    p_acq.add_argument("--ref", default="")
    p_acq.add_argument("--argv", default="")
    p_acq.set_defaults(func=cmd_acquire)

    p_rel = sub.add_parser("release", help="remove the holder sidecar")
    p_rel.add_argument("--compose-project", required=True)
    p_rel.set_defaults(func=cmd_release)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
