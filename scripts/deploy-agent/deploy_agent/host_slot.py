#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One lab host, one tenant at a time: the host-slot lease (OMN-19544 AC1).

WHY THIS EXISTS
---------------

The .105 lab host (omnibook, a laptop on Docker Desktop) runs a Docker VM of
15.6 GiB. The dogfood stack holds about 4.1 GiB of it, and two more tenants want
the rest: the dev-105 deployed lane that a deploy-agent instance rebuilds on
every routed merge (the dev-202 stack it copies idles at about 7.3 GiB), and the
prove-105 proof stack (about 5 GiB for each proof). Both do not fit at once
(lab deploy lanes plan, omni_home ledger TERMINAL 2026-09-25T10:29:02Z, lane
lab-deploy-lanes-plan). Growing the VM restarts Docker Desktop, and with it the
CI runner container on that host, so the tenants take turns instead.

Taking turns needs one record that both tenants read and write, on the host
where they run. The ledger HOLD a prover places lives in omni_home on another
machine, and the deploy agent cannot read it. So the prover mirrors its HOLD
here as a lease, and the agent takes the same lease for each job:

* the agent refuses to accept a command while another owner's lease is
  unexpired (rejection reason ``busy``, the reason an agent with a running job
  already gives), and holds the lease for the whole job plus a verify window
  afterwards, so the post-merge verify job reads the lane it just built;
* a prover's ``acquire`` refuses while the agent's lease is unexpired, and
  names the holder.

THE LEASE
---------

One JSON file, ``lease.json``, in a directory named by
``DEPLOY_AGENT_HOST_SLOT_DIR`` (or ``--dir`` on the command line). It records
the owner, when it was acquired, an absolute ``until`` and a reason (a prover
puts its ledger HOLD id there). Every read-modify-write happens under an
``fcntl`` lock on ``slot.lock`` beside it (macOS has no ``flock(1)``), and the
file is replaced atomically, so two acquirers can never both win.

A lease ends at its ``until`` or when its owner releases it. It is never
stolen: nobody but the owner removes it, and an expired lease is simply not
in force. The same owner acquiring again replaces its own lease, which is how
the agent turns its job lease into its verify-window lease.

WHAT THIS DOES NOT DO
---------------------

It does not stop or start either tenant's stack. A prover that needs the VM
headroom stops the dev-105 project after it holds the slot; the agent's next
job brings the lane back up. It is off unless ``DEPLOY_AGENT_HOST_SLOT_DIR`` is
set, which only a time-shared instance's env file sets, so the .201 and .202
agents behave exactly as before.

COMMAND LINE (for a prover; standard library only, so it runs from any clone)
------------------------------------------------------------------------------

    python3 host_slot.py --dir DIR acquire --owner prove-105 \\
        --until 2026-09-25T12:56:00Z --reason 2026-09-25T10:26:00Z-prove-105
    python3 host_slot.py --dir DIR release --owner prove-105
    python3 host_slot.py --dir DIR show

Exit codes: 0 done, 2 held by another owner (the holder is printed on stderr),
3 usage error.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

#: Names the lease directory. Unset means this agent has no host slot.
ENV_HOST_SLOT_DIR = "DEPLOY_AGENT_HOST_SLOT_DIR"
#: How long the agent keeps the slot after a job ends, so the post-merge verify
#: job reads the lane the job built. Seconds; the default is 30 minutes.
ENV_HOST_SLOT_VERIFY_WINDOW = "DEPLOY_AGENT_HOST_SLOT_VERIFY_WINDOW_SECONDS"
DEFAULT_VERIFY_WINDOW_SECONDS = 1800
#: How long the agent's lease runs while a job is in flight. The job replaces it
#: with its verify-window lease when it ends, so this bounds only a job whose
#: process died: the build ceiling (build_budget.HARD_UPPER_BOUND_SECONDS, one
#: hour) plus the lane-lock wait and the compose and verify phases, with room.
JOB_LEASE_SECONDS = 3 * 3600

LEASE_FILE = "lease.json"
LOCK_FILE = "slot.lock"

EXIT_OK = 0
EXIT_HELD = 2
EXIT_USAGE = 3


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> datetime:
    """Parse an absolute UTC timestamp such as ``2026-09-25T12:56:00Z``."""
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if value.tzinfo is None:
        msg = f"timestamp {text!r} carries no timezone; write it in UTC with Z"
        raise ValueError(msg)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class Lease:
    """Who holds the host slot, since when, until when, and why."""

    owner: str
    acquired_at: str
    until: str
    reason: str = ""

    def in_force(self, now: datetime) -> bool:
        return parse_utc(self.until) > now

    def describe(self) -> str:
        reason = f", reason {self.reason}" if self.reason else ""
        return (
            f"host slot held by {self.owner} since {self.acquired_at} "
            f"until {self.until}{reason}"
        )


class HostSlotHeldError(RuntimeError):
    """Another owner's lease is in force; the caller must not start."""

    def __init__(self, holder: Lease) -> None:
        super().__init__(holder.describe())
        self.holder = holder


class HostSlot:
    """The lease file for one host, read and written under one fcntl lock."""

    def __init__(
        self, directory: str | Path, *, clock: Callable[[], datetime] = _utc_now
    ) -> None:
        self.directory = Path(directory)
        self._clock = clock

    def now(self) -> datetime:
        """The clock this slot compares ``until`` with."""
        return self._clock()

    @property
    def lease_path(self) -> Path:
        return self.directory / LEASE_FILE

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.directory / LOCK_FILE, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _read_unlocked(self) -> Lease | None:
        try:
            raw = json.loads(self.lease_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        return Lease(
            owner=str(raw["owner"]),
            acquired_at=str(raw["acquired_at"]),
            until=str(raw["until"]),
            reason=str(raw.get("reason", "")),
        )

    def _write_unlocked(self, lease: Lease) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".lease-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(asdict(lease), fh, sort_keys=True)
                fh.write("\n")
            Path(tmp).replace(self.lease_path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def holder(self) -> Lease | None:
        """The lease in force now, or ``None`` when the slot is free."""
        with self._locked():
            lease = self._read_unlocked()
        if lease is None or not lease.in_force(self._clock()):
            return None
        return lease

    def held_by_other(self, owner: str) -> Lease | None:
        """The lease in force now when someone other than ``owner`` holds it."""
        lease = self.holder()
        if lease is None or lease.owner == owner:
            return None
        return lease

    def acquire(self, owner: str, until: datetime, reason: str = "") -> Lease:
        """Take or extend the slot for ``owner`` until ``until``.

        Raises ``HostSlotHeldError`` when another owner's lease is in force. The
        same owner acquiring again replaces its own lease.
        """
        if not owner.strip():
            msg = "a lease needs a non-empty owner"
            raise ValueError(msg)
        now = self._clock()
        if until <= now:
            msg = f"until {_format(until)} is not after now {_format(now)}"
            raise ValueError(msg)
        with self._locked():
            current = self._read_unlocked()
            if current is not None and current.owner != owner and current.in_force(now):
                raise HostSlotHeldError(current)
            acquired_at = (
                current.acquired_at
                if current is not None
                and current.owner == owner
                and current.in_force(now)
                else _format(now)
            )
            lease = Lease(
                owner=owner,
                acquired_at=acquired_at,
                until=_format(until),
                reason=reason,
            )
            self._write_unlocked(lease)
            return lease

    def release(self, owner: str) -> bool:
        """Remove ``owner``'s lease. Never removes another owner's."""
        with self._locked():
            current = self._read_unlocked()
            if current is None or current.owner != owner:
                return False
            self.lease_path.unlink()
            return True


@contextmanager
def job_lease(
    slot: HostSlot | None,
    owner: str,
    *,
    verify_window: timedelta,
    job_seconds: int = JOB_LEASE_SECONDS,
    reason: str = "",
) -> Iterator[None]:
    """Hold the host slot for one deploy job, then for its verify window.

    Raises ``HostSlotHeldError`` before the body runs when another owner holds
    the slot, so a job that cannot have the host touches nothing. When the body
    ends, however it ends, the lease is replaced by one that runs for
    ``verify_window`` from that moment, or released when the window is zero.
    ``slot`` ``None`` (no host slot on this instance) is a no-op.
    """
    if slot is None:
        yield
        return
    slot.acquire(
        owner, slot.now() + timedelta(seconds=job_seconds), reason or "deploy job"
    )
    try:
        yield
    finally:
        if verify_window > timedelta(0):
            slot.acquire(
                owner,
                slot.now() + verify_window,
                f"verify window after {reason or 'deploy job'}",
            )
        else:
            slot.release(owner)


def host_slot_from_env(env: dict[str, str] | None = None) -> HostSlot | None:
    """The host slot named by ``DEPLOY_AGENT_HOST_SLOT_DIR``, or ``None``."""
    source = os.environ if env is None else env
    directory = source.get(ENV_HOST_SLOT_DIR, "").strip()
    return HostSlot(directory) if directory else None


def verify_window_from_env(env: dict[str, str] | None = None) -> timedelta:
    """How long the agent keeps the slot after a job ends."""
    source = os.environ if env is None else env
    raw = source.get(ENV_HOST_SLOT_VERIFY_WINDOW, "").strip()
    if not raw:
        return timedelta(seconds=DEFAULT_VERIFY_WINDOW_SECONDS)
    seconds = int(raw)
    if seconds < 0:
        msg = f"{ENV_HOST_SLOT_VERIFY_WINDOW} must not be negative, got {seconds}"
        raise ValueError(msg)
    return timedelta(seconds=seconds)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Take, release or read the lab host slot (OMN-19544)."
    )
    parser.add_argument(
        "--dir",
        default=os.environ.get(ENV_HOST_SLOT_DIR, ""),
        help=f"the lease directory (default: ${ENV_HOST_SLOT_DIR})",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    acq = sub.add_parser("acquire", help="take or extend the slot")
    acq.add_argument("--owner", required=True)
    acq.add_argument("--until", required=True, help="absolute UTC, e.g. ...T12:56:00Z")
    acq.add_argument("--reason", default="", help="e.g. the ledger HOLD id")
    rel = sub.add_parser("release", help="give the slot back")
    rel.add_argument("--owner", required=True)
    sub.add_parser("show", help="print the lease in force, if any")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.dir:
        print(f"host_slot: pass --dir or set {ENV_HOST_SLOT_DIR}", file=sys.stderr)
        return EXIT_USAGE
    slot = HostSlot(args.dir)
    if args.command == "acquire":
        try:
            lease = slot.acquire(args.owner, parse_utc(args.until), args.reason)
        except HostSlotHeldError as exc:
            print(f"host_slot: REFUSED, {exc}", file=sys.stderr)
            return EXIT_HELD
        except ValueError as exc:
            print(f"host_slot: {exc}", file=sys.stderr)
            return EXIT_USAGE
        print(json.dumps(asdict(lease), sort_keys=True))
        return EXIT_OK
    if args.command == "release":
        released = slot.release(args.owner)
        print(
            f"host_slot: {'released' if released else 'nothing held by'} {args.owner}"
        )
        return EXIT_OK
    current = slot.holder()
    print(json.dumps(asdict(current), sort_keys=True) if current else "host_slot: free")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
