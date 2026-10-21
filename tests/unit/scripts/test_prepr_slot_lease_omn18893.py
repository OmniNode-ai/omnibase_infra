# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The pre-PR slot lease admits one holder under a real race (OMN-20085).

The driver leases a host with a directory on it (``mkdir`` is atomic). Breaking
an EXPIRED lease used to be two steps (``rm -rf`` the old one, then ``mkdir``),
so two acquirers that read the same expired lease could both break it: the
second ``rm -rf`` removed the first's fresh lease and both held the slot. These
tests run the driver's own remote commands in real shells, at the same moment,
against one lease directory, and hold the critical section to one winner.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
POOL_PY = REPO_ROOT / "scripts" / "runtime_build" / "prepr_runtime_pool.py"
POOL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "prepr_runtime_pool.yaml"

pytestmark = pytest.mark.unit

_spec = importlib.util.spec_from_file_location("prepr_runtime_pool_lease", POOL_PY)
assert _spec and _spec.loader
pool = importlib.util.module_from_spec(_spec)
sys.modules["prepr_runtime_pool_lease"] = pool
_spec.loader.exec_module(pool)

CFG = pool.load_pool_config(POOL_FIXTURE)
NOW = dt.datetime(2026, 10, 19, 1, 0, tzinfo=dt.UTC)
SLOT = CFG.host("pool-c-slot-1")
ROUNDS = 25
ACQUIRERS = 8


class ShellTransport:
    """Runs each remote command in a real bash whose $HOME is a scratch directory."""

    def __init__(self, home: Path) -> None:
        self.home = home

    def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
        done = subprocess.run(
            ["bash", "-c", command],
            env={**os.environ, "HOME": str(self.home)},
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return done.returncode, done.stdout + done.stderr

    def put(self, host: Any, local: Path, remote: str) -> int:
        return 0


def _plant(home: Path, holder: str, until: dt.datetime) -> Path:
    lease = home / SLOT.lease_dir
    lease.mkdir(parents=True)
    (lease / "lease.json").write_text(
        json.dumps(
            {
                "holder": holder,
                "until": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "taken": "x",
            }
        ),
        encoding="utf-8",
    )
    return lease


def test_renew_own_lease_rewrites_until_and_removes_guard(tmp_path: Path) -> None:
    lease = _plant(tmp_path, "me", NOW + dt.timedelta(minutes=1))
    until = NOW + dt.timedelta(hours=2)
    command = pool.renew_lease_command(SLOT, "me", until, NOW)
    rc, out = ShellTransport(tmp_path).run(SLOT, command + " && echo PHASE-RAN", 30)
    assert rc == 0, out
    assert "PHASE-RAN" in out
    assert (lease / "lease.json").read_text(encoding="utf-8") == json.dumps(
        {
            "holder": "me",
            "until": "2026-10-19T03:00:00Z",
            "taken": "2026-10-19T01:00:00Z",
        }
    )
    assert not (lease / "lease.json.tmp").exists()
    assert not (tmp_path / f"{SLOT.lease_dir}.break").exists()


@pytest.mark.parametrize("holder", ["peer", "me-too"])
def test_renew_foreign_lease_leaves_it_byte_identical(
    tmp_path: Path, holder: str
) -> None:
    lease = _plant(tmp_path, holder, NOW + dt.timedelta(hours=1))
    before = (lease / "lease.json").read_bytes()
    command = pool.renew_lease_command(SLOT, "me", NOW + dt.timedelta(hours=2), NOW)
    rc, out = ShellTransport(tmp_path).run(SLOT, command + " && echo PHASE-RAN", 30)
    assert rc == 97
    assert "LEASE-LOST" in out
    assert before.decode() in out
    assert "PHASE-RAN" not in out
    assert (lease / "lease.json").read_bytes() == before
    assert not (tmp_path / f"{SLOT.lease_dir}.break").exists()


def test_renew_held_guard_refuses_and_leaves_lease_alone(tmp_path: Path) -> None:
    lease = _plant(tmp_path, "me", NOW + dt.timedelta(hours=1))
    before = (lease / "lease.json").read_bytes()
    guard = tmp_path / f"{SLOT.lease_dir}.break"
    guard.mkdir()
    command = pool.renew_lease_command(SLOT, "me", NOW + dt.timedelta(hours=2), NOW)
    rc, out = ShellTransport(tmp_path).run(SLOT, command + " && echo PHASE-RAN", 30)
    assert rc == 97
    assert "LEASE-LOST guard busy" in out
    assert "PHASE-RAN" not in out
    assert (lease / "lease.json").read_bytes() == before
    assert guard.is_dir()


@pytest.mark.parametrize("holder", [None, "me", "peer"])
def test_renew_release_preserves_existing_replies(
    tmp_path: Path, holder: str | None
) -> None:
    if holder is not None:
        _plant(tmp_path, holder, NOW + dt.timedelta(hours=1))
    ok, why = pool.release_lease(CFG, ShellTransport(tmp_path), SLOT, "me")
    if holder == "peer":
        assert not ok and "FOREIGN" in why
        assert (tmp_path / SLOT.lease_dir / "lease.json").is_file()
    else:
        assert ok
        assert why == ("released" if holder else "no lease was present")
        assert not (tmp_path / SLOT.lease_dir).exists()
    assert not (tmp_path / f"{SLOT.lease_dir}.break").exists()


def _race(
    home: Path, names: list[str], release_holder: str | None = None
) -> list[tuple[str, bool]]:
    tp = ShellTransport(home)
    gate = threading.Barrier(len(names) + (release_holder is not None))
    won: list[tuple[str, bool]] = []
    lock = threading.Lock()

    def acquire(name: str) -> None:
        gate.wait()
        ok, _ = pool.acquire_lease(
            CFG, tp, SLOT, name, NOW + dt.timedelta(hours=1), NOW
        )
        with lock:
            won.append((name, ok))

    threads = [threading.Thread(target=acquire, args=(n,)) for n in names]
    if release_holder is not None:

        def release() -> None:
            gate.wait()
            pool.release_lease(CFG, tp, SLOT, release_holder)

        threads.append(threading.Thread(target=release))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
        assert not t.is_alive()
    return won


def test_one_acquirer_wins_a_race_for_an_expired_lease(tmp_path: Path) -> None:
    names = [f"lane-{i}" for i in range(ACQUIRERS)]
    for n in range(ROUNDS):
        home = tmp_path / f"round-{n}"
        _plant(home, "gone-lane", NOW - dt.timedelta(minutes=1))
        won = _race(home, names)
        winners = [name for name, ok in won if ok]
        assert len(won) == ACQUIRERS
        assert len(winners) == 1, f"round {n}: {winners} all hold the slot"
        on_disk = json.loads(
            (home / SLOT.lease_dir / "lease.json").read_text(encoding="utf-8")
        )
        assert on_disk["holder"] == winners[0]
        assert not (home / f"{SLOT.lease_dir}.break").exists()


def test_one_acquirer_wins_a_race_for_a_free_slot(tmp_path: Path) -> None:
    names = [f"lane-{i}" for i in range(ACQUIRERS)]
    for n in range(ROUNDS):
        home = tmp_path / f"free-{n}"
        home.mkdir()
        winners = [name for name, ok in _race(home, names) if ok]
        assert len(winners) == 1, f"round {n}: {winners}"


def test_a_live_lease_is_never_broken(tmp_path: Path) -> None:
    _plant(tmp_path, "peer", NOW + dt.timedelta(minutes=30))
    won = _race(tmp_path, ["a", "b", "c", "d"])
    assert [ok for _, ok in won] == [False] * 4
    holder = json.loads(
        (tmp_path / SLOT.lease_dir / "lease.json").read_text(encoding="utf-8")
    )["holder"]
    assert holder == "peer"


def test_the_break_re_reads_the_lease_and_leaves_a_fresh_one_alone(
    tmp_path: Path,
) -> None:
    """An acquirer that judged a lease expired must not break one a peer took since."""
    lease = _plant(tmp_path, "gone-lane", NOW - dt.timedelta(minutes=1))

    class PeerTakesItAfterTheRead(ShellTransport):
        taken = False

        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            if ".break" in command and not self.taken:
                # the peer broke the expired lease and took it between our read and our break
                type(self).taken = True
                (lease / "lease.json").write_text(
                    json.dumps(
                        {
                            "holder": "peer",
                            "until": "2026-10-19T03:00:00Z",
                            "taken": "x",
                        }
                    ),
                    encoding="utf-8",
                )
            return super().run(host, command, timeout)

    ok, why = pool.acquire_lease(
        CFG,
        PeerTakesItAfterTheRead(tmp_path),
        SLOT,
        "me",
        NOW + dt.timedelta(hours=1),
        NOW,
    )
    assert not ok and "peer" in why, why
    assert (
        json.loads((lease / "lease.json").read_text(encoding="utf-8"))["holder"]
        == "peer"
    )


def _renew_race(home: Path, renew: str, breakers: list[str]) -> list[str]:
    """The owner renews and the breakers acquire at one moment; who holds the slot."""
    tp = ShellTransport(home)
    gate = threading.Barrier(len(breakers) + 1)
    won: list[str] = []
    lock = threading.Lock()

    def acquire(name: str) -> None:
        gate.wait()
        ok, _ = pool.acquire_lease(
            CFG, tp, SLOT, name, NOW + dt.timedelta(hours=1), NOW
        )
        if ok:
            with lock:
                won.append(name)

    def owner() -> None:
        gate.wait()
        _, out = tp.run(SLOT, renew + " && echo PHASE-RAN", 30)
        if "PHASE-RAN" in out:
            with lock:
                won.append("owner")

    threads = [threading.Thread(target=acquire, args=(b,)) for b in breakers]
    threads.append(threading.Thread(target=owner))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return won


def test_renew_races_the_breakers_of_its_expired_lease_and_one_holds(
    tmp_path: Path,
) -> None:
    """The holder renews an expired lease while peers break it: one of them holds the slot.

    The renewal is one remote command and a break is two, so the holder usually
    keeps it; the other order (a breaker first) is the foreign-lease renewal test.
    """
    breakers = [f"lane-{i}" for i in range(ACQUIRERS - 1)]
    renew = pool.renew_lease_command(SLOT, "owner", NOW + dt.timedelta(hours=1), NOW)
    for n in range(ROUNDS):
        home = tmp_path / f"renew-{n}"
        _plant(home, "owner", NOW - dt.timedelta(minutes=1))
        won = _renew_race(home, renew, breakers)
        assert len(won) == 1, f"round {n}: {won} all hold the slot"
        on_disk = json.loads(
            (home / SLOT.lease_dir / "lease.json").read_text(encoding="utf-8")
        )
        assert on_disk["holder"] == won[0]
        assert not (home / f"{SLOT.lease_dir}.break").exists()


def test_release_waits_for_the_guard_and_leaves_the_lease_while_it_is_held(
    tmp_path: Path,
) -> None:
    lease = _plant(tmp_path, "me", NOW + dt.timedelta(hours=1))
    guard = tmp_path / f"{SLOT.lease_dir}.break"
    guard.mkdir()
    ok, why = pool.release_lease(CFG, ShellTransport(tmp_path), SLOT, "me")
    assert not ok and "guard" in why, why
    assert (
        json.loads((lease / "lease.json").read_text(encoding="utf-8"))["holder"] == "me"
    )
    assert guard.is_dir()


def test_release_removes_its_own_lease_and_the_guard(tmp_path: Path) -> None:
    lease = _plant(tmp_path, "me", NOW + dt.timedelta(hours=1))
    assert pool.release_lease(CFG, ShellTransport(tmp_path), SLOT, "me") == (
        True,
        "released",
    )
    assert not lease.exists()
    assert not (tmp_path / f"{SLOT.lease_dir}.break").exists()


def test_release_leaves_a_peers_lease_and_frees_the_guard(tmp_path: Path) -> None:
    lease = _plant(tmp_path, "peer", NOW + dt.timedelta(hours=1))
    ok, why = pool.release_lease(CFG, ShellTransport(tmp_path), SLOT, "me")
    assert not ok, why
    assert (
        json.loads((lease / "lease.json").read_text(encoding="utf-8"))["holder"]
        == "peer"
    )
    assert not (tmp_path / f"{SLOT.lease_dir}.break").exists()


def test_an_expired_holder_releasing_during_a_break_race_leaves_one_holder(
    tmp_path: Path,
) -> None:
    names = [f"lane-{i}" for i in range(6)]
    for n in range(10):
        home = tmp_path / f"release-{n}"
        _plant(home, "gone-lane", NOW - dt.timedelta(minutes=1))
        won = _race(home, names, release_holder="gone-lane")
        winners = {name for name, ok in won if ok}
        assert len(won) == len(names)
        assert len(winners) <= 1, f"round {n}: {winners} all hold the slot"
        if winners:
            on_disk = json.loads(
                (home / SLOT.lease_dir / "lease.json").read_text(encoding="utf-8")
            )
            assert on_disk["holder"] in winners
        assert not (home / f"{SLOT.lease_dir}.break").exists()
