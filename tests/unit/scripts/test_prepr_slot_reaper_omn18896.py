# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Abandoned slots: real leases/processes, with only shared-service teardown faked."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "scripts/runtime_build/prepr_teardown_slot.py"
pytestmark = pytest.mark.unit


@pytest.fixture
def reaper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(HELPER.parent))
    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    spec = importlib.util.spec_from_file_location("slot_reaper_test", HELPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plant(
    reaper: Any, root: Path, *, expired: bool = True, alive: bool = False
) -> dict[str, Any]:
    record = reaper.claim(
        root, 1, "proof-run", "lane", str(reaper.STAGING_PARENT / "slot-1"), ttl=300
    )
    record["expires"] = time.time() + (-1 if expired else 300)
    if not alive:
        record["owner_pid"] = 99999999
        record["owner_start"] = "gone"
    reaper.write_record(root, 1, record)
    return record


def test_expired_dead_owner_is_torn_down_and_only_clean_releases(
    reaper: Any, tmp_path: Path
) -> None:
    plant(reaper, tmp_path)
    calls: list[tuple[int, str]] = []

    def teardown(slot: int, staging: str) -> bool:
        calls.append((slot, staging))
        return True

    result = reaper.reap_slot(tmp_path, 1, teardown=teardown)
    assert result["verdict"] == "REAPED"
    assert calls == [(1, str(reaper.STAGING_PARENT / "slot-1"))]
    assert not reaper.record_path(tmp_path, 1).exists()
    assert reaper.reap_slot(tmp_path, 1, teardown=teardown)["verdict"] == "ABSENT"
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("expired", "alive"), [(False, False), (False, True), (True, True)]
)
def test_fresh_lease_or_live_owner_is_never_reaped(
    reaper: Any, tmp_path: Path, expired: bool, alive: bool
) -> None:
    plant(reaper, tmp_path, expired=expired, alive=alive)
    result = reaper.reap_slot(
        tmp_path, 1, teardown=lambda *_: pytest.fail("live slot deleted")
    )
    assert result["verdict"] == "HELD"
    assert reaper.record_path(tmp_path, 1).exists()


def test_failed_teardown_keeps_lease_for_retry(reaper: Any, tmp_path: Path) -> None:
    old = plant(reaper, tmp_path)
    assert (
        reaper.reap_slot(tmp_path, 1, teardown=lambda *_: False)["verdict"]
        == "INCOMPLETE"
    )
    assert reaper.read_record(tmp_path, 1) == old
    assert (
        reaper.reap_slot(tmp_path, 1, teardown=lambda *_: True)["verdict"] == "REAPED"
    )


@pytest.mark.parametrize(
    "text", ["{", "{}", "[]", '{"expires": 0}', '{"schema": "other"}']
)
def test_unknown_lease_is_reported_and_never_guessed(
    reaper: Any, tmp_path: Path, text: str
) -> None:
    reaper.record_path(tmp_path, 1).write_text(text)
    assert (
        reaper.reap_slot(
            tmp_path, 1, teardown=lambda *_: pytest.fail("unknown slot deleted")
        )["verdict"]
        == "UNKNOWN"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("owner_pid", 2**100), ("owner_start", ""), ("child_pid", 2**100)],
)
def test_corrupt_process_identity_is_unknown(
    reaper: Any, tmp_path: Path, field: str, value: Any
) -> None:
    record = plant(reaper, tmp_path)
    record.update(child_pid=99999999, child_start="gone")
    record[field] = value
    reaper.write_record(tmp_path, 1, record)
    assert (
        reaper.reap_slot(
            tmp_path, 1, teardown=lambda *_: pytest.fail("corrupt identity deleted")
        )["verdict"]
        == "UNKNOWN"
    )


def test_reaper_plan_is_read_only(reaper: Any, tmp_path: Path) -> None:
    old = plant(reaper, tmp_path)
    assert (
        reaper.reap_slot(
            tmp_path, 1, plan_only=True, teardown=lambda *_: pytest.fail("plan mutated")
        )["verdict"]
        == "WOULD_REAP"
    )
    assert reaper.read_record(tmp_path, 1) == old


def test_a_new_claim_cannot_overwrite_even_an_expired_abandoned_slot(
    reaper: Any, tmp_path: Path
) -> None:
    old = plant(reaper, tmp_path)
    with pytest.raises(reaper.LeaseRefusedError):
        reaper.claim(tmp_path, 1, "peer", "peer", str(reaper.STAGING_PARENT / "slot-1"))
    assert reaper.read_record(tmp_path, 1) == old


def test_teardown_holds_the_lease_guard_against_a_new_claim(
    reaper: Any, tmp_path: Path
) -> None:
    plant(reaper, tmp_path)
    script = """import sys
sys.path.insert(0, sys.argv[1])
import prepr_teardown_slot as r
from pathlib import Path
try:
    r.claim(Path(sys.argv[2]), 1, 'peer', 'peer', str(r.STAGING_PARENT / 'slot-1'))
except r.LeaseRefusedError:
    sys.exit(8)
"""

    def teardown(*_: Any) -> bool:
        proc = subprocess.run(
            [sys.executable, "-c", script, str(HELPER.parent), str(tmp_path)],
            timeout=10,
            check=False,
        )
        assert proc.returncode == 8
        return True

    assert reaper.reap_slot(tmp_path, 1, teardown=teardown)["verdict"] == "REAPED"


def test_reaper_respects_the_bringup_lane_lock(
    reaper: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl

    monkeypatch.setenv("ONEX_LANE_LOCK_DIR", str(tmp_path / "lane-locks"))
    plant(reaper, tmp_path)
    path = reaper.lane_lock_path(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert (
            reaper.reap_slot(
                tmp_path, 1, teardown=lambda *_: pytest.fail("locked slot deleted")
            )["verdict"]
            == "BUSY"
        )


def test_heartbeat_renews_only_its_own_token(reaper: Any, tmp_path: Path) -> None:
    old = reaper.claim(
        tmp_path, 1, "proof-run", "lane", str(reaper.STAGING_PARENT / "slot-1"), ttl=1
    )
    reaper.heartbeat(tmp_path, 1, "proof-run", ttl=300)
    assert reaper.read_record(tmp_path, 1)["expires"] > old["expires"]
    with pytest.raises(reaper.LeaseRefusedError):
        reaper.heartbeat(tmp_path, 1, "peer")


@pytest.mark.parametrize("slot", [0, 3, -1, "dev", "judge"])
def test_only_the_closed_slot_policy_is_accepted(
    reaper: Any, tmp_path: Path, slot: Any
) -> None:
    with pytest.raises((reaper.LeaseRefusedError, ValueError)):
        reaper.claim(
            tmp_path, slot, "run", "lane", str(reaper.STAGING_PARENT / "slot-1")
        )


def test_real_command_heartbeat_and_exit_status(reaper: Any, tmp_path: Path) -> None:
    env = {**os.environ, "ONEX_PREPR_LEASE_DIR": str(tmp_path)}
    proc = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "run",
            "--slot",
            "1",
            "--holder",
            "proof",
            "--token",
            "proof-run",
            "--interval",
            "0.05",
            "--ttl",
            "1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(.2); raise SystemExit(7)",
        ],
        env=env,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 7
    record = reaper.read_record(tmp_path, 1)
    assert record["heartbeat"] > record["created"]
    assert record["holder"] == "proof"
    assert (
        reaper.reap_slot(
            tmp_path, 1, teardown=lambda *_: True, now=record["expires"] + 1
        )["verdict"]
        == "REAPED"
    )


def test_sanctioned_teardown_requires_clean_report(
    reaper: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []

    def run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        report = Path(argv[argv.index("--report-out") + 1])
        report.write_text(json.dumps({"verdict": "INCOMPLETE"}))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(reaper.subprocess, "run", run)
    assert (
        reaper.teardown_slot(1, str(reaper.STAGING_PARENT / "slot-1"), tmp_path)
        is False
    )
    argv, kwargs = calls[0]
    assert Path(argv[1]).name == "prepr_teardown_slot.sh"
    assert argv[argv.index("--slot") + 1] == "1"
    assert "--reason" in argv and "--lock-timeout" in argv
    assert "ONEX_PREPR_REAPER_HELD" in kwargs["env"]


def test_standalone_bringup_checks_an_inherited_lease_before_mutating(
    tmp_path: Path,
) -> None:
    text = (HELPER.parent / "prepr_verify_lane.sh").read_text()
    assert "prepr_teardown_slot.py" in text
    assert text.index('"${LEASE_PY}" check') < text.index('rm -rf "${STAGING_ROOT}"')
    assert '"${LEASE_PY}" run' in text


def test_clean_teardown_clears_the_resource_lease_only_after_readback() -> None:
    text = (HELPER.parent / "prepr_teardown_slot.sh").read_text()
    assert '"${LEASE_PY}" clear' in text
    assert text.index('"${PY}" "${TEARDOWN_PY}"') < text.index('"${LEASE_PY}" clear')
    assert "ONEX_PREPR_REAPER_HELD" in text


def test_pool_copies_and_wraps_each_slot_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    from test_prepr_runtime_pool_omn18893 import (
        CFG,
        FakeHost,
        FakeTransport,
        _params,
        pool,
    )

    class RecordingTransport(FakeTransport):
        def __init__(self) -> None:
            super().__init__({"pool-c-slot-1": FakeHost()})
            self.runs: list[str] = []
            self.copies: list[str] = []

        def run(self, host: Any, command: str, timeout: float) -> tuple[int, str]:
            self.runs.append(command)
            return super().run(host, command, timeout)

        def put(self, host: Any, local: Path, remote: str) -> int:
            self.copies.append(str(local))
            return super().put(host, local, remote)

    transport = RecordingTransport()
    host = CFG.host("pool-c-slot-1")
    pool._run_stack(
        CFG,
        transport,
        host,
        {},
        _params(tmp_path),
        "pool-run",
        "lane",
        True,
        lambda _: None,
    )
    assert str(HELPER) in transport.copies
    phases = [
        cmd
        for cmd in transport.runs
        if "/prove.sh " in cmd and not cmd.endswith(" snap-post")
    ]
    assert phases
    assert all(
        "slot-lease.py run --slot 1" in cmd and "--token pool-run" in cmd
        for cmd in phases
    )


def test_installer_snapshots_dependencies_and_ships_a_working_hosted_runner_probe(
    reaper: Any,
    tmp_path: Path,
) -> None:
    prefix, units, leases = (
        tmp_path / "installed",
        tmp_path / "units",
        tmp_path / "leases",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "install-reaper",
            "--prefix",
            str(prefix),
            "--unit-dir",
            str(units),
            "--no-enable",
        ],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout)
    assert reaper.install_reaper(prefix, units, enable=False) == info
    installed = Path(info["helper"])
    assert installed.is_file()
    for name in (
        "prepr_teardown_slot.sh",
        "prepr_teardown_slot.py",
        "prepr_slot_policy.py",
        "lane_lock.py",
        "lane_lock.sh",
    ):
        assert (installed.parent / name).is_file()
    assert (installed.parents[2] / "scripts/provision_db_slot.sh").is_file()
    assert (installed.parents[2] / "scripts/smoke/smoke_delegation.sh").is_file()
    assert installed.read_bytes() == HELPER.read_bytes()
    service = (units / "onex-prepr-slot-reaper.service").read_text()
    timer = (units / "onex-prepr-slot-reaper.timer").read_text()
    assert str(installed) in service and " reap" in service
    assert "Type=oneshot" in service and "TimeoutStartSec=1300" in service
    assert "OnUnitActiveSec=60s" in timer and "WantedBy=timers.target" in timer
    probe = subprocess.run(
        [sys.executable, str(installed), "reap", "--plan-only"],
        env={**os.environ, "ONEX_PREPR_LEASE_DIR": str(leases)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    assert [r["verdict"] for r in json.loads(probe.stdout)["slots"]] == [
        "ABSENT",
        "ABSENT",
    ]


def test_an_orphan_teardown_child_keeps_the_lane_locked(
    reaper: Any, tmp_path: Path
) -> None:
    path = tmp_path / "orphan.lock"
    child = None
    try:
        with reaper.locked(path) as fd:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(.3)"],
                pass_fds=(fd,),
                start_new_session=True,
            )
        with pytest.raises(reaper.LeaseRefusedError):
            with reaper.locked(path):
                pytest.fail("orphan teardown lost its lock")
        child.wait(timeout=10)
        with reaper.locked(path):
            pass
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=10)


def test_manual_teardown_refuses_before_touching_a_live_run(
    reaper: Any, tmp_path: Path
) -> None:
    plant(reaper, tmp_path, alive=True)
    with pytest.raises(reaper.LeaseRefusedError):
        reaper.authorize_teardown(tmp_path, 1, "")
    with pytest.raises(reaper.LeaseRefusedError):
        reaper.authorize_teardown(tmp_path, 1, "peer")
    reaper.authorize_teardown(tmp_path, 1, "proof-run")


def test_reused_pid_is_not_treated_as_the_dead_owner(
    reaper: Any, tmp_path: Path
) -> None:
    record = plant(reaper, tmp_path, alive=True)
    record["owner_start"] = "another-boot:another-start"
    reaper.write_record(tmp_path, 1, record)
    assert (
        reaper.reap_slot(tmp_path, 1, teardown=lambda *_: True)["verdict"] == "REAPED"
    )


def test_live_child_of_killed_heartbeat_owner_is_protected(
    reaper: Any, tmp_path: Path
) -> None:
    record = plant(reaper, tmp_path)
    record.update(child_pid=os.getpid(), child_start=reaper.process_start(os.getpid()))
    reaper.write_record(tmp_path, 1, record)
    assert (
        reaper.reap_slot(
            tmp_path, 1, teardown=lambda *_: pytest.fail("live orphan deleted")
        )["verdict"]
        == "HELD"
    )


def test_killed_command_owner_leaves_a_reapable_lease(
    reaper: Any, tmp_path: Path
) -> None:
    env = {**os.environ, "ONEX_PREPR_LEASE_DIR": str(tmp_path)}
    child_started = tmp_path / "child-started"
    command = "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(60)"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(HELPER),
            "run",
            "--slot",
            "1",
            "--holder",
            "crash-proof",
            "--interval",
            "0.05",
            "--ttl",
            "1",
            "--",
            sys.executable,
            "-c",
            command,
            str(child_started),
        ],
        env=env,
        start_new_session=True,
    )
    child_pid = None
    try:
        deadline = time.monotonic() + 10
        while not child_started.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_started.exists()
        record = reaper.read_record(tmp_path, 1)
        child_pid = record["child_pid"]
        proc.kill()
        proc.wait(timeout=10)
        assert (
            reaper.reap_slot(
                tmp_path,
                1,
                teardown=lambda *_: pytest.fail("live child deleted"),
                now=record["expires"] + 1,
            )["verdict"]
            == "HELD"
        )
        os.killpg(child_pid, 9)
        deadline = time.monotonic() + 10
        while (
            reaper.process_alive(child_pid, record["child_start"])
            and time.monotonic() < deadline
        ):
            time.sleep(0.02)
        assert (
            reaper.reap_slot(
                tmp_path, 1, teardown=lambda *_: True, now=record["expires"] + 1
            )["verdict"]
            == "REAPED"
        )
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
            proc.wait(timeout=10)
        if child_pid is not None:
            try:
                os.killpg(child_pid, 9)
            except ProcessLookupError:
                pass


def test_reap_report_persists_unknown_lease_failure(tmp_path: Path) -> None:
    (tmp_path / "slot-1.json").write_text("broken")
    proc = subprocess.run(
        [sys.executable, str(HELPER), "reap"],
        env={**os.environ, "ONEX_PREPR_LEASE_DIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert proc.returncode == 1
    report = json.loads((tmp_path / "last-reap.json").read_text())
    assert report["slots"][0]["verdict"] == "UNKNOWN"
    assert report["slots"][1]["verdict"] == "ABSENT"


def test_installer_propagates_systemd_enable_failure(tmp_path: Path) -> None:
    tools = tmp_path / "bin"
    tools.mkdir()
    systemctl = tools / "systemctl"
    systemctl.write_text("#!/bin/sh\nexit 17\n")
    systemctl.chmod(0o755)
    proc = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "install-reaper",
            "--prefix",
            str(tmp_path / "installed"),
            "--unit-dir",
            str(tmp_path / "units"),
        ],
        env={**os.environ, "PATH": f"{tools}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert proc.returncode != 0
    assert '"enabled": true' not in proc.stdout


def test_run_command_starts_only_after_its_pid_is_in_the_lease(
    reaper: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONEX_PREPR_LEASE_DIR", str(tmp_path))
    monkeypatch.setattr(reaper, "STAGING_PARENT", tmp_path.resolve() / "staging")
    real_start = reaper.process_start

    def slow_start(pid: int) -> str:
        if pid != os.getpid():
            time.sleep(0.5)
        return str(real_start(pid))

    monkeypatch.setattr(reaper, "process_start", slow_start)
    check = (
        "import json, os, sys; "
        "r = json.load(open(sys.argv[1])); "
        "sys.exit(0 if r.get('child_pid') == os.getpid() else 9)"
    )
    args = reaper.lease_main.__globals__["argparse"].Namespace(
        slot=1,
        holder="gate-proof",
        token="gate-run",
        staging_root=None,
        ttl=30.0,
        interval=0.05,
        command=["--", sys.executable, "-c", check, str(tmp_path / "slot-1.json")],
    )
    assert reaper.run_command(args) == 0
