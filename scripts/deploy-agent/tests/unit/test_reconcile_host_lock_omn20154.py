# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Deploy builds share reconcile-host's holder protocol (OMN-20154)."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from deploy_agent.reconcile_host_lock import (
    RECONCILE_HOST_LOCK_DIRNAME,
    ReconcileHostLockTimeoutError,
    hold_reconcile_host_lock,
)

pytestmark = pytest.mark.unit


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def _holder(tmp_path: Path, *, pid: int, host: str, age: int = 0) -> Path:
    lock = tmp_path / RECONCILE_HOST_LOCK_DIRNAME
    lock.mkdir()
    holder = lock / "holder"
    holder.write_text(f"pid={pid}\nhost={host}\nstarted_at=2026-09-30T19:28:28Z\n")
    mtime = int(time.time()) - age
    os.utime(holder, (mtime, mtime))
    return lock


def _expect_timeout(tmp_path: Path) -> str:
    clock = _Clock()
    with pytest.raises(ReconcileHostLockTimeoutError) as raised:
        with hold_reconcile_host_lock(
            str(tmp_path),
            purpose="test build",
            wait_seconds=2,
            poll_seconds=1,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        ):
            pytest.fail("A respected holder must prevent acquisition")
    assert clock.now == 2
    message = str(raised.value)
    assert message.startswith("reconcile_host_lock_timeout:")
    assert str(tmp_path / RECONCILE_HOST_LOCK_DIRNAME) in message
    return message


def test_acquire_writes_exact_holder_and_releases(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock = tmp_path / RECONCILE_HOST_LOCK_DIRNAME
    with caplog.at_level("INFO"):
        with hold_reconcile_host_lock(
            str(tmp_path), purpose="image build abc123"
        ) as held:
            assert held is True
            record = (lock / "holder").read_text()
            assert re.fullmatch(
                rf"pid={os.getpid()}\nhost={re.escape(socket.gethostname())}\n"
                r"started_at=\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n"
                r"holder=deploy-agent\npurpose=image build abc123\n",
                record,
            )
    assert not lock.exists()
    assert "acquir" in caplog.text.lower()
    assert "releas" in caplog.text.lower()


@pytest.mark.parametrize("omni_home", ["", " \t\n"])
def test_blank_home_yields_false_and_creates_nothing(
    omni_home: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING"):
        with hold_reconcile_host_lock(omni_home, purpose="uncoordinated") as held:
            assert held is False
    assert list(tmp_path.iterdir()) == []
    assert "not coordinated with reconcile-host" in caplog.text


def test_live_local_holder_times_out_with_holder_identity(tmp_path: Path) -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock = _holder(tmp_path, pid=child.pid, host=socket.gethostname(), age=7200)
        message = _expect_timeout(tmp_path)
        assert str(child.pid) in message
        assert socket.gethostname() in message
        assert "2026-09-30T19:28:28Z" in message
        assert lock.exists()
    finally:
        child.kill()
        child.wait()


def test_dead_local_holder_is_reclaimed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    lock = _holder(tmp_path, pid=child.pid, host=socket.gethostname())
    with caplog.at_level("WARNING"):
        with hold_reconcile_host_lock(str(tmp_path), purpose="reclaim") as held:
            assert held is True
            assert f"pid={os.getpid()}\n" in (lock / "holder").read_text()
    assert not lock.exists()
    assert "not running" in caplog.text


def test_permission_denied_means_local_holder_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _holder(tmp_path, pid=os.getpid(), host=socket.gethostname(), age=7200)

    def denied(pid: int, signal: int) -> None:
        raise PermissionError("peer process")

    monkeypatch.setattr(os, "kill", denied)
    _expect_timeout(tmp_path)
    assert lock.exists()


def test_young_foreign_holder_is_respected(tmp_path: Path) -> None:
    lock = _holder(tmp_path, pid=os.getpid(), host="other-host", age=10)
    assert "other-host" in _expect_timeout(tmp_path)
    assert lock.exists()


def test_old_foreign_holder_is_reclaimed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    lock = _holder(tmp_path, pid=os.getpid(), host="other-host", age=7200)
    with caplog.at_level("WARNING"):
        with hold_reconcile_host_lock(str(tmp_path), purpose="foreign reclaim"):
            assert f"host={socket.gethostname()}\n" in (lock / "holder").read_text()
    assert not lock.exists()
    assert "other-host" in caplog.text


@pytest.mark.parametrize(("age", "reclaimed"), [(10, False), (7200, True)])
def test_missing_holder_uses_directory_age(
    tmp_path: Path, age: int, reclaimed: bool
) -> None:
    lock = tmp_path / RECONCILE_HOST_LOCK_DIRNAME
    lock.mkdir()
    mtime = int(time.time()) - age
    os.utime(lock, (mtime, mtime))
    if reclaimed:
        with hold_reconcile_host_lock(str(tmp_path), purpose="missing holder"):
            assert (lock / "holder").is_file()
        assert not lock.exists()
    else:
        _expect_timeout(tmp_path)
        assert lock.exists()


def test_unreadable_age_is_not_reclaimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = _holder(tmp_path, pid=os.getpid(), host="other-host", age=7200)
    original_stat = Path.stat

    def unreadable(path: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        if path in (lock, lock / "holder"):
            raise PermissionError("cannot inspect lock age")
        return original_stat(path, follow_symlinks=follow_symlinks)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", unreadable)
        _expect_timeout(tmp_path)
    assert lock.exists()


@pytest.mark.parametrize("replace_host", [False, True])
def test_release_preserves_replaced_holder(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, replace_host: bool
) -> None:
    lock = tmp_path / RECONCILE_HOST_LOCK_DIRNAME
    replacement = (
        f"pid={os.getpid() if replace_host else os.getpid() + 1}\n"
        f"host={'other-host' if replace_host else socket.gethostname()}\n"
    )
    with caplog.at_level("WARNING"):
        with hold_reconcile_host_lock(str(tmp_path), purpose="replaced"):
            (lock / "holder").write_text(replacement)
    assert lock.exists()
    assert (lock / "holder").read_text() == replacement
    assert caplog.records


def test_release_on_exception(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="broken build"):
        with hold_reconcile_host_lock(str(tmp_path), purpose="failing build"):
            raise RuntimeError("broken build")
    assert not (tmp_path / RECONCILE_HOST_LOCK_DIRNAME).exists()


def test_missing_home_directory_yields_false_and_creates_nothing(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent"
    with hold_reconcile_host_lock(str(missing), purpose="build") as held:
        assert held is False
    assert not missing.exists()


def test_holder_write_failure_removes_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.write_text

    def failing_write(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "holder":
            raise OSError("disk full")
        return original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", failing_write)
    with pytest.raises(OSError, match="disk full"):
        with hold_reconcile_host_lock(str(tmp_path), purpose="build"):
            pytest.fail("the body must not run without a holder record")
    assert not (tmp_path / RECONCILE_HOST_LOCK_DIRNAME).exists()
