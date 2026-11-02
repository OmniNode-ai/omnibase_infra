# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21913: exercise the real fcntl helper used by the verifier-install hook.

The helper's CLI, directory creation, lock ownership, exit status and diagnostics
are exercised as subprocesses. Install handshakes replace scheduling assumptions:
a recording install stays parked until the test explicitly releases it.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.ci import check_deploy_scope_dod as hook

pytestmark = pytest.mark.unit

_TAG = "dev-20261101000200-aaaaaaaaaaaa"
# Each child announces its start and waits for an explicit release from the test.
# The localhost socket is a test handshake and contacts no external service.
_RECORDING_PROGRAM = r"""
import os, socket, sys
from pathlib import Path
with socket.create_connection(("127.0.0.1", int(os.environ["INSTALL_PORT"])), timeout=60) as peer:
    with open(os.environ["INSTALL_LOG"], "a") as log:
        log.write("start\n")
    peer.sendall(b"start\n")
    assert peer.recv(1) == b"x", "test did not release the install"
    entry = Path(sys.argv[1])
    entry.mkdir(parents=True, exist_ok=True)
    (entry / "installed").touch()
    with open(os.environ["INSTALL_LOG"], "a") as log:
        log.write("end\n")
    peer.sendall(b"end\n")
"""
_RECORDING_INSTALL = f'exec "$INSTALL_PYTHON" -c {shlex.quote(_RECORDING_PROGRAM)} "$2"'


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("INSTALL_LOG", str(tmp_path / "install.log"))
    monkeypatch.setattr(hook, "_VERIFIER_INSTALL_SCRIPT", _RECORDING_INSTALL)
    return tmp_path / "cache" / "onex-dod-verifier"


def _events(tmp_path: Path) -> list[str]:
    return (tmp_path / "install.log").read_text(encoding="utf-8").splitlines()


@dataclass
class InstallControl:
    listener: socket.socket
    pool: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(2))
    waiting: threading.Event = field(default_factory=threading.Event)
    children: list[subprocess.Popen[str]] = field(default_factory=list)
    peers: list[socket.socket] = field(default_factory=list)

    def run(
        self,
        argv: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        assert not check and capture_output and text
        if hook._HEAVY_LOCK in argv:
            # Only increase diagnostic frequency; the real helper still acquires
            # and supervises the command with the production lock and hold cap.
            delimiter = argv.index("--")
            argv = [*argv[:delimiter], "--notice-every=0", *argv[delimiter:]]
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        self.children.append(proc)
        assert proc.stderr is not None and proc.stdout is not None
        stderr = []
        with proc.stderr:
            for line in proc.stderr:
                stderr.append(line)
                if line.startswith("heavy_lock: waiting for "):
                    self.waiting.set()
        with proc.stdout:
            stdout = proc.stdout.read()
        return subprocess.CompletedProcess(
            argv, proc.wait(timeout=60), stdout, "".join(stderr)
        )

    def started(self) -> socket.socket:
        peer, _ = self.listener.accept()
        self.peers.append(peer)
        peer.settimeout(60)
        with peer.makefile("rb") as reader:
            assert reader.readline() == b"start\n"
        return peer

    @staticmethod
    def finish(peer: socket.socket) -> None:
        peer.sendall(b"x")
        with peer.makefile("rb") as reader:
            assert reader.readline() == b"end\n"


@pytest.fixture
def control(cache: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[InstallControl]:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        listener.settimeout(60)
        monkeypatch.setenv("INSTALL_PORT", str(listener.getsockname()[1]))
        monkeypatch.setenv("INSTALL_PYTHON", sys.executable)
        control = InstallControl(listener)
        monkeypatch.setattr(hook, "subprocess", SimpleNamespace(run=control.run))
        try:
            yield control
        finally:
            for peer in control.peers:
                peer.close()
            for proc in control.children:
                if proc.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=60)
            control.pool.shutdown(wait=True, cancel_futures=True)


def test_the_install_script_does_not_call_the_flock_binary() -> None:
    pattern = r"\bflock\b"
    assert re.search(pattern, hook._VERIFIER_INSTALL_SCRIPT) is None
    # The old command is an executable shell snippet, rather than a source comment.
    assert re.search(
        pattern, 'set -euo pipefail; flock 9; [[ -f "$2/installed" ]] && exit 0'
    )


def test_two_installs_of_one_tag_run_one_after_the_other(
    control: InstallControl,
    cache: Path,
    tmp_path: Path,
) -> None:
    first = control.pool.submit(hook.install_verifier, _TAG)
    first_peer = control.started()
    # The install is running: prove ownership of the exact kernel lock path.
    with (cache / f"{_TAG}.lock").open("r+") as probe:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    second = control.pool.submit(hook.install_verifier, _TAG)
    assert control.waiting.wait(timeout=60), "second helper did not report contention"
    assert _events(tmp_path) == ["start"]
    assert not first.done() and not second.done()
    control.finish(first_peer)
    second_peer = control.started()
    control.finish(second_peer)
    assert first.result(timeout=60) == second.result(timeout=60)
    assert _events(tmp_path) == ["start", "end", "start", "end"]


def test_unserialized_runs_of_the_script_do_interleave(
    control: InstallControl,
    tmp_path: Path,
) -> None:
    argv = ["bash", "-c", _RECORDING_INSTALL, "install", _TAG, str(tmp_path / "entry")]
    first = control.pool.submit(
        control.run,
        argv,
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    first_peer = control.started()
    # Deliberately start the second child only AFTER the first is parked. There
    # is no deadline by which the second must be scheduled for overlap to occur.
    second = control.pool.submit(
        control.run,
        argv,
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    second_peer = control.started()
    assert _events(tmp_path) == ["start", "start"]
    control.finish(first_peer)
    control.finish(second_peer)
    assert (
        first.result(timeout=60).returncode == second.result(timeout=60).returncode == 0
    )
    assert _events(tmp_path) == ["start", "start", "end", "end"]


def test_the_install_waits_while_a_peer_holds_the_lock_path(
    control: InstallControl,
    cache: Path,
    tmp_path: Path,
) -> None:
    cache.mkdir(parents=True)
    with (cache / f"{_TAG}.lock").open("w") as peer_lock:
        fcntl.flock(peer_lock, fcntl.LOCK_EX)
        install = control.pool.submit(hook.install_verifier, _TAG)
        assert control.waiting.wait(timeout=60), "helper did not reach lock contention"
        assert not install.done()
        assert not (tmp_path / "install.log").exists()
    peer = control.started()
    control.finish(peer)
    install.result(timeout=60)
    assert _events(tmp_path) == ["start", "end"]


@pytest.fixture(params=["not-directory", "unwritable-directory"])
def unusable_cache(cache: Path, request: pytest.FixtureRequest) -> Iterator[Path]:
    cache.parent.mkdir(parents=True)
    if request.param == "not-directory":
        cache.write_text("not a directory", encoding="utf-8")
    else:
        cache.mkdir()
        cache.chmod(0o500)
    try:
        yield cache
    finally:
        if cache.is_dir():
            cache.chmod(0o700)


def test_a_lock_that_cannot_be_taken_fails_the_install_closed(
    unusable_cache: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        hook.DeployScopeHookError, match=r"rc=75.*FAILED CLOSED"
    ) as failure:
        hook.install_verifier(_TAG)
    if unusable_cache.is_dir():
        assert "Permission denied" in str(failure.value)
    assert not (tmp_path / "install.log").exists(), "the script ran without the lock"


def test_the_hook_exits_nonzero_with_a_message_when_the_lock_cannot_be_taken(
    unusable_cache: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(Path(hook.__file__).resolve().parents[2])
    monkeypatch.setattr(
        hook,
        "load_canonical_validator",
        lambda _home: SimpleNamespace(find_runtime_paths=lambda _files: []),
    )
    monkeypatch.setattr(hook, "read_head_contract", lambda _repo, _ticket: "x: y\n")
    monkeypatch.setattr(hook, "read_verifier_tag", lambda _repo: _TAG)
    code = hook.main(
        [
            "--omni-home",
            str(tmp_path),
            "--changed-files",
            "docs/note.md",
            "--pr-body",
            "none",
            "--contracts-dir",
            str(tmp_path),
            "--subjects",
            "OMN-21913 fix the lock",
        ]
    )
    assert code != 0
    stderr = capsys.readouterr().err
    assert "rc=75" in stderr and "FAILED CLOSED" in stderr
    if unusable_cache.is_dir():
        assert "Permission denied" in stderr
    assert not (tmp_path / "install.log").exists()


def test_the_lock_helper_passes_through_the_install_exit_code_and_stderr(
    cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hook, "_VERIFIER_INSTALL_SCRIPT", "echo install-refused >&2; exit 42"
    )
    with pytest.raises(hook.DeployScopeHookError, match=r"rc=42.*install-refused"):
        hook.install_verifier(_TAG)
    assert (cache / f"{_TAG}.lock").is_file(), "helper creates the cache parent"
