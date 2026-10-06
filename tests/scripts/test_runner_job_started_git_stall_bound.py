# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20598 -- the runner job-started hook bounds a silent git HTTP transfer.

On 2026-10-05 a depth-1 ``actions/checkout`` fetch on the self-hosted fleet
hung 10-20 minutes per attempt before ``curl 56 GnuTLS recv error``, so one
checkout ran up to 1831s. ``wire_git_transfer_stall_bound`` adds
``http.lowSpeedLimit``/``http.lowSpeedTime`` to the job's GIT_CONFIG_* set
through the shared accumulator, so a silent transfer aborts and checkout's own
retry opens a fresh connection.

These tests pin: the default values reach GITHUB_ENV, overrides are honoured,
an invalid override fails open with nothing written, the flushed settings
really abort a fetch from a server that goes silent mid-body, and the real
script reaches the call on its success path.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from tests.scripts.test_runner_job_started_mirror_rewrite import (
    HOOK_SCRIPT,
    _functions_only_script,
    _github_env_pairs,
    _has_gnu_realpath_m,
)

pytestmark = [pytest.mark.unit]

_DEFAULT_PAIRS = [("http.lowSpeedLimit", "1"), ("http.lowSpeedTime", "60")]


def _run_stall_bound(
    tmp_path: Path, overrides: dict[str, str]
) -> tuple[subprocess.CompletedProcess[str], Path]:
    github_env = tmp_path / "github_env"
    github_env.write_text("")
    driver = tmp_path / "driver.sh"
    driver.write_text(
        _functions_only_script()
        + "\nwire_git_transfer_stall_bound\n_c2_rewrite_flush\n"
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OMNI_GIT_LOW_SPEED_LIMIT", "OMNI_GIT_LOW_SPEED_TIME")
    }
    env.update({"GITHUB_ENV": str(github_env), **overrides})
    result = subprocess.run(
        ["bash", str(driver)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result, github_env


def test_default_stall_bound_is_flushed_to_github_env(tmp_path: Path) -> None:
    result, github_env = _run_stall_bound(tmp_path, {})
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(github_env)
    assert count == 2
    assert pairs == _DEFAULT_PAIRS
    assert "[git-stall-bound]" in result.stdout


def test_overrides_are_honoured(tmp_path: Path) -> None:
    result, github_env = _run_stall_bound(
        tmp_path,
        {"OMNI_GIT_LOW_SPEED_LIMIT": "500", "OMNI_GIT_LOW_SPEED_TIME": "30"},
    )
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(github_env)
    assert count == 2
    assert pairs == [("http.lowSpeedLimit", "500"), ("http.lowSpeedTime", "30")]


@pytest.mark.parametrize("var", ["OMNI_GIT_LOW_SPEED_LIMIT", "OMNI_GIT_LOW_SPEED_TIME"])
@pytest.mark.parametrize("bad", ["0", "abc", "-5", ""])
def test_invalid_override_fails_open_with_nothing_written(
    tmp_path: Path, var: str, bad: str
) -> None:
    result, github_env = _run_stall_bound(tmp_path, {var: bad})
    assert result.returncode == 0, result.stderr
    if bad == "":
        # An empty value is the shell default `${VAR:-...}`, not an override.
        assert _github_env_pairs(github_env) == (2, _DEFAULT_PAIRS)
        return
    assert github_env.read_text() == ""
    assert "invalid" in result.stdout


@pytest.fixture
def silent_mid_body_server() -> Iterator[int]:
    """An HTTP server that starts a git ref advertisement, then goes silent
    with the socket held open -- the shape of the 2026-10-05 hung fetches."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    port: int = srv.getsockname()[1]
    held: list[socket.socket] = []
    stop = threading.Event()

    def serve() -> None:
        srv.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            held.append(conn)
            try:
                conn.recv(4096)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/x-git-upload-pack-advertisement\r\n"
                    b"Content-Length: 100000\r\n\r\n"
                    b"001e# service=git-upload-pack\n" + b"0" * 2000
                )
            except OSError:
                pass

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        stop.set()
        thread.join(timeout=5)
        for conn in held:
            conn.close()
        srv.close()


def test_flushed_bound_aborts_a_silent_fetch(
    tmp_path: Path, silent_mid_body_server: int
) -> None:
    result, github_env = _run_stall_bound(tmp_path, {"OMNI_GIT_LOW_SPEED_TIME": "3"})
    assert result.returncode == 0, result.stderr
    count, pairs = _github_env_pairs(github_env)
    assert count == len(pairs) == 2
    # The flushed pairs are applied as `-c` options: the git-env scrub (OMN-18434)
    # strips every inherited GIT_CONFIG_* from a test subprocess, and `-c` is
    # the same configuration source the job's GIT_CONFIG_* set feeds.
    config_args = [arg for key, value in pairs for arg in ("-c", f"{key}={value}")]

    started = time.monotonic()
    fetch = subprocess.run(
        [
            "git",
            *config_args,
            "ls-remote",
            f"http://127.0.0.1:{silent_mid_body_server}/x.git",
        ],
        cwd=tmp_path,
        env=scrub_git_location_env({**os.environ, "GIT_TERMINAL_PROMPT": "0"}),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    elapsed = time.monotonic() - started
    assert fetch.returncode != 0
    assert elapsed < 20, f"silent fetch was not bounded: {elapsed:.1f}s"


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
    return port


@pytest.mark.skipif(
    not _has_gnu_realpath_m(),
    reason="full-script run needs GNU `realpath -m` (Ubuntu 22.04 runner image "
    "and Linux CI; absent on BSD/macOS) -- same gate as "
    "test_runner_job_started_root_owned_debris.py",
)
def test_full_script_success_path_flushes_stall_bound(tmp_path: Path) -> None:
    runner_home = tmp_path / "actions-runner"
    workspace = runner_home / "_work" / "omnibase_infra" / "omnibase_infra"
    workspace.mkdir(parents=True)
    github_env = tmp_path / "github_env"
    github_env.write_text("")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OMNI_GIT_LOW_SPEED_LIMIT", "OMNI_GIT_LOW_SPEED_TIME")
    }
    env.update(
        {
            "RUNNER_HOME": str(runner_home),
            "GITHUB_WORKSPACE": str(workspace),
            "GITHUB_ENV": str(github_env),
            "GITHUB_REPOSITORY": "OmniNode-ai/omnibase_infra",
            "OMNI_GIT_MIRROR_HOST": "127.0.0.1",
            "OMNI_GIT_MIRROR_PORT": str(_closed_port()),
            "OMNI_GIT_MIRROR_REWRITE_DISABLE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert _github_env_pairs(github_env) == (2, _DEFAULT_PAIRS)
