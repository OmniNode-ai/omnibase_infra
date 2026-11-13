#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the dispatch-venv staging lock and floor promotion engine."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ENGINE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omnibase_infra"
    / "handlers"
    / "venv_atomic_install"
    / "dispatch_venv_promotion.py"
)


def _load_engine():
    spec = importlib.util.spec_from_file_location("dispatch_venv_promotion", _ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE = _load_engine()


def _write_record(
    path: Path, pid: int, host: str, at: str = "2026-01-01T00:00:00Z", extra: str = ""
) -> None:
    text = f"pid={pid};host={host};at={at}"
    if extra:
        text += f";extra={extra}"
    path.symlink_to(text)


def test_live_holder_declines(tmp_path, capsys):
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        lock = tmp_path / "staging.lock"
        _write_record(lock, proc.pid, socket.gethostname())
        rc = ENGINE.staging_lock_acquire(lock, 99999)
        out = capsys.readouterr().out
        assert rc == ENGINE.EXIT_DECLINED
        assert "DECLINED" in out
        assert f"pid {proc.pid}" in out
        # the link is unchanged: still names the live holder
        assert lock.readlink().as_posix().startswith(f"pid={proc.pid};")
    finally:
        proc.kill()
        proc.wait()


def test_holder_on_another_host_never_reclaimed(tmp_path, capsys):
    lock = tmp_path / "staging.lock"
    _write_record(lock, 424242, "some-other-host")
    rc = ENGINE.staging_lock_acquire(lock, 99999)
    out = capsys.readouterr().out
    assert rc == ENGINE.EXIT_DECLINED
    assert "DECLINED" in out
    assert lock.readlink().as_posix().startswith("pid=424242;")


def test_dead_holder_reclaimed(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-c", "pass"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    proc.wait()
    lock = tmp_path / "staging.lock"
    _write_record(lock, proc.pid, socket.gethostname())
    rc = ENGINE.staging_lock_acquire(lock, 99999)
    assert rc == ENGINE.EXIT_OK
    assert lock.readlink().as_posix().startswith("pid=99999;")


def test_release_removes_only_our_record(tmp_path):
    lock = tmp_path / "staging.lock"
    _write_record(lock, 777, socket.gethostname())
    rc = ENGINE.staging_lock_release(lock, 777)
    assert rc == ENGINE.EXIT_OK
    assert not lock.is_symlink()
    # release on a link naming a different pid leaves it alone
    _write_record(lock, 555, socket.gethostname())
    rc = ENGINE.staging_lock_release(lock, 777)
    assert rc == ENGINE.EXIT_OK
    assert lock.is_symlink()
    assert lock.readlink().as_posix().startswith("pid=555;")


def test_a_record_with_a_very_large_extra_field_is_parsed_in_one_pass() -> None:
    # A symlink target cannot exceed PATH_MAX, so the parser is exercised directly:
    # the record text goes through `_parse_record`, never as an argument or a regex.
    record = "pid=4242;host=lab-host;at=2026-01-01T00:00:00Z;note=" + "x" * (200 * 1024)
    fields = ENGINE._parse_record(record)
    assert (fields["pid"], fields["host"], fields["at"]) == (
        "4242",
        "lab-host",
        "2026-01-01T00:00:00Z",
    )
    assert len(fields["note"]) == 200 * 1024


def test_held_live_and_dead(tmp_path):
    lock = tmp_path / "staging.lock"
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _write_record(lock, proc.pid, socket.gethostname())
        assert ENGINE.staging_lock_held(lock) == ENGINE.EXIT_OK
    finally:
        proc.kill()
        proc.wait()
    # dead holder: held is False; the link is still there and must not be removed
    assert lock.is_symlink()
    assert ENGINE.staging_lock_held(lock) == 1
    assert lock.is_symlink()


def _make_venv(tmp_path, commit: str | None) -> Path:
    venv = tmp_path / "venv"
    site = venv / "lib" / "python3.13" / "site-packages"
    dist = site / "omnimarket-0.4.11.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: omnimarket\nVersion: 0.4.11\n", encoding="utf-8"
    )
    if commit is not None:
        (dist / "direct_url.json").write_text(
            json.dumps(
                {
                    "url": "git+https://example/omnimarket",
                    "vcs_info": {"commit_id": commit},
                }
            ),
            encoding="utf-8",
        )
    return venv


def test_promote_floor_no_floor(tmp_path, capsys):
    venv = _make_venv(tmp_path, "a" * 40)
    floor = tmp_path / "floor.json"
    rc = ENGINE.promote_floor(venv, tmp_path / "uv.lock", tmp_path, floor, changed=True)
    assert rc == ENGINE.EXIT_OK
    assert not floor.exists()
    assert "no floor to advance" in capsys.readouterr().out


def test_promote_floor_changed_advances(tmp_path, capsys):
    venv = _make_venv(tmp_path, "a" * 40)
    floor = tmp_path / "floor.json"
    floor.write_text(json.dumps({"omnimarket_commit": "b" * 40}), encoding="utf-8")
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("", encoding="utf-8")
    workspace_root = tmp_path / "workspace_root"
    workspace_root.mkdir()
    rc = ENGINE.promote_floor(venv, uv_lock, workspace_root, floor, changed=True)
    assert rc == ENGINE.EXIT_OK
    data = json.loads(floor.read_text(encoding="utf-8"))
    assert data["omnimarket_commit"] == "a" * 40


def test_promote_floor_unchanged_equal(tmp_path, capsys):
    venv = _make_venv(tmp_path, "a" * 40)
    floor = tmp_path / "floor.json"
    original = json.dumps({"omnimarket_commit": "a" * 40})
    floor.write_text(original, encoding="utf-8")
    rc = ENGINE.promote_floor(
        venv, tmp_path / "uv.lock", tmp_path, floor, changed=False
    )
    assert rc == ENGINE.EXIT_OK
    assert floor.read_text(encoding="utf-8") == original


def test_promote_floor_changed_no_dist_info(tmp_path, capsys):
    venv = tmp_path / "venv"
    venv.mkdir()
    floor = tmp_path / "floor.json"
    original = json.dumps({"omnimarket_commit": "b" * 40})
    floor.write_text(original, encoding="utf-8")
    rc = ENGINE.promote_floor(venv, tmp_path / "uv.lock", tmp_path, floor, changed=True)
    assert rc == ENGINE.EXIT_FAILED
    assert floor.read_text(encoding="utf-8") == original


def test_promote_floor_unchanged_no_dist_info(tmp_path, capsys):
    venv = tmp_path / "venv"
    venv.mkdir()
    floor = tmp_path / "floor.json"
    original = json.dumps({"omnimarket_commit": "b" * 40})
    floor.write_text(original, encoding="utf-8")
    rc = ENGINE.promote_floor(
        venv, tmp_path / "uv.lock", tmp_path, floor, changed=False
    )
    assert rc == ENGINE.EXIT_OK
    assert "WARNING" in capsys.readouterr().out


def test_engine_imports_nothing_newer_than_the_macos_system_python() -> None:
    """The wrapper and the reconciler run the engine with a bare `python3`, which a
    launchd PATH resolves to the macOS system 3.9: `datetime.UTC` (3.11) fails there."""
    import ast

    tree = ast.parse(_ENGINE.read_text(encoding="utf-8"), feature_version=(3, 9))
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "datetime"
        for alias in node.names
    }
    assert "UTC" not in names
