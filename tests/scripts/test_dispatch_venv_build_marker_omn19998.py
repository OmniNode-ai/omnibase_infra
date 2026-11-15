#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the dispatch-venv build marker stamp (OMN-19998)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _make_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    venv.mkdir()
    return venv


def _make_lock(tmp_path: Path, text: str = "version = 1\n") -> Path:
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text(text, encoding="utf-8")
    return uv_lock


def _marker_lines(venv: Path) -> list[str]:
    marker = venv / ENGINE.BUILD_MARKER_NAME
    return marker.read_text(encoding="utf-8").splitlines()


def test_stamp_writes_marker_with_lock_path_hash_and_copy(tmp_path):
    venv = _make_venv(tmp_path)
    uv_lock = _make_lock(tmp_path, '[manifest]\nmembers = ["omnimarket"]\n')
    lock_bytes = uv_lock.read_bytes()

    rc = ENGINE.stamp_build_lock(venv, uv_lock)

    assert rc == ENGINE.EXIT_OK
    assert _marker_lines(venv) == [
        f"lock_path={uv_lock.resolve()}",
        f"lock_sha256={hashlib.sha256(lock_bytes).hexdigest()}",
        f"lock_copy={ENGINE.BUILD_LOCK_COPY_NAME}",
    ]
    copy = venv / ENGINE.BUILD_LOCK_COPY_NAME
    assert copy.is_file()
    assert copy.read_bytes() == lock_bytes


def test_stamp_unreadable_lock_fails_and_writes_nothing(tmp_path, capsys):
    venv = _make_venv(tmp_path)
    missing = tmp_path / "no-such-uv.lock"

    rc = ENGINE.stamp_build_lock(venv, missing)

    assert rc == ENGINE.EXIT_FAILED
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert str(missing) in out
    assert not (venv / ENGINE.BUILD_MARKER_NAME).exists()
    assert not (venv / ENGINE.BUILD_LOCK_COPY_NAME).exists()


def test_stamp_missing_venv_fails(tmp_path, capsys):
    uv_lock = _make_lock(tmp_path)
    venv = tmp_path / "no-such-venv"

    rc = ENGINE.stamp_build_lock(venv, uv_lock)

    assert rc == ENGINE.EXIT_FAILED
    assert "FAILED" in capsys.readouterr().out
    assert not venv.exists()


def test_promote_floor_command_stamps_marker(tmp_path, capsys):
    venv = _make_venv(tmp_path)
    uv_lock = _make_lock(tmp_path)
    floor = tmp_path / "floor.json"  # never created: promote_floor exits OK early

    rc = ENGINE.run_promotion_command(
        "promote_floor",
        venv=venv,
        uv_lock=uv_lock,
        workspace_root=tmp_path,
        floor=floor,
        changed=False,
    )

    assert rc == ENGINE.EXIT_OK
    assert "no floor to advance" in capsys.readouterr().out
    assert (venv / ENGINE.BUILD_MARKER_NAME).is_file()
    assert (venv / ENGINE.BUILD_LOCK_COPY_NAME).read_bytes() == uv_lock.read_bytes()


def test_promote_floor_failure_leaves_marker_unwritten(tmp_path, capsys):
    venv = tmp_path / "venv"  # exists but has no lib/site-packages
    venv.mkdir()
    uv_lock = _make_lock(tmp_path)
    floor = tmp_path / "floor.json"
    floor.write_text(json.dumps({"omnimarket_commit": "b" * 40}), encoding="utf-8")

    rc = ENGINE.run_promotion_command(
        "promote_floor",
        venv=venv,
        uv_lock=uv_lock,
        workspace_root=tmp_path,
        floor=floor,
        changed=True,
    )

    assert rc == ENGINE.EXIT_FAILED
    assert "FAILED" in capsys.readouterr().out
    assert not (venv / ENGINE.BUILD_MARKER_NAME).exists()
    assert not (venv / ENGINE.BUILD_LOCK_COPY_NAME).exists()


def test_restamp_after_lock_moves_updates_hash(tmp_path):
    venv = _make_venv(tmp_path)
    uv_lock = _make_lock(tmp_path, "old-lock-content\n")
    assert ENGINE.stamp_build_lock(venv, uv_lock) == ENGINE.EXIT_OK
    old_hash = hashlib.sha256(b"old-lock-content\n").hexdigest()
    assert f"lock_sha256={old_hash}" in _marker_lines(venv)

    new_text = "new-lock-content\n"
    uv_lock.write_text(new_text, encoding="utf-8")
    assert ENGINE.stamp_build_lock(venv, uv_lock) == ENGINE.EXIT_OK

    new_hash = hashlib.sha256(new_text.encode("utf-8")).hexdigest()
    assert _marker_lines(venv) == [
        f"lock_path={uv_lock.resolve()}",
        f"lock_sha256={new_hash}",
        f"lock_copy={ENGINE.BUILD_LOCK_COPY_NAME}",
    ]
    assert (venv / ENGINE.BUILD_LOCK_COPY_NAME).read_bytes() == new_text.encode("utf-8")
    # both stamps replaced their temp files into place; nothing is left behind
    assert sorted(p.name for p in venv.iterdir()) == sorted(
        [ENGINE.BUILD_MARKER_NAME, ENGINE.BUILD_LOCK_COPY_NAME]
    )
