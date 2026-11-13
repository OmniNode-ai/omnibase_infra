#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dispatch-venv staging lock and floor promotion engine (OMN-16932).

The shell reconciler (scripts/reconcile-workspace-venvs.sh) may not grow, so its
staging-lock and floor-promotion logic lives here as a stdlib-only engine module
that runs under a bare python3. It is the node-side successor of the shell
functions ``staging_lock_field``, ``acquire_staging_lock``,
``release_staging_lock`` and ``promote_floor_with_venv``.

Stdlib only, no package imports: it must run on a host outside any project
venv, from the host's bare ``python3``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from datetime import UTC, datetime, timezone
from pathlib import Path

EXIT_OK = 0
EXIT_FAILED = 2
EXIT_DECLINED = 4


def _say(message: str) -> None:
    sys.stdout.write(f"[reconcile] {message}\n")


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record(pid: int, host: str, at: str) -> str:
    return f"pid={pid};host={host};at={at}"


def _parse_record(text: str) -> dict[str, str]:
    """Parse a lock record by splitting on ';'.

    A record may carry a large extra field (up to ~200 KiB); the split is
    linear and never uses a regex that scales badly with that field.
    """
    fields: dict[str, str] = {}
    for part in text.split(";"):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        fields[key] = value
    return fields


def _holder_is_live(fields: dict[str, str], host: str) -> bool:
    """A holder is LIVE when it has a pid and (its host differs from this host's
    name, or os.kill(pid, 0) succeeds or raises PermissionError)."""
    pid_text = fields.get("pid", "")
    if not pid_text:
        return False
    try:
        pid = int(pid_text)
    except ValueError:
        return False
    holder_host = fields.get("host", "")
    if holder_host and holder_host != host:
        return True
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _staging_display(path: Path) -> str:
    text = str(path)
    if text.endswith(".lock"):
        return text[: -len(".lock")]
    return text


def staging_lock_acquire(path: Path, pid: int, host: str | None = None) -> int:
    host = host if host is not None else socket.gethostname()
    record = _record(pid, host, _utc_now())
    path = Path(path)
    for _attempt in range(2):
        try:
            path.symlink_to(record)
            return EXIT_OK
        except FileExistsError:
            try:
                held = path.readlink().as_posix()
            except OSError:
                held = None
            if held is not None:
                fields = _parse_record(held)
                if _holder_is_live(fields, host):
                    _say(
                        "DECLINED: another run of this script "
                        f"(pid {fields.get('pid', '?')} on {fields.get('host', '?')}, "
                        f"since {fields.get('at', '?')}) is staging {_staging_display(path)};"
                    )
                    _say(
                        "  two runs on one staging directory delete each other's venv."
                    )
                    _say(
                        "  this run reconciled NOTHING and left the staging path and the lock untouched."
                    )
                    return EXIT_DECLINED
                # Dead or unreadable holder: reclaim only if the text still
                # equals the record just read.
                try:
                    current = path.readlink().as_posix()
                except OSError:
                    current = None
                if current == held:
                    try:
                        path.unlink()
                    except OSError:
                        pass
            # retry the claim
            continue
    _say(f"FAILED: could not take the staging lock {path}")
    return EXIT_FAILED


def staging_lock_release(path: Path, pid: int) -> int:
    path = Path(path)
    try:
        held = path.readlink().as_posix()
    except OSError:
        # Never raise on a missing link.
        return EXIT_OK
    fields = _parse_record(held)
    if fields.get("pid", "") == str(pid):
        try:
            path.unlink()
        except OSError:
            pass
    return EXIT_OK


def staging_lock_held(path: Path, host: str | None = None) -> int:
    """READ ONLY. EXIT_OK when a live holder holds it, else 1."""
    host = host if host is not None else socket.gethostname()
    path = Path(path)
    try:
        held = path.readlink().as_posix()
    except OSError:
        return 1
    fields = _parse_record(held)
    if _holder_is_live(fields, host):
        return EXIT_OK
    return 1


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_verify_movement() -> Path:
    return _repo_root() / "scripts" / "reconcile_verify_movement.py"


def _first_site_packages(venv: Path) -> Path | None:
    lib = Path(venv) / "lib"
    if not lib.is_dir():
        return None
    for child in sorted(lib.iterdir()):
        candidate = child / "site-packages"
        if candidate.is_dir():
            return candidate
    return None


def promote_floor(
    venv: Path,
    uv_lock: Path,
    workspace_root: Path,
    floor: Path,
    *,
    changed: bool,
) -> int:
    floor = Path(floor)
    if not floor.exists():
        _say(
            "dispatch venv: no floor to advance; the first floor is stamped by the host reconcile"
        )
        return EXIT_OK

    verify_path = _load_verify_movement()
    spec = importlib.util.spec_from_file_location(
        "reconcile_verify_movement", verify_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    del sys.modules[spec.name]

    site = _first_site_packages(venv)
    installed = module.observe_installed_commit(site, "omnimarket") if site else None

    floor_commit: str | None = None
    try:
        data = json.loads(floor.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            floor_commit = data.get("omnimarket_commit")
    except (OSError, json.JSONDecodeError):
        floor_commit = None

    if not changed and installed is not None and installed == floor_commit:
        return EXIT_OK

    if site is None:
        if changed:
            _say(
                "FAILED: the dispatch venv moved but the floor could not be advanced with it; "
                "the floor was left at its previous value."
            )
            return EXIT_FAILED
        _say(
            "WARNING: the floor could not be brought level with an unchanged venv; "
            "it is left as it was."
        )
        return EXIT_OK

    proc = subprocess.run(
        [
            sys.executable,
            str(verify_path),
            "floor-from-venv",
            "--site-packages",
            str(site),
            "--lock",
            str(uv_lock),
            "--omni-home",
            str(workspace_root),
            "--output",
            str(floor),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        if changed:
            _say(
                "FAILED: the dispatch venv moved but the floor could not be advanced with it; "
                "the floor was left at its previous value."
            )
            return EXIT_FAILED
        _say(
            "WARNING: the floor could not be brought level with an unchanged venv; "
            "it is left as it was."
        )
        return EXIT_OK

    short = installed[:12] if installed else "?"
    _say(f"dispatch venv: floor advanced with the venv (omnimarket {short})")
    return EXIT_OK


def run_promotion_command(
    command: str,
    *,
    path=None,
    pid=None,
    venv=None,
    uv_lock=None,
    workspace_root=None,
    floor=None,
    changed=False,
) -> int:
    if command == "staging_lock_acquire":
        if path is None or pid is None:
            sys.stderr.write("staging_lock_acquire requires path and pid" + "\n")
            return EXIT_FAILED
        return staging_lock_acquire(Path(path), int(pid))
    if command == "staging_lock_release":
        if path is None or pid is None:
            sys.stderr.write("staging_lock_release requires path and pid" + "\n")
            return EXIT_FAILED
        return staging_lock_release(Path(path), int(pid))
    if command == "staging_lock_held":
        if path is None:
            sys.stderr.write("staging_lock_held requires path" + "\n")
            return EXIT_FAILED
        return staging_lock_held(Path(path))
    if command == "promote_floor":
        if venv is None or uv_lock is None or workspace_root is None or floor is None:
            sys.stderr.write(
                "promote_floor requires venv, uv_lock, workspace_root and floor" + "\n"
            )
            return EXIT_FAILED
        return promote_floor(
            Path(venv),
            Path(uv_lock),
            Path(workspace_root),
            Path(floor),
            changed=changed,
        )
    sys.stderr.write(f"unknown promotion command: {command}\n")
    return EXIT_FAILED
