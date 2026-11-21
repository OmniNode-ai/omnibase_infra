# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registered dev-lane branch installs: one store, a reason for every hold (OMN-26300).

The dev deploy agent kept registered branch installs in two stores and read one of them:
``state/jobs-dev/branch-installs`` (its own state directory) and the old
``state/jobs/branch-installs``, which it never read. An install registered in the old one
was invisible, so every recreate came from the unfixed build (devwindow lanes, 2026-11-20).

This module holds the agent-side rules that are not the handler's:

* ``migrate_legacy_store``: the old location is read once through this function, its
  records and content are merged into the agent's store, and the old directory is retired.
  A declared legacy location comes from the unit's environment
  (the agent reads ``DEPLOY_AGENT_LEGACY_BRANCH_INSTALL_STORE`` and passes it in); a unit that
  names none migrates nothing.
* ``held_installs``: every declared install that is not whole carries a reason (never an
  empty one), served by ``/branch-installs`` and written into ``status.json``.
* ``restore_yield``: whether the idle restore leaves the recreate window to a waiting install.

Standard library only: the deploy agent loads this by file path.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STORE_DIRNAME = "branch-installs"
RECORDS_FILE = "records.json"
LOCK_FILE = "records.lock"
RELEASED_FILE = "released.jsonl"
STATUS_FILE = "status.json"
CONTENT_DIRNAME = "content"

HOLD_WINDOW = "waiting_recreate_window"
HOLD_LANE_BUSY = "lane_lock_held_by_deploy"
HOLD_NOT_APPLIED = "not_applied"
HOLD_NOT_CHECKED = "not_checked"
HOLD_APPLY_FAILED = "apply_failed"
#: A failed apply is held for a person to read, not waiting for a window.
_NOT_WAITING = (HOLD_APPLY_FAILED,)
_WHOLE_STATES = ("present", "reapplied")
#: The only presence at which a restore leaves the window: the main runtime is up, so the
#: install's apply has a container to act on. A DOWN lane is restored, and the restore build
#: applies the installs before its health gate.
_YIELDING_PRESENCE = "companion_down"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        tmp.chmod(0o644)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _records(path: Path) -> dict[str, Any] | None:
    """The record file as a mapping; ``None`` when absent, a ValueError when unreadable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("installs"), list):
        raise ValueError(f"{path} is not a branch-install record file")
    return payload


def migrate_legacy_store(state_dir: Path, legacy: Path | None) -> tuple[str, ...]:
    """Merge the declared legacy store into ``state_dir`` and retire it; the ids it moved.

    Under the store's own lock. The agent's records win on an id both hold. The legacy
    directory is removed only after every moved record and every content file reads back
    in the agent's store; a legacy store that cannot be read is left where it is and logged.
    """
    store = state_dir / STORE_DIRNAME
    if legacy is None or not legacy.is_dir() or legacy.resolve() == store.resolve():
        return ()
    store.mkdir(parents=True, exist_ok=True)
    with (store / LOCK_FILE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            return _merge_legacy(store, legacy)
        except (OSError, ValueError) as exc:
            logger.exception(
                "branch install store migration from %s FAILED, the legacy store is kept: "
                "%s friction_type=branch_install_store_migration_failed",
                legacy,
                exc,
            )
            return ()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _merge_legacy(store: Path, legacy: Path) -> tuple[str, ...]:
    old = _records(legacy / RECORDS_FILE)
    new = _records(store / RECORDS_FILE)
    old_installs: list[dict[str, Any]] = old["installs"] if old is not None else []
    merged: dict[str, Any] = new or {
        "schema_version": (old or {}).get("schema_version", ""),
        "installs": [],
    }
    have = {str(i["id"]) for i in merged["installs"]}
    moved = [i for i in old_installs if str(i["id"]) not in have]
    for blob in sorted((legacy / CONTENT_DIRNAME).glob("*")):
        target = store / CONTENT_DIRNAME / blob.name
        if blob.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".blob.")
            os.close(handle)
            shutil.copyfile(blob, tmp_name)
            Path(tmp_name).replace(target)
    if moved:
        records_path = store / RECORDS_FILE
        if records_path.exists():
            shutil.copy2(records_path, records_path.with_suffix(".json.bak"))
        merged["installs"] = [*merged["installs"], *moved]
        _atomic_write(records_path, json.dumps(merged, indent=2) + "\n")
    released = legacy / RELEASED_FILE
    if released.is_file():
        with (store / RELEASED_FILE).open("a", encoding="utf-8") as out:
            out.write(released.read_text(encoding="utf-8"))
    after = _records(store / RECORDS_FILE)
    present = {str(i["id"]) for i in (after or {}).get("installs", [])}
    missing = {str(i["id"]) for i in old_installs} - present
    blobs = {p.name for p in (legacy / CONTENT_DIRNAME).glob("*") if p.is_file()}
    lost = {b for b in blobs if not (store / CONTENT_DIRNAME / b).is_file()}
    if missing or lost:
        raise ValueError(
            f"merged store lacks installs {sorted(missing)} and content {sorted(lost)}"
        )
    shutil.rmtree(legacy)
    ids = tuple(str(i["id"]) for i in moved)
    logger.info(
        "branch install store MIGRATED from %s into %s: moved=%s; the legacy store is retired",
        legacy,
        store,
        ",".join(ids) or "none (every id was already declared)",
    )
    return ids


def read_declared(state_dir: Path, legacy: Path | None) -> list[dict[str, Any]]:
    """Every declared install, after the legacy store is merged. Raises on an unreadable file."""
    migrate_legacy_store(state_dir, legacy)
    payload = _records(state_dir / STORE_DIRNAME / RECORDS_FILE)
    return payload["installs"] if payload is not None else []


def declared_repos(state_dir: Path, legacy: Path | None) -> list[str]:
    """The repositories of the declared installs; none when unreadable."""
    try:
        return sorted({str(i["repo"]) for i in read_declared(state_dir, legacy)})
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _status(state_dir: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(
            (state_dir / STORE_DIRNAME / STATUS_FILE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _unexpired(install: dict[str, Any], now: datetime) -> bool:
    try:
        return datetime.fromisoformat(str(install["expires_at"])) > now
    except (KeyError, ValueError):
        return True


def held_installs(
    state_dir: Path, legacy: Path | None, now: datetime | None = None
) -> list[dict[str, str]]:
    """Each declared, unexpired install that is not whole, with the reason it is held.

    The reason is the one the last pass wrote into ``status.json`` (the window, the lane
    lock, the readback). An install nothing has read back yet is held as not checked. No
    held install is ever without a non-empty reason.
    """
    clock = now or datetime.now(UTC)
    status = _status(state_dir) or {}
    written = {
        str(h["id"]): h
        for h in status.get("held") or []
        if isinstance(h, dict) and h.get("id") and h.get("reason")
    }
    states = {
        str(r.get("install_key")): str(r.get("state"))
        for r in status.get("readbacks") or []
        if isinstance(r, dict)
    }
    out: list[dict[str, str]] = []
    for install in read_declared(state_dir, legacy):
        key = str(install["id"])
        if not _unexpired(install, clock) or states.get(key) in _WHOLE_STATES:
            continue
        if key in written:
            hold = written[key]
            out.append(
                {
                    "id": key,
                    "code": str(hold.get("code") or HOLD_NOT_APPLIED),
                    "reason": str(hold["reason"]),
                }
            )
        elif key not in states:
            out.append(
                {
                    "id": key,
                    "code": HOLD_NOT_CHECKED,
                    "reason": "registered; no readback of the running containers recorded yet",
                }
            )
        else:
            failed = states[key] == "failed"
            out.append(
                {
                    "id": key,
                    "code": HOLD_APPLY_FAILED if failed else HOLD_NOT_APPLIED,
                    "reason": f"readback state {states[key]}; not whole in the running containers",
                }
            )
    return out


def record_hold(state_dir: Path, legacy: Path | None, code: str, reason: str) -> None:
    """Write ``reason`` as the hold of every install that is not whole, into ``status.json``.

    Read back by ``/branch-installs`` and by ``held_installs``. The next handler pass
    rewrites the file with the reasons its own readbacks give; a pass that holds again
    writes this one again.
    """
    if not reason.strip():
        raise ValueError("a held install carries a reason")
    held = [
        {"id": h["id"], "code": code, "reason": reason}
        for h in held_installs(state_dir, legacy)
        if h["code"] not in _NOT_WAITING
    ]
    status = _status(state_dir) or {}
    status["held"] = held
    _atomic_write(
        state_dir / STORE_DIRNAME / STATUS_FILE, json.dumps(status, indent=2) + "\n"
    )


def waiting(state_dir: Path, legacy: Path | None) -> str | None:
    """One line naming the registered installs that wait for the lane; ``None`` when none do."""
    try:
        holds = [
            h for h in held_installs(state_dir, legacy) if h["code"] not in _NOT_WAITING
        ]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not holds:
        return None
    return "; ".join(f"{h['id']} ({h['reason']})" for h in holds)


def restore_yield(presence: str, state_dir: Path, legacy: Path | None) -> str | None:
    """Why the idle restore leaves the recreate window to a waiting install, else ``None``.

    The restore yields only while the main runtime is up, because only then can the install's
    own apply act on it. A lane that is down is restored at once: the restore's build applies
    the installs before its health gate judges it, so nothing waits on the window.
    """
    if presence != _YIELDING_PRESENCE:
        return None
    return waiting(state_dir, legacy)


def listing(state_dir: Path, legacy: Path | None) -> dict[str, object]:
    """The body of ``/branch-installs``: declared installs, the last readback, the held ones."""
    try:
        declared = read_declared(state_dir, legacy)
        held = held_installs(state_dir, legacy)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"installs": None, "reason": f"{type(exc).__name__}: {exc}"}
    status = _status(state_dir)
    readbacks = status.get("readbacks") if status is not None else None
    return {
        "read_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "declared": declared,
        "last_check": status,
        "missing": [
            str(r.get("install_key"))
            for r in (readbacks if isinstance(readbacks, list) else [])
            if isinstance(r, dict) and r.get("state") in ("missing", "failed")
        ],
        "held": held,
    }


class RegisteredInstalls:
    """The rules above bound to one agent's state directory and its declared legacy store."""

    def __init__(self, state_dir: Path, legacy_store: Path | None) -> None:
        self.state_dir = state_dir
        self.legacy_store = legacy_store

    def declared_repos(self) -> list[str]:
        return declared_repos(self.state_dir, self.legacy_store)

    def listing(self) -> dict[str, object]:
        return listing(self.state_dir, self.legacy_store)

    def record_hold(self, code: str, reason: str) -> None:
        record_hold(self.state_dir, self.legacy_store, code, reason)

    def restore_yield(self, presence: str) -> str | None:
        return restore_yield(presence, self.state_dir, self.legacy_store)
