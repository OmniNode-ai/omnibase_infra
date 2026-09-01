#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Durable owner-only FIFO queue for governed .201 pre-push lanes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path("/home/jonah/push-lanes")
QUEUE = ROOT / "QUEUE"
JOURNAL = ROOT / "QUEUE.journal"
QUEUE_LOCK = ROOT / "QUEUE.lock"
RUNNER_LOCK = ROOT / ".runner.lock"
CONTRACT_ROOT = Path.home() / ".omnibase" / "infra" / "push-lane-contracts"
VALIDATOR = ROOT / "queue-contract-validator.py"
RUNLOG = ROOT / "queue-runner.log"
SAFE_PATH = f"{Path.home()}/.local/bin:/usr/local/bin:/usr/bin:/bin"
EXPECTED_HOOK_SHA256 = (
    "cf13426d2c9aa074803e9e797ee60a2077b5260bea092f650bc2295d1e76d2a2"
)
LANE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,199}$")
STATES = frozenset(
    {"ACTIVE", "WAITING", "PUSH_STARTED", "FAILED", "HELD", "REMOTE_VERIFIED"}
)


def log(message: str) -> None:
    RUNLOG.open("a", encoding="utf-8").write(
        f"[queue-runner {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}\n"
    )


def initialize() -> None:
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    CONTRACT_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(ROOT, 0o700)
    os.chmod(CONTRACT_ROOT, 0o700)
    for path in (QUEUE, QUEUE_LOCK, RUNNER_LOCK, RUNLOG):
        path.touch(exist_ok=True)
        os.chmod(path, 0o600)


def file_is_owner_regular(path: Path, mode: int) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return (
        stat.S_ISREG(info.st_mode)
        and not stat.S_ISLNK(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == mode
        and info.st_nlink == 1
    )


def atomic_write(path: Path, raw: bytes) -> None:
    parent_info = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) != 0o700
    ):
        raise RuntimeError("unsafe queue parent")
    if path.exists() or path.is_symlink():
        if not file_is_owner_regular(path, 0o600):
            raise RuntimeError("unsafe queue artifact")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


@contextmanager
def queue_guard() -> Any:
    with QUEUE_LOCK.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validated_queue_lines() -> list[str]:
    raw = QUEUE.read_bytes()
    if any((byte < 0x20 and byte != 0x0A) or byte == 0x7F for byte in raw):
        raise RuntimeError("queue contains a raw control byte")
    try:
        lines = raw.decode("utf-8", "strict").split("\n")
    except UnicodeDecodeError as exc:
        raise RuntimeError("queue is not UTF-8") from exc
    for line in lines:
        if line and not line.startswith("#") and LANE_RE.fullmatch(line) is None:
            raise RuntimeError("queue lane is invalid")
    return lines


def queue_head() -> str | None:
    for line in validated_queue_lines():
        if line and not line.startswith("#"):
            return line
    return None


def remove_queue_head(lane: str) -> None:
    lines = validated_queue_lines()
    if (
        next((line for line in lines if line and not line.startswith("#")), None)
        != lane
    ):
        raise RuntimeError("queue head changed")
    removed = False
    result: list[str] = []
    for line in lines:
        if line == lane and not removed:
            removed = True
            continue
        result.append(line)
    atomic_write(QUEUE, "\n".join(result).encode("utf-8"))


def read_journal() -> dict[str, str] | None:
    if not JOURNAL.exists() and not JOURNAL.is_symlink():
        return None
    if not file_is_owner_regular(JOURNAL, 0o600):
        raise RuntimeError("unsafe journal")
    try:
        value = json.loads(JOURNAL.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid journal") from exc
    required = {
        "schema_version",
        "state",
        "note",
        "updated_at",
        "lane",
        "contract_sha256",
        "head_sha",
        "branch",
        "repo",
        "worktree",
        "git_dir",
        "git_common_dir",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema_version") != 1
    ):
        raise RuntimeError("journal schema rejected")
    if (
        value.get("state") not in STATES
        or not isinstance(value.get("lane"), str)
        or not LANE_RE.fullmatch(value["lane"])
    ):
        raise RuntimeError("journal state rejected")
    for key in required - {"schema_version"}:
        item = value.get(key)
        if (
            not isinstance(item, str)
            or not item
            or any(ord(c) < 32 or ord(c) == 127 for c in item)
        ):
            raise RuntimeError("journal control byte rejected")
    return value


def write_journal(state: str, projection: dict[str, Any], note: str) -> None:
    if state not in STATES or any(ord(c) < 32 or ord(c) == 127 for c in note):
        raise RuntimeError("invalid journal transition")
    value = {
        "schema_version": 1,
        "state": state,
        "note": note,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane": projection["queue_lane"],
        "contract_sha256": projection["contract_sha256"],
        "head_sha": projection["head_sha"],
        "branch": projection["branch"],
        "repo": projection["repo"],
        "worktree": projection["worktree"],
        "git_dir": projection["git_dir"],
        "git_common_dir": projection["git_common_dir"],
    }
    atomic_write(
        JOURNAL,
        (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )


def remove_journal() -> None:
    if not file_is_owner_regular(JOURNAL, 0o600):
        raise RuntimeError("unsafe journal")
    JOURNAL.unlink()
    descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def safe_git(
    worktree: str, *args: str, capture: bool = True
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "HOME": str(Path.home()),
        "PATH": SAFE_PATH,
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_TERMINAL_PROMPT": "0",
    }
    return subprocess.run(
        ["git", "-C", worktree, *args],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_text(worktree: str, *args: str) -> str:
    result = safe_git(worktree, *args)
    if result.returncode:
        raise RuntimeError("git identity probe failed")
    return result.stdout.decode("utf-8", "strict").strip()


def load_projection(lane: str) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), str(CONTRACT_ROOT / f"{lane}.yaml")],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("contract rejected")
    try:
        projection = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("invalid contract projection") from exc
    required = {
        "contract_sha256",
        "lane",
        "repo",
        "repository",
        "worktree",
        "branch",
        "head_sha",
        "remote",
        "argv",
        "cwd",
        "hook_name",
        "hook_value",
        "max_load_ratio",
        "poll_seconds",
        "git_dir",
        "git_common_dir",
    }
    if not isinstance(projection, dict) or set(projection) != required:
        raise RuntimeError("projection identity rejected")
    # The queue filename identifier is lowercase while the signed contract
    # lane is uppercase; retain both rather than silently normalizing either.
    projection["queue_lane"] = lane
    return projection


def config_has_url_rewrite(worktree: str) -> bool:
    result = safe_git(worktree, "config", "--null", "--list")
    if result.returncode:
        raise RuntimeError("git config probe failed")
    for entry in result.stdout.split(b"\0"):
        if not entry:
            continue
        key = entry.split(b"\n", 1)[0].lower()
        if key.startswith(b"url.") and (
            key.endswith(b".insteadof") or key.endswith(b".pushinsteadof")
        ):
            return True
    return False


def validate_url_stream(repository: str, raw: bytes) -> bool:
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--validate-remote-urls", repository],
        input=raw,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def validate_remote(projection: dict[str, Any]) -> bool:
    worktree, remote, repository = (
        projection["worktree"],
        projection["remote"],
        projection["repository"],
    )
    if config_has_url_rewrite(worktree):
        return False
    for args in (
        ("config", "--get-all", f"remote.{remote}.url"),
        ("remote", "get-url", "--all", remote),
        ("remote", "get-url", "--push", "--all", remote),
    ):
        result = safe_git(worktree, *args)
        if result.returncode or not validate_url_stream(repository, result.stdout):
            return False
    pushurl = safe_git(worktree, "config", "--get-all", f"remote.{remote}.pushurl")
    if pushurl.returncode not in (0, 1):
        return False
    return pushurl.returncode != 0 or validate_url_stream(repository, pushurl.stdout)


def validate_hook(projection: dict[str, Any]) -> bool:
    worktree, common = projection["worktree"], projection["git_common_dir"]
    hooks_path = safe_git(worktree, "config", "--get-all", "core.hooksPath")
    if hooks_path.returncode == 0:
        return False
    if hooks_path.returncode != 1:
        return False
    try:
        hook = Path(git_text(worktree, "rev-parse", "--git-path", "hooks/pre-push"))
    except RuntimeError:
        return False
    if hook != Path(common) / "hooks/pre-push":
        return False
    if not file_is_owner_regular(hook, 0o755):
        return False
    raw = hook.read_bytes()
    return (
        hashlib.sha256(raw).hexdigest() == EXPECTED_HOOK_SHA256
        and b"ARGS=(hook-impl --config=.pre-commit-config.yaml --hook-type=pre-push)"
        in raw
    )


def validate_identity(projection: dict[str, Any]) -> bool:
    worktree = projection["worktree"]
    try:
        if os.uname().nodename.split(".", 1)[0] != projection["hook_value"]:
            return False
        if projection["hook_name"] != "PREPUSH_201_GATE_RUNNER_HOSTNAME":
            return False
        if str(Path(worktree).resolve(strict=True)) != worktree:
            return False
        if git_text(worktree, "rev-parse", "--show-toplevel") != worktree:
            return False
        if git_text(worktree, "rev-parse", "--git-dir") != projection["git_dir"]:
            return False
        if (
            git_text(worktree, "rev-parse", "--git-common-dir")
            != projection["git_common_dir"]
        ):
            return False
        if git_text(worktree, "rev-parse", "HEAD") != projection["head_sha"]:
            return False
        if git_text(worktree, "branch", "--show-current") != projection["branch"]:
            return False
        if safe_git(
            worktree, "status", "--porcelain=v1", "--untracked-files=all"
        ).stdout:
            return False
        return validate_remote(projection) and validate_hook(projection)
    except (OSError, RuntimeError, UnicodeDecodeError):
        return False


def capacity_available(ratio_limit: float) -> bool:
    busy = (
        subprocess.run(
            ["pgrep", "-f", r"prepush_smart_tests\.sh"], stdout=subprocess.DEVNULL
        ).returncode
        == 0
    )
    load_one = float(Path("/proc/loadavg").read_text().split()[0])
    return not busy and load_one / os.cpu_count() <= ratio_limit


def claim_or_recover() -> tuple[str, bool] | None:
    with queue_guard():
        journal = read_journal()
        if journal is not None:
            lane = journal["lane"]
            if journal["state"] == "REMOTE_VERIFIED":
                return lane, True
            if queue_head() != lane:
                raise RuntimeError("journal/queue order invariant rejected")
            return lane, journal["state"] in {"HELD", "FAILED"}
        lane = queue_head()
        if lane is None:
            return None
        projection = load_projection(lane)
        write_journal("ACTIVE", projection, "claimed; queue head retained")
        return lane, False


def transition(state: str, projection: dict[str, Any], note: str) -> None:
    with queue_guard():
        journal = read_journal()
        if (
            journal is None
            or journal["lane"] != projection["queue_lane"]
            or queue_head() != projection["queue_lane"]
            or journal["contract_sha256"] != projection["contract_sha256"]
            or journal["head_sha"] != projection["head_sha"]
            or journal["branch"] != projection["branch"]
            or journal["worktree"] != projection["worktree"]
            or journal["git_dir"] != projection["git_dir"]
            or journal["git_common_dir"] != projection["git_common_dir"]
        ):
            raise RuntimeError("transition queue invariant rejected")
        write_journal(state, projection, note)


def verify_remote_sha(projection: dict[str, Any]) -> bool:
    result = safe_git(
        projection["worktree"],
        "ls-remote",
        "--heads",
        projection["remote"],
        f"refs/heads/{projection['branch']}",
    )
    if result.returncode:
        return False
    expected = f"{projection['head_sha']}\trefs/heads/{projection['branch']}\n".encode()
    return result.stdout == expected


def commit_verified_completion(lane: str, projection: dict[str, Any]) -> None:
    with queue_guard():
        journal = read_journal()
        if (
            journal is None
            or journal["lane"] != lane
            or queue_head() != lane
            or journal["contract_sha256"] != projection["contract_sha256"]
            or journal["head_sha"] != projection["head_sha"]
            or journal["branch"] != projection["branch"]
        ):
            raise RuntimeError("completion queue invariant rejected")
        write_journal("REMOTE_VERIFIED", projection, "exact remote SHA verified")
        remove_queue_head(lane)
        remove_journal()


def run_lane(lane: str, held: bool) -> int:
    projection = load_projection(lane)
    if held:
        log(f"LANE {lane}: HELD; explicit --resume required")
        return 2
    while True:
        transition("WAITING", projection, "waiting for host capacity")
        if not capacity_available(float(projection["max_load_ratio"])):
            log(f"LANE {lane}: WAITING, capacity gate remains closed")
            time.sleep(int(projection["poll_seconds"]))
            continue
        projection = load_projection(lane)
        if not validate_identity(projection):
            transition(
                "HELD",
                projection,
                "pre-push identity, remote, or hook validation failed",
            )
            log(f"LANE {lane}: HELD, final validation failed")
            return 1
        if not capacity_available(float(projection["max_load_ratio"])):
            continue
        transition(
            "PUSH_STARTED", projection, "final validation passed; governed hook invoked"
        )
        plog = Path(projection["worktree"]) / ".onex_state" / "push.log"
        plog.parent.mkdir(mode=0o700, exist_ok=True)
        os.chmod(plog.parent, 0o700)
        with plog.open("ab") as handle:
            handle.write(f"=== queue-runner lane {lane} START ===\n".encode())
            environment = {
                "HOME": str(Path.home()),
                "PATH": SAFE_PATH,
                "LANG": "C",
                "LC_ALL": "C",
                "GIT_TERMINAL_PROMPT": "0",
                projection["hook_name"]: projection["hook_value"],
            }
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    projection["worktree"],
                    "push",
                    projection["remote"],
                    f"{projection['branch']}:{projection['branch']}",
                ],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
            handle.write(
                f"=== queue-runner lane {lane} END rc={result.returncode} ===\n".encode()
            )
        if result.returncode:
            transition("FAILED", projection, "governed git push failed; queue retained")
            log(f"LANE {lane}: FAILED, queue retained")
            return 1
        if not verify_remote_sha(projection):
            transition(
                "FAILED",
                projection,
                "push succeeded but exact remote SHA was not observed",
            )
            log(f"LANE {lane}: FAILED, remote SHA not verified")
            return 1
        commit_verified_completion(lane, projection)
        log(f"LANE {lane}: complete; exact remote SHA verified and queue advanced")
        return 0


def self_test() -> int:
    validator = subprocess.run(
        [sys.executable, str(VALIDATOR), "--self-test"], check=False
    )
    return validator.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    initialize()
    if args.self_test:
        return self_test()
    with RUNNER_LOCK.open("a+", encoding="utf-8") as runner:
        try:
            fcntl.flock(runner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another runner holds runner lock; exiting")
            return 0
        if args.resume:
            with queue_guard():
                journal = read_journal()
                if journal is None or journal["state"] not in {"HELD", "FAILED"}:
                    log("resume rejected: no held journal")
                    return 1
                projection = load_projection(journal["lane"])
                if queue_head() != journal["lane"]:
                    log("resume rejected: queue order")
                    return 1
                write_journal("ACTIVE", projection, "explicit operator resume")
        log("runner started (durable journal mode)")
        claimed = claim_or_recover()
        if claimed is None:
            log("queue empty; runner exiting")
            return 0
        lane, held = claimed
        journal = read_journal()
        if journal is not None and journal["state"] == "REMOTE_VERIFIED":
            log(f"LANE {lane}: held after crash at remote-verified boundary")
            return 1
        return run_lane(lane, held)


if __name__ == "__main__":
    raise SystemExit(main())
