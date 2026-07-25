# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Detect staleness of the operator-maintained runner host checkout (OMN-15114).

``deploy-runners.sh`` rsyncs a fixed set of artifacts (``SYNC_PATHS``) from a
repo checkout onto the self-hosted runner host (``~/.omnibase/runners/`` on
.201 / omninode-pc), including ``docker/runners/runner-image.lock.json``,
the ``Dockerfile``, ``entrypoint.sh``, and ``docker-compose.runners.yml``.

That rsync only runs as part of ``deploy-runners.sh``'s full pipeline, which
also fetches a fresh GitHub registration token and force-recreates every
runner container -- a disruptive operation operators avoid for a small fix.
In practice, image rebuilds + container recreates have repeatedly been done
via ad hoc ``docker build`` / ``docker compose --force-recreate`` invocations
directly on the host instead, so the rsync step goes un-run indefinitely.

OMN-15104 (2026-07-09 -> 2026-07-25) closed the *container-vs-repo* half of
this defect class (a running container's baked image lagging the checked-in
lock) with ``check_runner_fleet_image_drift.py``. It did not close the
adjacent *host-artifact-vs-repo* half: the host's own staged checkout of
``runner-image.lock.json`` (and its SYNC_PATHS siblings) sat at
``image_version: 5`` for 19 days after ``origin/dev`` moved to
``image_version: 6`` -- unnoticed because nothing compared the two. This
script closes that gap: it diffs every ``SYNC_PATHS`` entry between a local
(assumed-current) repo checkout and its rsynced copy on the runner host via
sha256, and reports every path that differs or is unreadable on the host.

Modes:
* ``report`` (default) -- print a table of stale/unreadable paths; exit 1 if
  any are found, else 0. Suitable for a cron entry piping to a Slack webhook.
* ``discover`` -- print the live remote-vs-local sha256 pairs as JSON
  (debugging aid).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

DEFAULT_DEPLOY_SCRIPT = Path("scripts/deploy-runners.sh")
DEFAULT_RUNNER_HOST_DIR = "/home/jonah/.omnibase/runners"

# The one SYNC_PATHS entry in deploy-runners.sh that is a shell variable
# (${RUNNER_FLEET_CONFIG}) rather than a literal path -- parse_sync_paths
# only extracts literal quoted strings, so this is added back explicitly.
# Its default resolution (see deploy-runners.sh RUNNER_FLEET_CONFIG) is
# "config/runner_fleet.yaml".
RUNNER_FLEET_CONFIG_DEFAULT_PATH = "config/runner_fleet.yaml"

_SYNC_PATHS_BLOCK_RE = re.compile(r"SYNC_PATHS=\((.*?)\)", re.DOTALL)
_QUOTED_ENTRY_RE = re.compile(r'"([^"]*)"')


class ModelPathDriftFinding:
    """One stale-or-unreadable host artifact finding."""

    def __init__(self, path: str, local_sha256: str, remote_sha256: str | None) -> None:
        self.path = path
        self.local_sha256 = local_sha256
        self.remote_sha256 = remote_sha256

    def as_line(self) -> str:
        remote = (
            "UNREADABLE (missing or ssh/hash failure)"
            if self.remote_sha256 is None
            else self.remote_sha256[:12]
        )
        return f"{self.path}: local={self.local_sha256[:12]} remote={remote}"


def parse_sync_paths(deploy_script_text: str) -> list[str]:
    """Extract the literal quoted paths from deploy-runners.sh's SYNC_PATHS array.

    Non-literal entries (shell variable expansions such as
    ``"${RUNNER_FLEET_CONFIG}"``) are skipped by construction -- the caller is
    responsible for adding those back explicitly if they matter (see
    ``RUNNER_FLEET_CONFIG_DEFAULT_PATH``). Deliberately parsing this out of
    the real script rather than hand-maintaining a second copy of the list:
    a second copy is exactly the kind of divergent-lists bug this ticket
    exists to prevent.
    """
    match = _SYNC_PATHS_BLOCK_RE.search(deploy_script_text)
    if not match:
        raise ValueError("SYNC_PATHS array not found in deploy script text")
    block = match.group(1)
    # Extract the (at most one) quoted entry per line independently, so a
    # variable-expansion entry earlier in the array (e.g.
    # "${RUNNER_FLEET_CONFIG}") can never shift quote-pairing for a later
    # literal entry -- a naive single findall() over the whole block mismatches
    # a closing quote against the next line's opening quote once any entry
    # contains an internal '$'.
    entries: list[str] = []
    for line in block.splitlines():
        found = _QUOTED_ENTRY_RE.findall(line)
        entries.extend(found)
    return [entry for entry in entries if entry.strip() and "$" not in entry]


def find_stale_paths(
    local_hashes: dict[str, str],
    remote_hashes: dict[str, str | None],
) -> list[ModelPathDriftFinding]:
    """Return findings for every path whose remote hash differs or is unreadable.

    Fail-closed: a path we could not hash on the remote (``None``) is
    reported, never silently treated as in-sync -- per the "optional input
    means the check does not exist" lesson, an unverifiable path is
    unverified, not passing.
    """
    findings: list[ModelPathDriftFinding] = []
    for path in sorted(local_hashes):
        local_sha = local_hashes[path]
        remote_sha = remote_hashes.get(path)
        if remote_sha is None or remote_sha != local_sha:
            findings.append(ModelPathDriftFinding(path, local_sha, remote_sha))
    return findings


def render_report(findings: list[ModelPathDriftFinding]) -> str:
    if not findings:
        return "runner host artifact freshness: OK — all synced paths match origin checkout"
    lines = [f"runner host artifact freshness: {len(findings)} path(s) stale on host:"]
    lines.extend(f"  - {finding.as_line()}" for finding in findings)
    return "\n".join(lines)


def _local_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_local_hashes(repo_root: Path, paths: list[str]) -> dict[str, str]:
    return {path: _local_sha256(repo_root / path) for path in paths}


def _remote_sha256(ssh_host: str, remote_path: str) -> str | None:
    """Return the sha256 of ``remote_path`` on ``ssh_host``, or ``None``.

    ``None`` covers every failure mode uniformly (ssh failure, missing file,
    unreadable/unparsable shasum output) -- all mean "cannot verify this
    path is in sync" and must be reported, not swallowed.
    """
    try:
        result = subprocess.run(
            ["ssh", ssh_host, f"shasum -a256 {remote_path!r} 2>/dev/null"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    fields = result.stdout.strip().split()
    if not fields:
        return None
    return fields[0]


def compute_remote_hashes(
    ssh_host: str, runner_host_dir: str, paths: list[str]
) -> dict[str, str | None]:
    return {
        path: _remote_sha256(ssh_host, f"{runner_host_dir}/{path}") for path in paths
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect staleness of the operator-maintained runner host "
            "checkout relative to a repo checkout (OMN-15114)."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--deploy-script", type=Path, default=DEFAULT_DEPLOY_SCRIPT)
    parser.add_argument("--runner-host", required=True)
    parser.add_argument("--runner-host-dir", default=DEFAULT_RUNNER_HOST_DIR)
    parser.add_argument("--mode", choices=("report", "discover"), default="report")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    deploy_script_path = (
        args.deploy_script
        if args.deploy_script.is_absolute()
        else repo_root / args.deploy_script
    )
    paths = parse_sync_paths(deploy_script_path.read_text(encoding="utf-8"))
    if RUNNER_FLEET_CONFIG_DEFAULT_PATH not in paths:
        paths.append(RUNNER_FLEET_CONFIG_DEFAULT_PATH)

    local_hashes = compute_local_hashes(repo_root, paths)
    remote_hashes = compute_remote_hashes(args.runner_host, args.runner_host_dir, paths)

    if args.mode == "discover":
        print(
            json.dumps(
                {"local": local_hashes, "remote": remote_hashes},
                sort_keys=True,
                indent=2,
            )
        )
        return 0

    findings = find_stale_paths(local_hashes, remote_hashes)
    print(render_report(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
