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

OMN-18819 closes the two halves this left open.

*The baseline was the working tree.* Every hash above came from a local
checkout the docstring calls "assumed-current". On an operator machine with an
in-flight edit that is false, and an uncommitted edit would read as the thing
the fleet ought to carry. Expected content now comes from a GIT REF
(``--ref``, default ``origin/dev``), resolved to a sha that every readback
line names.

*It only reported.* A detector nobody acts on is advisory, and this one was
not even scheduled on the machine that owned it. ``converge`` replaces a
drifted host copy from the ref and RE-HASHES it, because a write whose result
is never read back is a hope rather than a convergence.

The cost of leaving it report-only, measured 2026-09-19: ``omnibase_infra#3814``
merged and the deployed hook stayed at its 2026-09-04 bytes until a human
copied it; and OMN-16056 read Done for a month with its change absent from the
fleet, because ``docker/runners/git-mirror-refresh.sh`` was not in
``SYNC_PATHS`` at all and no mechanism carried it.

Modes:
* ``report`` (default) -- print a table of stale/unreadable paths; exit 1 if
  any are found, else 0. Suitable for a cron entry piping to a Slack webhook.
* ``converge`` -- replace every drifted host path from ``--ref`` and read each
  one back; exit 1 if any path could not be proven converged.
* ``discover`` -- print the live remote-vs-local sha256 pairs as JSON
  (debugging aid).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
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


class RefResolutionError(RuntimeError):
    """A git ref or blob could not be resolved.

    Raised rather than falling back to the working tree: a silent fallback
    would restore exactly the "assumed-current" behaviour this replaces, and
    would do it at the moment the ref is broken -- when a wrong answer is
    most likely and least visible.
    """


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        check=False,
        timeout=60,
    )


def resolve_ref_sha(repo_root: Path, ref: str) -> str:
    """Resolve ``ref`` to a 40-hex commit sha, or raise."""
    result = _git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
    sha = result.stdout.decode().strip()
    if result.returncode != 0 or len(sha) != 40:
        raise RefResolutionError(
            f"cannot resolve ref {ref!r} in {repo_root}: "
            f"{result.stderr.decode().strip() or 'no such ref'}"
        )
    return sha


def read_ref_blob(repo_root: Path, ref: str, path: str) -> bytes:
    """Return ``path``'s bytes AT ``ref`` -- never the working tree's bytes."""
    result = _git(repo_root, "show", f"{ref}:{path}")
    if result.returncode != 0:
        raise RefResolutionError(
            f"{path} is not present at {ref}: {result.stderr.decode().strip()}"
        )
    return result.stdout


def read_ref_mode(repo_root: Path, ref: str, path: str) -> str:
    """Return the file mode git records for ``path`` at ``ref``.

    Carried across deliberately: the synced set mixes executable hooks with
    plain data files, and a converge that wrote everything 0644 would leave
    the runner unable to execute its own job-started hook.
    """
    result = _git(repo_root, "ls-tree", ref, "--", path)
    fields = result.stdout.decode().split()
    if result.returncode != 0 or not fields:
        raise RefResolutionError(f"cannot read mode for {path} at {ref}")
    return "0755" if fields[0] == "100755" else "0644"


def compute_ref_hashes(repo_root: Path, ref: str, paths: list[str]) -> dict[str, str]:
    return {
        path: hashlib.sha256(read_ref_blob(repo_root, ref, path)).hexdigest()
        for path in paths
    }


class ModelConvergeResult:
    """What happened to one path, in enough detail to audit afterwards."""

    def __init__(
        self,
        path: str,
        expected_sha256: str,
        before_sha256: str | None,
        readback_sha256: str | None,
        action: str,
        ref_sha: str,
        error: str = "",
    ) -> None:
        self.path = path
        self.expected_sha256 = expected_sha256
        self.before_sha256 = before_sha256
        self.readback_sha256 = readback_sha256
        self.action = action
        self.ref_sha = ref_sha
        self.error = error

    @property
    def converged(self) -> bool:
        """True only when the bytes on the host were READ BACK and matched.

        Deliberately derived rather than set by the caller: a field an emitter
        can write is a field an emitter can write wrongly.
        """
        return (
            not self.error
            and self.readback_sha256 is not None
            and self.readback_sha256 == self.expected_sha256
        )

    def as_line(self) -> str:
        before = (self.before_sha256 or "UNREADABLE")[:12]
        after = (self.readback_sha256 or "UNREADABLE")[:12]
        verdict = "converged" if self.converged else "MISMATCH"
        suffix = f" error={self.error}" if self.error else ""
        return (
            f"{self.path}: {self.action} {before} -> {after} "
            f"(expected {self.expected_sha256[:12]} from {self.ref_sha}) "
            f"{verdict}{suffix}"
        )


def converge_paths(
    *,
    paths: list[str],
    ref_contents: dict[str, bytes],
    remote_hash: Callable[[str], str | None],
    push: Callable[[str, bytes], None],
    ref_sha: str,
) -> list[ModelConvergeResult]:
    """Bring every drifted path in line with the ref, and read each one back.

    Fail-closed on all three failure shapes: an unreadable remote before the
    write, a push that raises, and a readback that disagrees. None of them
    counts as converged, and none of them aborts the remaining paths -- a run
    that dies on the first bad path leaves the rest unconverged AND unreported,
    which is strictly worse than the drift it was called to fix.
    """
    results: list[ModelConvergeResult] = []
    for path in paths:
        expected = hashlib.sha256(ref_contents[path]).hexdigest()
        before = remote_hash(path)

        if before == expected:
            results.append(
                ModelConvergeResult(
                    path, expected, before, before, "already-current", ref_sha
                )
            )
            continue

        try:
            push(path, ref_contents[path])
        except Exception as exc:  # noqa: BLE001 - every push failure is a finding
            results.append(
                ModelConvergeResult(
                    path, expected, before, None, "push-failed", ref_sha, str(exc)
                )
            )
            continue

        results.append(
            ModelConvergeResult(
                path, expected, before, remote_hash(path), "replaced", ref_sha
            )
        )
    return results


def render_converge_report(
    results: list[ModelConvergeResult], ref: str, ref_sha: str
) -> str:
    replaced = [r for r in results if r.action == "replaced" and r.converged]
    failed = [r for r in results if not r.converged]
    lines = [
        f"runner host hook converge: ref={ref} sha={ref_sha} "
        f"{len(results)} path(s), {len(replaced)} replaced, {len(failed)} unresolved"
    ]
    lines.extend(f"  - {result.as_line()}" for result in results)
    return "\n".join(lines)


def _push_to_host(
    ssh_host: str, runner_host_dir: str, mode: str
) -> Callable[[str, bytes], None]:
    """Write bytes to the host atomically, preserving the ref's file mode.

    Writes a sibling temp file and renames it, so a runner that starts a job
    mid-converge never reads a half-written hook.
    """

    def push(path: str, content: bytes) -> None:
        remote = f"{runner_host_dir}/{path}"
        script = (
            f'set -eu; tmp=$(mktemp {remote}.XXXXXX); cat > "$tmp"; '
            f'chmod {mode} "$tmp"; mv "$tmp" {remote!r}'
        )
        result = subprocess.run(
            ["ssh", ssh_host, script],
            input=content,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode != 0:
            raise OSError(result.stderr.decode().strip() or "ssh push failed")

    return push


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
    parser.add_argument(
        "--mode", choices=("report", "converge", "discover"), default="report"
    )
    # The baseline is a REF, never the working tree. See the OMN-18819 note in
    # the module docstring for why a working-tree baseline is not a baseline.
    parser.add_argument("--ref", default="origin/dev")
    # Scope a converge to named paths. The scheduled entry passes none and
    # converges the whole synced set, which is the point; this exists so a
    # first live proof can be taken on one file without shipping an unrelated
    # pending change on another as a side effect.
    parser.add_argument("--only", action="append", default=[])
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

    try:
        ref_sha = resolve_ref_sha(repo_root, args.ref)
    except RefResolutionError as exc:
        # Fail closed and loudly. Continuing against the working tree here is
        # the single change that would silently undo OMN-18819.
        print(f"runner host artifact freshness: REFUSED -- {exc}")
        return 1

    if args.only:
        unknown = sorted(set(args.only) - set(paths))
        if unknown:
            print(
                "runner host artifact freshness: REFUSED -- --only names "
                f"path(s) outside the synced set: {unknown}"
            )
            return 1
        paths = [path for path in paths if path in set(args.only)]

    if args.mode == "converge":
        results: list[ModelConvergeResult] = []
        contents: dict[str, bytes] = {}
        resolvable: list[str] = []
        for path in paths:
            try:
                contents[path] = read_ref_blob(repo_root, args.ref, path)
                resolvable.append(path)
            except RefResolutionError as exc:
                # A synced path absent at the ref is a real finding: either
                # SYNC_PATHS names something deleted, or the ref is wrong.
                results.append(
                    ModelConvergeResult(
                        path, "", None, None, "absent-at-ref", ref_sha, str(exc)
                    )
                )
        for path in resolvable:
            mode = read_ref_mode(repo_root, args.ref, path)
            results.extend(
                converge_paths(
                    paths=[path],
                    ref_contents=contents,
                    remote_hash=lambda candidate: _remote_sha256(
                        args.runner_host, f"{args.runner_host_dir}/{candidate}"
                    ),
                    push=_push_to_host(args.runner_host, args.runner_host_dir, mode),
                    ref_sha=ref_sha,
                )
            )
        print(render_converge_report(results, args.ref, ref_sha))
        return 0 if all(result.converged for result in results) else 1

    local_hashes = compute_ref_hashes(repo_root, args.ref, paths)
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
