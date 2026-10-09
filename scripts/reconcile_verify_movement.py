#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Movement verification for workspace reconciliation (OMN-17307).

WHAT THIS EXISTS TO END
Every reconcile step in this workspace has been judged by the **exit status of
the command that was supposed to move the surface**, never by reading the
surface back. Under that rule a repair and a no-op are the same observation, and
a structurally impossible repair looks like a clean one.

The proof is not hypothetical. On `.201`, 2026-08-31 (OMN-17291), the
``omnibase_core`` deploy-source clone carried ``core.bare=true`` while having a
full working tree::

    $ git fetch origin dev --prune                  # exit 0
    $ git checkout -B dev origin/dev
    fatal: this operation must be run in a work tree # exit 128

``fetch`` succeeded forever; ``checkout`` failed forever. A sync loop reading
the fetch's status reported that clone as advancing for as long as it existed.
A loop reading HEAD would have caught it on the first tick.

The venv surface has the same hole. ``reconcile-workspace-venvs.sh`` (OMN-17190)
runs ``uv sync --frozen --inexact`` and ``install-node-skill-package.sh`` and
exits 0 when both return 0. Neither result is read back -- and the provider
co-install is *known* to move pins nobody asked it to move (OMN-16262: a
hardcoded ``COMPAT_PIN`` downgrading ``omnibase-compat`` 0.5.6 -> 0.5.5, which
broke the ``occ`` CLI extension so badly the ``onex`` binary would not start).
That is precisely a content change no exit code can see.

THE CONTRACT
``verdict()`` takes ``(before, after, target)`` and **nothing else**. It has no
parameter for an exit status, deliberately: a signature that accepted one would
let any caller re-introduce the defect. The absence is the enforcement.

    MOVED             after == target, after != before      -> ok
    ALREADY_AT_TARGET before == after == target             -> ok
    DID_NOT_MOVE      after != target                       -> FAIL
    INDETERMINATE     after or target unreadable            -> FAIL

``INDETERMINATE`` fails closed. This is the same posture CLAUDE.md rule 12 takes
on prod health -- "could not determine" is never "fine" -- applied to host state.

The proven floor is stamped by ``reconcile-host.sh`` on a verified full
reconcile, or by an onboarding run after its delegation passed. Onboarding's
``floor-from-venv`` reads the installed build without advancing any clone.

WHY STDLIB-ONLY, AND WHY IT READS DIRECTORIES RATHER THAN IMPORTING
Two independent reasons, both load-bearing:

1. It runs on `.201` outside any project venv, from the host's bare ``python3``.
2. It must be able to verify a venv whose interpreter does not work. A verifier
   that imports ``importlib.metadata`` *from the environment under test* cannot
   report on the failure modes that matter most -- a half-written venv, a
   broken console script, an interpreter uv is mid-way through replacing.

So installed versions are read from ``*.dist-info`` directory names, which
encode ``name-version`` by packaging spec, and installed VCS commits from
``direct_url.json``. Both are plain files. No interpreter starts.

For the same reason it targets the OLDEST interpreter a lab host is likely to
carry rather than the newest: ``timezone.utc`` over ``datetime.UTC`` (3.11+),
and a regex over ``uv.lock``'s machine-generated ``[[package]]`` blocks rather
than ``tomllib`` (also 3.11+). A verifier that will not import on the host it is
meant to verify verifies nothing.

INTERIM BY DESIGN
Same successor as OMN-17190 names: a ``NodeCompute`` drift-detect handler behind
a ``NodeEffect`` reconcile publisher. ``verdict()`` is already the pure function
that handler will be -- total, side-effect free, typed -- so the port is a lift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

FLOOR_SCHEMA = "onex.workspace.floor.v1"
GOVERNED_DISTS = ("omnibase-infra", "omnibase-core", "omnibase-spi", "omnibase-compat")

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 64


# --------------------------------------------------------------------------- #
# Verdict -- the pure core
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Verdict:
    """A movement verdict for one surface.

    ``ok`` is derived, never passed in, so no caller can construct a passing
    verdict for a surface that did not reach its target.
    """

    name: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.name in ("MOVED", "ALREADY_AT_TARGET")


def verdict(before: str | None, after: str | None, target: str | None) -> Verdict:
    """Judge one surface by content alone.

    Note the signature: there is no exit-status parameter and there never will
    be. Whether the repair command succeeded is not evidence that the surface
    moved, and conflating the two is the entire defect class this module closes.
    """
    if not after:
        return Verdict(
            "INDETERMINATE",
            "post-reconcile state is unreadable; refusing to assume it is correct",
        )
    if not target:
        return Verdict(
            "INDETERMINATE",
            "no target to compare against; a surface with no target cannot be attested",
        )
    if after != target:
        return Verdict(
            "DID_NOT_MOVE",
            f"observed {after} but target is {target}"
            + (f" (unchanged from {before})" if before == after else ""),
        )
    if before == after:
        return Verdict("ALREADY_AT_TARGET", f"already at {target}")
    return Verdict("MOVED", f"{before or '<absent>'} -> {after}")


# --------------------------------------------------------------------------- #
# Venv observations -- no interpreter start
# --------------------------------------------------------------------------- #
_DIST_INFO_RE = re.compile(r"^(?P<name>.+?)-(?P<version>[^-]+)\.dist-info$")


def resolve_site_packages(venv: Path) -> Path | None:
    """Locate ``site-packages`` under a venv root without running its python.

    Returns ``None`` rather than raising: an absent venv is a state the caller
    has to report on, not an exception to unwind through.
    """
    lib = Path(venv) / "lib"
    if not lib.is_dir():
        return None
    for child in sorted(lib.iterdir()):
        candidate = child / "site-packages"
        if candidate.is_dir():
            return candidate
    return None


def _dist_info_dir(site_packages: Path, dist: str) -> Path | None:
    site_packages = Path(site_packages)
    if not site_packages.is_dir():
        return None
    prefix = f"{dist}-"
    for child in sorted(site_packages.iterdir()):
        if child.name.startswith(prefix) and child.name.endswith(".dist-info"):
            return child
    return None


def observe_installed_version(site_packages: Path, dist: str) -> str | None:
    """The installed version of ``dist``, read from its ``*.dist-info`` name.

    ``dist`` is spelled as the dist-info prefix (underscores), which is what an
    installer actually writes -- so no name normalisation happens here and none
    can go wrong here.
    """
    found = _dist_info_dir(site_packages, dist)
    if found is None:
        return None
    matched = _DIST_INFO_RE.match(found.name)
    return matched.group("version") if matched else None


def observe_installed_commit(site_packages: Path, dist: str) -> str | None:
    """The VCS commit a git-installed distribution came from.

    ``None`` covers both "not installed" and "installed from PyPI, so there is
    no commit to read" -- which is exactly the state the OMN-17190 foreign
    interpreter was in. Both must read as *cannot tell*, so the verdict table
    fails closed on them, rather than as an empty string that merely compares
    unequal to a SHA.
    """
    found = _dist_info_dir(site_packages, dist)
    if found is None:
        return None
    direct_url = found / "direct_url.json"
    if not direct_url.is_file():
        return None
    try:
        data = json.loads(direct_url.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    commit = data.get("vcs_info", {}).get("commit_id")
    return commit or None


# --------------------------------------------------------------------------- #
# Clone observations
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CloneHealth:
    healthy: bool
    reason: str


def _git(clone: Path, *args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(clone), *args],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env
        return 1, str(exc)
    return proc.returncode, (proc.stdout or proc.stderr).strip()


def observe_clone_head(clone: Path) -> str | None:
    code, out = _git(clone, "rev-parse", "HEAD")
    return out if code == 0 and out else None


def observe_clone_target(clone: Path, ref: str) -> str | None:
    code, out = _git(clone, "rev-parse", ref)
    return out if code == 0 and out else None


def observe_clone_health(clone: Path) -> CloneHealth:
    """Can this clone accept a checkout at all?

    The one non-obvious case is the reason this function exists. A clone with
    ``core.bare=true`` and a real working tree fetches cleanly forever and
    refuses every checkout with exit 128. Nothing that looks at fetch can see
    it; the config key and the presence of a working tree together can.
    """
    clone = Path(clone)
    if not (clone / ".git").exists() and not (clone / "HEAD").exists():
        return CloneHealth(False, f"no git clone at {clone}")

    code, out = _git(clone, "rev-parse", "--git-dir")
    if code != 0:
        return CloneHealth(False, f"git cannot read {clone}: {out}")

    code, bare = _git(clone, "config", "--get", "core.bare")
    declared_bare = code == 0 and bare.strip().lower() == "true"
    has_worktree = (clone / ".git").is_dir() and any(
        (clone / entry).exists()
        for entry in ("src", "README.md", "pyproject.toml", "AGENT.md")
    )
    if declared_bare and has_worktree:
        return CloneHealth(
            False,
            "core.bare=true on a clone that has a working tree: fetch will "
            "succeed and every checkout will fail with 'must be run in a work "
            "tree'. Repair with: git -C "
            f"{clone} config core.bare false",
        )
    return CloneHealth(True, "checkout-capable")


# --------------------------------------------------------------------------- #
# Refused clones: who owns the work, and what is in the way (OMN-20403)
# --------------------------------------------------------------------------- #
# On 2026-10-03 the workspace reconcile log held 66 FAILED and 62 declined ticks
# and no ``ok`` one, because two canonical clones carried staged files nobody
# owned in the ledger. The refusal said the clone did not move; it never said
# whose staged work was in the way, so nothing and nobody acted on it.
#
# This block reads, and in the one place named below writes, but it NEVER
# discards, resets, stashes, cleans or checks out anything in a clone. Removing
# a stale lock is a command it prints for a human.

# The paths the clone reconciler's own build writes into a clone and therefore
# does not read as operator work (``BUILD_SCRATCH_PREFIXES`` in
# ``runtime_build/deploy_source_ref.py``). Repeated here because this module is
# stdlib-only and standalone; a dirty tree of nothing else is not a refusal.
BUILD_SCRATCH_PREFIXES = ("workspace/",)

MSG_SENDER = "reconcile-host"
MSG_TICKET = "OMN-20403"
OPERATOR = "operator"
REFUSAL_STATE_FILE = "clone-refusals.json"
BACKUP_DIRNAME = "dirty-clone-backups"
_LEDGER_ROW_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ \| ([A-Z][A-Z-]*) \|")
_LANE_RE = re.compile(r"(?<![\w-])lane=([^\s|]+)")


@dataclass(frozen=True)
class DirtyState:
    """Staged or modified tracked paths, and the index they sit in."""

    paths: tuple[str, ...]  # "XY path" porcelain lines, sorted
    index_mtime_ns: int
    digest: str


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def observe_dirty_state(clone: Path) -> DirtyState | None:
    """Staged or dirty TRACKED paths in ``clone``; ``None`` when there are none.

    Untracked files are not a refusal reason for the clone reconciler, so they
    are not listed here either.
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={clone}",
                "-C",
                str(clone),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=no",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot run git status in {clone}: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"git status failed in {clone}: {proc.stderr.strip()}")
    entries = [e for e in proc.stdout.split("\0") if e]
    lines: list[str] = []
    skip_next = False
    for entry in entries:
        if skip_next:  # the origin half of a rename or copy
            skip_next = False
            continue
        status, path = entry[:2], entry[3:]
        if status[0] in "RC":
            skip_next = True
        if path.startswith(BUILD_SCRATCH_PREFIXES):
            continue
        lines.append(f"{status} {path}")
    if not lines:
        return None
    lines.sort()
    code, index_path = _git(
        clone, "rev-parse", "--path-format=absolute", "--git-path", "index"
    )
    index = Path(index_path) if code == 0 and index_path else clone / ".git" / "index"
    try:
        mtime_ns = index.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:12]
    return DirtyState(tuple(lines), mtime_ns, digest)


def _ledger_files(ledger: Path) -> list[Path]:
    """The live ledger first, then its archive newest-first."""
    files = [ledger]
    archive = ledger.parent / "archive"
    if archive.is_dir():
        files.extend(sorted(archive.glob("*.md"), reverse=True))
    return files


def find_owner(ledger: Path, repo: str, paths: tuple[str, ...]) -> str | None:
    """The lane of the newest CLAIM row naming this clone and one of ``paths``.

    A reader only: it never opens the ledger for writing. A row owns the work
    when it names the clone AND a dirty path (the bare path or ``<repo>/<path>``),
    so a path such as ``README.md`` alone never claims every clone's staged
    README. Rows are read newest first; the first match wins.
    """
    needles = {p[3:] for p in paths}
    for file in _ledger_files(ledger):
        try:
            rows = file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for row in reversed(rows):
            match = _LEDGER_ROW_RE.match(row)
            if not match or match.group(1) != "CLAIM" or repo not in row:
                continue
            if not any(needle in row for needle in needles):
                continue
            lane = _LANE_RE.search(row)
            if lane:
                return lane.group(1)
    return None


def _load_refusal_state(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_refusal_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_patch_backup(
    clone: Path, repo: str, dirty: DirtyState, directory: Path
) -> Path:
    """Write the staged and dirty diff against HEAD where an approved converge cannot lose it."""
    code, patch = _git_raw(clone, "diff", "--binary", "HEAD")
    if code != 0 or not patch:
        raise RuntimeError(f"cannot read the diff of {clone}: {patch.strip()}")
    directory.mkdir(parents=True, exist_ok=True)
    target = (
        directory
        / f"{repo}-{_utc(dirty.index_mtime_ns / 1e9).replace(':', '')}-{dirty.digest}.patch"
    )
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(patch, encoding="utf-8")
    tmp.replace(target)
    return target


def _git_raw(clone: Path, *args: str) -> tuple[int, str]:
    """Like :func:`_git` but the output is not stripped: a patch ends in a newline."""
    try:
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={clone}", "-C", str(clone), *args],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return proc.returncode, proc.stdout or proc.stderr


def append_msg(ledger: Path, sender: str, to: str, text: str) -> str:
    """Append one MSG row through ``onex-ledger``; returns the row's id.

    The id is ``<timestamp>-<sender>``, so a sender per clone keeps two rows
    written in the same second distinct.
    """
    raw = os.environ.get("OMNI_HOME")
    if not raw:
        raise RuntimeError(
            "OMNI_HOME is not set; the canonical ledger writer is required"
        )
    internal_home = os.environ.get(
        "OMNIBASE_INTERNAL_HOME", str(Path(raw).parent / "omnibase_internal")
    )
    if not internal_home:
        raise RuntimeError(
            "OMNIBASE_INTERNAL_HOME is empty; set the canonical clone path"
        )
    internal = Path(internal_home)
    if not (internal / "pyproject.toml").is_file():
        raise RuntimeError(f"canonical omnibase_internal project missing: {internal}")
    stamp = _utc(time.time())
    msg_id = f"{stamp}-{sender}"
    row = (
        f"{stamp} | MSG | from={sender} | to={to} | id={msg_id} | "
        f"ticket={MSG_TICKET} | {text}"
    )
    proc = subprocess.run(
        [
            "env",
            "-u",
            "PYTHONPATH",
            "uv",
            "run",
            "--project",
            str(internal),
            "onex-ledger",
            str(ledger),
            "--append",
            row,
        ],
        capture_output=True,
        text=True,
        timeout=330,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ledger append refused (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return msg_id


def find_stale_ref_locks(clone: Path, max_age_s: int) -> list[tuple[Path, int]]:
    """Git lock files in the clone's git dir older than ``max_age_s``, with their age.

    Covers ``*.lock`` beside the git dir's own files (``index.lock``,
    ``HEAD.lock``, ``packed-refs.lock``) and everything under ``refs/``. Age is
    the file's mtime against now; a younger lock is a live writer's.
    """
    code, git_dir = _git(
        clone, "rev-parse", "--path-format=absolute", "--git-common-dir"
    )
    if code != 0 or not git_dir:
        return []
    root = Path(git_dir)
    candidates = list(root.glob("*.lock")) + list((root / "refs").rglob("*.lock"))
    now = time.time()
    found = []
    for lock in candidates:
        try:
            age = int(now - lock.stat().st_mtime)
        except OSError:
            continue
        if age > max_age_s:
            found.append((lock, age))
    return sorted(found)


def _cmd_clone_refusal(args: argparse.Namespace) -> int:
    """Explain why one clone is refused. One ``tag<TAB>text`` line per fact on stdout.

    Tags: ``dirty``, ``index``, ``owner``, ``backup``, ``msg``, ``lock``. With
    ``--record`` the call also keeps the consecutive-refusal state under
    ``--state-dir`` and, on the second consecutive refusal at the same index
    state and dirty-path digest, saves a patch backup and appends exactly one
    MSG. Without it nothing is written anywhere.
    """
    clone = Path(args.clone)
    repo = args.repo
    state_dir = Path(args.state_dir)
    state_path = state_dir / REFUSAL_STATE_FILE
    state = _load_refusal_state(state_path) if args.record else {}
    dirty = observe_dirty_state(clone)

    for lock, age in find_stale_ref_locks(clone, args.lock_age_s):
        print(
            f"lock\tstale ref lock: {lock} age {age}s "
            f"(converge timeout {args.lock_age_s}s) — if no git process is using "
            f"{repo}, remove it with: rm -f {lock}"
        )

    if dirty is None:
        if args.record and repo in state:
            del state[repo]
            _save_refusal_state(state_path, state)
        return EXIT_OK

    index_iso = _utc(dirty.index_mtime_ns / 1e9)
    print(f"dirty\t{len(dirty.paths)} dirty paths in {repo}:")
    for line in dirty.paths:
        print(f"dirty\t  {line}")
    print(f"index\tindex mtime {index_iso}")

    owner: str | None = None
    ledger = Path(args.ledger) if args.ledger else None
    if ledger is None:
        print(
            "owner\towner: unknown — ONEX_LEDGER_PATH is not set, so the ledger was not read"
        )
    elif not ledger.is_file():
        print(f"owner\towner: unknown — no ledger at {ledger}")
    else:
        owner = find_owner(ledger, repo, tuple(p[3:] for p in dirty.paths)) or "unowned"
        print(f"owner\towner: {owner}")

    if not args.record:
        return EXIT_OK

    previous = state.get(repo, {})
    same = (
        previous.get("index_mtime_ns") == dirty.index_mtime_ns
        and previous.get("digest") == dirty.digest
    )
    count = int(previous.get("count", 0)) + 1 if same else 1
    msg_id = str(previous.get("msg_id", "")) if same else ""
    print(f"owner\tconsecutive refusals at this index state: {count}")

    if count >= 2 and not msg_id and owner is not None and ledger is not None:
        try:
            backup = save_patch_backup(clone, repo, dirty, state_dir / BACKUP_DIRNAME)
            to = OPERATOR if owner == "unowned" else owner
            listed = ", ".join(p[3:] for p in dirty.paths[:8]).replace("|", "/")
            more = len(dirty.paths) - 8
            tail = f" (+{more} more)" if more > 0 else ""
            msg_id = append_msg(
                ledger,
                f"{MSG_SENDER}-{repo}",
                to,
                f"canonical clone {repo} was refused twice by reconcile-host at the same "
                f"index state ({index_iso}), staged or dirty paths: {listed}{tail}. A patch "
                f"of the diff is saved at {backup}. Land or park this work; then converge "
                f"with bash converge-canonical-clone.sh {repo} --execute",
            )
            print(f"backup\tpatch backup: {backup}")
            print(f"msg\tledger MSG {msg_id} to={to}")
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            print(f"msg\tMSG NOT SENT: {exc}")
    elif msg_id:
        print(f"msg\tledger MSG {msg_id} already sent for this index state")

    state[repo] = {
        "index_mtime_ns": dirty.index_mtime_ns,
        "digest": dirty.digest,
        "count": count,
        "msg_id": msg_id,
    }
    _save_refusal_state(state_path, state)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Lock targets
# --------------------------------------------------------------------------- #
_LOCK_PACKAGE_RE = re.compile(
    r'^\s*name\s*=\s*"(?P<name>[^"]+)"\s*$\n\s*version\s*=\s*"(?P<version>[^"]+)"\s*$',
    re.MULTILINE,
)


def lock_targets(lock: Path, dists: list[str]) -> dict[str, str]:
    """Target versions for lock-governed distributions.

    Parsed with a regex rather than ``tomllib`` on purpose: this module has to
    execute on whatever ``python3`` the host happens to carry, and ``tomllib``
    is 3.11+. The lock's ``[[package]]`` blocks are machine-generated with a
    stable ``name``/``version`` adjacency, so a regex over them is exact.
    """
    lock = Path(lock)
    text = lock.read_text(encoding="utf-8")  # FileNotFoundError is the right failure
    wanted = set(dists)
    found: dict[str, str] = {}
    for match in _LOCK_PACKAGE_RE.finditer(text):
        name = match.group("name")
        if name in wanted:
            found[name] = match.group("version")
    return found


# --------------------------------------------------------------------------- #
# Floor emission -- the OMN-17309 contract
# --------------------------------------------------------------------------- #
def write_floor(
    output: Path,
    omni_home: Path,
    distributions: dict[str, str],
    omnimarket_commit: str | None,
) -> Path:
    """Stamp the proven floor.

    Called by reconcile-host.sh on a verified full reconcile (every dispatch
    premise verdicted ok, ``is_dispatch_premise``, OMN-20111), or by an
    onboarding run after its delegation passed. The floor describes a build
    that was proven rather than merely attempted. Failed reconciliation or
    delegation leaves the previous floor in place.

    The emitted shape is a consumed contract, not an implementation detail:
    ``scripts/onex`` parses this in awk with no JSON parser, so the indentation
    and the key spelling are pinned by the tests. Distribution keys are the
    ``*.dist-info`` prefix (underscores), which removes name normalisation from
    the hot path entirely -- a hyphenated key would silently never match, and a
    floor entry that never matches reads as "not governed" and passes a stale
    venv, so it is refused here at write time.
    """
    for name in distributions:
        if "-" in name:
            raise ValueError(
                f"floor distribution key {name!r} is hyphenated; use the "
                f"*.dist-info spelling ({name.replace('-', '_')!r}) so the "
                "wrapper never has to normalise a name on the hot path"
            )
    document = {
        "schema": FLOOR_SCHEMA,
        # timezone.utc, not datetime.UTC: this module has to import on the
        # oldest python3 a lab host carries, and datetime.UTC is 3.11+.
        "generated_at": datetime.now(timezone.utc).strftime(  # noqa: UP017
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "host": os.uname().nodename,
        "omni_home": str(omni_home),
        "distributions": dict(sorted(distributions.items())),
        "omnimarket_commit": omnimarket_commit or "",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)  # atomic: a reader never sees a half-written floor
    return output


# --------------------------------------------------------------------------- #
# CLI -- the surface the shell reconciler drives
# --------------------------------------------------------------------------- #
def _cmd_verdict(args: argparse.Namespace) -> int:
    """Emit exactly ONE tab-separated line on stdout: surface, verdict, detail.

    One line, one stream, machine-first. The shell caller splits on the tab and
    formats for humans itself. An earlier shape printed a human sentence on
    stderr and a short line on stdout; a caller capturing ``2>&1`` then got both,
    and the interleaving corrupted the detail it parsed back out. A verifier
    whose own output is ambiguous is not a good place to economise.
    """
    result = verdict(before=args.before, after=args.after, target=args.target)
    print(f"{args.surface}\t{result.name}\t{result.detail}")
    return EXIT_OK if result.ok else EXIT_FAIL


def _cmd_observe(args: argparse.Namespace) -> int:
    site_packages = Path(args.site_packages)
    payload = {
        "site_packages": str(site_packages),
        "versions": {
            dist: observe_installed_version(site_packages, dist) for dist in args.dist
        },
        "commits": {
            dist: observe_installed_commit(site_packages, dist)
            for dist in args.commit_dist
        },
    }
    print(json.dumps(payload, indent=2))
    return EXIT_OK


def _cmd_clone_health(args: argparse.Namespace) -> int:
    """One tab-separated line on stdout, same shape as ``verdict``."""
    health = observe_clone_health(Path(args.clone))
    state = "HEALTHY" if health.healthy else "UNHEALTHY"
    print(f"{args.clone}\t{state}\t{health.reason}")
    return EXIT_OK if health.healthy else EXIT_FAIL


def _cmd_lock_targets(args: argparse.Namespace) -> int:
    print(json.dumps(lock_targets(Path(args.lock), args.dist), indent=2))
    return EXIT_OK


def _cmd_floor(args: argparse.Namespace) -> int:
    distributions: dict[str, str] = {}
    for pair in args.distribution:
        name, _, version = pair.partition("=")
        if not name or not version:
            print(f"--distribution expects NAME=VERSION, got {pair!r}", file=sys.stderr)
            return EXIT_USAGE
        distributions[name] = version
    path = write_floor(
        output=Path(args.output),
        omni_home=Path(args.omni_home),
        distributions=distributions,
        omnimarket_commit=args.omnimarket_commit,
    )
    print(f"floor stamped: {path}")
    return EXIT_OK


def _cmd_floor_from_venv(args: argparse.Namespace) -> int:
    """Stamp installed metadata after an onboarding delegation has passed."""
    site_packages = Path(args.site_packages)
    commit = observe_installed_commit(site_packages, "omnimarket")
    if not commit:
        print(
            "cannot stamp floor: installed omnimarket commit is missing",
            file=sys.stderr,
        )
        return EXIT_FAIL
    try:
        targets = lock_targets(Path(args.lock), list(GOVERNED_DISTS))
    except OSError as exc:
        print(f"cannot stamp floor: cannot read lock: {exc}", file=sys.stderr)
        return EXIT_FAIL
    distributions: dict[str, str] = {}
    for dist in targets:
        name = dist.replace("-", "_")
        version = observe_installed_version(site_packages, name)
        if not version:
            print(
                f"cannot stamp floor: lock-governed {dist} is not installed",
                file=sys.stderr,
            )
            return EXIT_FAIL
        distributions[name] = version
    path = write_floor(
        output=Path(args.output),
        omni_home=Path(args.omni_home),
        distributions=distributions,
        omnimarket_commit=commit,
    )
    print(f"floor stamped: {path}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reconcile_verify_movement.py",
        description="Verify a reconcile step by reading the surface back (OMN-17307).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verdict", help="judge one surface by content")
    p.add_argument("--surface", required=True)
    p.add_argument("--before", default=None)
    p.add_argument("--after", default=None)
    p.add_argument("--target", default=None)
    p.set_defaults(func=_cmd_verdict)

    p = sub.add_parser("observe", help="read installed versions/commits from a venv")
    p.add_argument("--site-packages", required=True)
    p.add_argument("--dist", action="append", default=[])
    p.add_argument("--commit-dist", action="append", default=[])
    p.set_defaults(func=_cmd_observe)

    p = sub.add_parser("clone-health", help="can this clone accept a checkout at all")
    p.add_argument("--clone", required=True)
    p.set_defaults(func=_cmd_clone_health)

    p = sub.add_parser(
        "clone-refusal",
        help="name the owner, backup and stale locks of a refused clone",
    )
    p.add_argument("--clone", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--state-dir", required=True)
    p.add_argument("--ledger", default=None)
    p.add_argument("--lock-age-s", type=int, required=True)
    p.add_argument(
        "--record",
        action="store_true",
        help="keep the refusal count and send the MSG (a repair run; never --check)",
    )
    p.set_defaults(func=_cmd_clone_refusal)

    p = sub.add_parser("lock-targets", help="target versions from a uv.lock")
    p.add_argument("--lock", required=True)
    p.add_argument("--dist", action="append", default=[])
    p.set_defaults(func=_cmd_lock_targets)

    p = sub.add_parser("floor", help="stamp the proven floor marker")
    p.add_argument("--output", required=True)
    p.add_argument("--omni-home", required=True)
    p.add_argument("--distribution", action="append", default=[])
    p.add_argument("--omnimarket-commit", default=None)
    p.set_defaults(func=_cmd_floor)

    p = sub.add_parser(
        "floor-from-venv", help="stamp installed build after onboarding passed"
    )
    p.add_argument("--site-packages", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--omni-home", required=True)
    p.add_argument("--output", required=True)
    p.set_defaults(func=_cmd_floor_from_venv)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
