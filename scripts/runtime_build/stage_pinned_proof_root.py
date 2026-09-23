#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stage a pre-PR proof source root from a pinned snapshot, and refuse a bad one (OMN-19086).

THE DEFECT
----------
A proof lane stages a private source root on the surface host and builds the
runtime from it. The runbook's earlier procedure produced that root by copying
the canonical clones. A scheduled ten-minute ``pull --ff-only origin dev``
fast-forwarded ``omnibase_infra`` twelve seconds after such a copy began, and
the staged root ended up with its git directory at the new commit and most of
its working tree at the old one: 8,084 files reading as modified, at a commit
nobody chose. Nothing failed. A lane that trusted the copy would have built an
image from a tree that is not any commit and written a receipt naming a sha its
artifact does not correspond to.

Telling lanes to check first loses the same race, because a check and a copy are
not atomic either. The root has to be taken from something that cannot move.

WHAT THIS SCRIPT DOES
---------------------
``stage``  resolves ONE concrete commit sha per repository before any copying
           starts (an explicit ``--pin REPO=SHA``, or the source's HEAD read
           once), then makes each staged repository with ``git clone --local
           --no-checkout`` and ``git checkout --detach <pin>``. Commit objects
           are immutable and a fast-forward pull never deletes one, so a pull
           that fires mid-stage cannot change what a pinned checkout produces.
           A local clone hard-links its objects, so this costs seconds and
           drops build artifacts. It writes ``proof-root-pins.json`` at the
           root and then runs ``verify`` on what it made.

``verify`` reads that manifest and, for every repository in it, asserts that
           ``git rev-parse HEAD`` equals the pin and ``git status --porcelain``
           is empty. Any failure REFUSES with exit 1 and a line naming the
           repository and both shas. It also refuses a root with no manifest,
           a manifest missing a repository the build needs, and a pin that is
           not a full commit sha: a branch name is not a pin, it is a promise to
           look again.

``cut-lab-ref.sh`` runs ``verify`` before any dogfood-lane build, and before any
build whose ``OMNI_HOME`` carries a pin manifest, so the refusal is on the build
path and not only in the runbook.

The repository set the build needs is read from the two scripts that already own
it, never restated here: ``SIBLING_CLONE_MANIFEST`` in
``sibling_clone_manifest.sh`` (the clones the sibling-pin preflight opens) and
``LAB_REF_REPOS`` in ``cut-lab-ref.sh`` (the clones the hot-patch entrypoint
tags).

This script runs on the surface hosts, one of which ships Python 3.9, so it
imports only the standard library and keeps 3.9-compatible runtime syntax.

Exit codes: 0 staged or verified, 1 refused, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIBLING_MANIFEST_SH = SCRIPT_DIR / "sibling_clone_manifest.sh"
CUT_LAB_REF_SH = SCRIPT_DIR / "cut-lab-ref.sh"

MANIFEST_NAME = "proof-root-pins.json"
MANIFEST_SCHEMA = "proof-root-pins.v1"

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_USAGE = 2

# A full object name: SHA-1 (40) or SHA-256 (64) hex. Abbreviations and ref
# names are refused, because only a full name cannot be re-resolved later.
_FULL_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")

# Variables git exports into hooks that override both -C and cwd. Inheriting
# any of them would point every command below at the invoking repository.
_GIT_LOCATION_ENV_VARS = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_PREFIX",
)

_CLONE_ATTEMPTS = 3


class RefusedError(Exception):
    """A staged root, a pin or a source that must not be built from."""


class UsageError(Exception):
    """A malformed invocation."""


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _GIT_LOCATION_ENV_VARS}
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def _git(
    repo: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        env=_git_env(),
    )
    if check and result.returncode != 0:
        raise RefusedError(
            f"git -C {repo} {' '.join(args)} exited {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result


def _clone_local(source_git_dir: Path, dest: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "clone",
            "--local",
            "--no-checkout",
            "--quiet",
            "--",
            str(source_git_dir),
            str(dest),
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_git_env(),
    )


def _read_bash_array(path: Path, name: str) -> list[str]:
    """Return the quoted or bare words of the bash array ``name=( ... )`` in ``path``."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UsageError(f"cannot read {path}: {exc}") from exc
    match = re.search(rf"^{re.escape(name)}=\(\s*(.*?)\)", text, re.M | re.S)
    if match is None:
        raise UsageError(f"array {name} not found in {path}")
    body = "\n".join(line.split("#", 1)[0] for line in match.group(1).splitlines())
    words = [w.strip("\"'") for w in body.split()]
    if not words:
        raise UsageError(f"array {name} in {path} is empty")
    return words


def required_repos() -> list[str]:
    """The repositories a proof root must contain, in a stable order."""
    ordered: list[str] = []
    for repo in _read_bash_array(
        SIBLING_MANIFEST_SH, "SIBLING_CLONE_MANIFEST"
    ) + _read_bash_array(CUT_LAB_REF_SH, "LAB_REF_REPOS"):
        if repo not in ordered:
            ordered.append(repo)
    return ordered


def _parse_assignments(values: Sequence[str], flag: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        repo, sep, rhs = value.partition("=")
        if not sep or not repo or not rhs:
            raise UsageError(f"{flag} expects REPO=VALUE, got {value!r}")
        if repo in parsed:
            raise UsageError(f"{flag} names {repo} twice")
        parsed[repo] = rhs
    return parsed


def _require_full_sha(repo: str, pin: str) -> str:
    if not _FULL_SHA.match(pin):
        raise RefusedError(
            f"REFUSE {repo}: pin {pin!r} is not a full commit sha. A branch, tag or "
            "abbreviation is re-resolved later and cannot pin a proof root."
        )
    return pin


def _source_git_dir(source: Path) -> Path:
    result = _git(source, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(result.stdout.strip())


def stage(
    dest: Path,
    source_root: Path,
    sources: Mapping[str, Path],
    pins: Mapping[str, str],
    repos: Sequence[str],
) -> dict[str, str]:
    """Stage ``repos`` into ``dest`` at pinned shas; return the verified repo -> sha map."""
    unknown = sorted(set(sources) | set(pins))
    unknown = [r for r in unknown if r not in repos]
    if unknown:
        raise UsageError(f"--source/--pin name repositories not staged: {unknown}")
    if dest.exists() and any(dest.iterdir()):
        raise RefusedError(
            f"REFUSE: {dest} exists and is not empty. A proof root is staged fresh, "
            "so a re-run cannot build a union of two stagings."
        )

    # Resolve every pin BEFORE the first clone. This is the whole fix: after this
    # loop no later movement of any source can change what gets staged.
    plan: dict[str, dict[str, str]] = {}
    for repo in repos:
        source = sources.get(repo, source_root / repo)
        if not source.is_dir():
            raise RefusedError(f"REFUSE {repo}: source {source} does not exist")
        if repo in sources:
            # An explicitly named source is the branch under test. Uncommitted
            # content there would be silently left out of a snapshot at its
            # commit, so the lane would prove a tree without its own change.
            dirt = _git(source, "status", "--porcelain").stdout.strip()
            if dirt:
                raise RefusedError(
                    f"REFUSE {repo}: the source under test {source} has uncommitted "
                    "changes, which a pinned snapshot would leave out. Commit them."
                )
        if repo in pins:
            pin = _require_full_sha(repo, pins[repo])
            pin_origin = "explicit"
        else:
            pin = _git(source, "rev-parse", "HEAD").stdout.strip()
            pin_origin = "source-head"
        plan[repo] = {
            "pin": pin,
            "pin_origin": pin_origin,
            "source": str(source),
            "source_git_dir": str(_source_git_dir(source)),
        }

    dest.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        entry = plan[repo]
        target = dest / repo
        last_error = ""
        for _attempt in range(_CLONE_ATTEMPTS):
            # A clone that overlaps a pull can see a ref whose objects are still
            # arriving and fail. That failure is loud, and a retry is safe
            # because the pinned commit was complete before staging began.
            if target.exists():
                shutil.rmtree(target)
            result = _clone_local(Path(entry["source_git_dir"]), target)
            if result.returncode == 0:
                break
            last_error = result.stderr.strip()
        else:
            raise RefusedError(
                f"REFUSE {repo}: git clone --local from {entry['source_git_dir']} "
                f"failed {_CLONE_ATTEMPTS} times: {last_error}"
            )
        if _git(
            target, "cat-file", "-e", f"{entry['pin']}^{{commit}}", check=False
        ).returncode:
            raise RefusedError(
                f"REFUSE {repo}: pinned commit {entry['pin']} is not in the source "
                f"{entry['source']}. Fetch it there first."
            )
        _git(target, "checkout", "--quiet", "--detach", entry["pin"])

    manifest = {
        "schema": MANIFEST_SCHEMA,
        # time.gmtime, not datetime.UTC: UTC is 3.11+, and the .105 surface host
        # runs this under its system Python 3.9 (measured on the lab proof).
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repos": {
            repo: {
                "pin": plan[repo]["pin"],
                "pin_origin": plan[repo]["pin_origin"],
                "source": plan[repo]["source"],
            }
            for repo in repos
        },
    }
    (dest / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return verify(dest, repos)


def _load_manifest(root: Path) -> dict[str, str]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        raise RefusedError(
            f"REFUSE: {path} does not exist, so this root was not staged from a "
            "pinned snapshot and no commit can be named for it. Stage it with "
            "stage_pinned_proof_root.py stage."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RefusedError(f"REFUSE: {path} is unreadable: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != MANIFEST_SCHEMA:
        raise RefusedError(f"REFUSE: {path} is not a {MANIFEST_SCHEMA} manifest")
    repos = data.get("repos")
    if not isinstance(repos, dict) or not repos:
        raise RefusedError(f"REFUSE: {path} names no repositories")
    pins: dict[str, str] = {}
    for repo, entry in repos.items():
        pin = entry.get("pin") if isinstance(entry, dict) else None
        if not isinstance(pin, str):
            raise RefusedError(f"REFUSE {repo}: {path} carries no pin for it")
        pins[str(repo)] = _require_full_sha(str(repo), pin)
    return pins


def verify(root: Path, required: Sequence[str]) -> dict[str, str]:
    """Assert every staged repository is clean and at its pin; raise on any failure."""
    pins = _load_manifest(root)
    refusals: list[str] = []
    for repo in required:
        if repo not in pins:
            refusals.append(
                f"REFUSE {repo}: the build needs it and the pin manifest does not "
                "name it"
            )
    verified: dict[str, str] = {}
    for repo, pin in sorted(pins.items()):
        staged = root / repo
        head_result = (
            _git(staged, "rev-parse", "HEAD", check=False) if staged.is_dir() else None
        )
        if head_result is None or head_result.returncode != 0:
            refusals.append(f"REFUSE {repo}: HEAD unreadable at {staged}; pinned {pin}")
            continue
        head = head_result.stdout.strip()
        if head != pin:
            refusals.append(f"REFUSE {repo}: HEAD {head} != pinned {pin}")
        dirt = _git(staged, "status", "--porcelain", "--untracked-files=all").stdout
        dirty_paths = [line for line in dirt.splitlines() if line.strip()]
        if dirty_paths:
            shown = "; ".join(p.strip() for p in dirty_paths[:5])
            more = f" (+{len(dirty_paths) - 5} more)" if len(dirty_paths) > 5 else ""
            refusals.append(
                f"REFUSE {repo}: working tree not clean at HEAD {head}, pinned {pin}; "
                f"{len(dirty_paths)} path(s): {shown}{more}"
            )
        if head == pin and not dirty_paths:
            verified[repo] = head
    if refusals:
        raise RefusedError("\n".join(refusals))
    return verified


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage a proof source root from a pinned snapshot, or verify one."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    stage_p = sub.add_parser(
        "stage", help="stage a fresh root at pinned shas, then verify it"
    )
    stage_p.add_argument("--dest", required=True, type=Path, help="the new, empty root")
    stage_p.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="directory holding one clone per repository (read only, never written)",
    )
    stage_p.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="REPO=PATH",
        help="take REPO from PATH instead (the branch under test); must be clean",
    )
    stage_p.add_argument(
        "--pin",
        action="append",
        default=[],
        metavar="REPO=SHA",
        help="pin REPO to this full sha; unpinned repositories take the source HEAD, "
        "read once before any copy",
    )

    verify_p = sub.add_parser(
        "verify", help="refuse a root that is dirty or off its pins"
    )
    verify_p.add_argument("--root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        repos = required_repos()
        if args.command == "stage":
            root = args.dest.resolve()
            verified = stage(
                dest=root,
                source_root=args.source_root.resolve(),
                sources={
                    r: Path(p).resolve()
                    for r, p in _parse_assignments(args.source, "--source").items()
                },
                pins=_parse_assignments(args.pin, "--pin"),
                repos=repos,
            )
        else:
            root = args.root.resolve()
            verified = verify(root, repos)
    except UsageError as exc:
        print(f"[stage-pinned-proof-root] usage: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except RefusedError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "[stage-pinned-proof-root] staged root REFUSED; not building",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    for repo, sha in sorted(verified.items()):
        print(f"[stage-pinned-proof-root] OK {repo} {sha}", file=sys.stderr)
    # One JSON line on stdout: the pinned sha per repository and the fact that the
    # readback matched, for the receipt (runbook section 6.1).
    print(
        json.dumps(
            {"verdict": "VERIFIED", "root": str(root), "repos": verified},
            sort_keys=True,
        )
    )
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
