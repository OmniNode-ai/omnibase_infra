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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FLOOR_SCHEMA = "onex.workspace.floor.v1"
CANDIDATE_SOURCE_SCHEMA = "onex.workspace.candidate-source.v1"
_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SIBLING_MANIFEST_BLOCK = re.compile(
    r"^SIBLING_CLONE_MANIFEST=\(\s*(?P<body>.*?)^\)", re.MULTILINE | re.DOTALL
)
_SIBLING_MANIFEST_ENTRY = re.compile(r'^\s*"(?P<name>[a-z0-9_]+)"\s*$', re.MULTILINE)

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
    candidate_source: dict[str, str] | None = None,
) -> Path:
    """Stamp the proven floor.

    Only ever called on a reconcile where every surface verdicted ok, so the
    floor always describes a state that was once *proven* rather than one that
    was merely attempted. A failed reconcile leaves the previous floor in place.

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
    if candidate_source is not None:
        document["candidate_source"] = candidate_source
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)  # atomic: a reader never sees a half-written floor
    return output


def _safe_repo_path(omni_home: Path, name: str) -> Path:
    """Return one declared sibling path, refusing traversal and symlink escape."""
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"candidate repository name is invalid: {name!r}")
    root = omni_home.resolve()
    candidate = (root / name).resolve()
    if candidate.parent != root:
        raise ValueError(f"candidate repository escapes OMNI_HOME: {name!r}")
    return candidate


def _git_text(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or "git command failed"
        raise ValueError(f"candidate repository {repo.name!r} is unreadable: {detail}")
    return proc.stdout.strip()


def _required_candidate_repositories(repositories: list[str]) -> list[str]:
    if not repositories:
        raise ValueError("candidate manifest requires at least one declared repository")
    if len(set(repositories)) != len(repositories):
        raise ValueError(
            "candidate manifest required repository list contains duplicates"
        )
    for name in repositories:
        if not re.fullmatch(r"[a-z0-9_]+", name):
            raise ValueError(f"candidate repository name is invalid: {name!r}")
    return sorted(repositories)


def candidate_repositories_from_sibling_manifest(path: Path) -> list[str]:
    """Read the governed clone set from its one source-of-truth shell manifest.

    The verifier deliberately parses only the declaration grammar it needs;
    evaluating a workspace shell file merely to obtain names would turn
    verification into code execution.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"candidate sibling manifest is unreadable: {exc}") from exc
    match = _SIBLING_MANIFEST_BLOCK.search(source)
    if match is None:
        raise ValueError("candidate sibling manifest has no SIBLING_CLONE_MANIFEST")
    names = _SIBLING_MANIFEST_ENTRY.findall(match.group("body"))
    return _required_candidate_repositories(names)


def _candidate_source_state(omni_home: Path, name: str) -> str:
    repo = _safe_repo_path(omni_home, name)
    if not repo.is_dir() or not (repo / ".git").exists():
        raise ValueError(f"candidate repository {name!r} is missing or not a git clone")
    if _git_text(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError(f"candidate repository {name!r} is dirty")
    head = _git_text(repo, "rev-parse", "HEAD")
    if not _FULL_GIT_SHA.fullmatch(head):
        raise ValueError(f"candidate repository {name!r} has invalid HEAD {head!r}")
    return head


def _strict_json_load(path: Path) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"candidate manifest is unreadable: {exc}") from exc


def _candidate_manifest_path(omni_home: Path, manifest: Path) -> Path:
    root = omni_home.resolve()
    resolved = manifest.resolve()
    if resolved.parent != root:
        raise ValueError("candidate manifest must live directly under OMNI_HOME")
    return resolved


def write_candidate_source_manifest(
    *, output: Path, omni_home: Path, repositories: list[str]
) -> Path:
    """Atomically capture a clean, exact source set for a governed candidate."""
    names = _required_candidate_repositories(repositories)
    output = _candidate_manifest_path(omni_home, output)
    document = {
        "schema": CANDIDATE_SOURCE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017
        "omni_home": str(omni_home.resolve()),
        "repositories": [
            {"name": name, "commit": _candidate_source_state(omni_home, name)}
            for name in names
        ],
    }
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output


def verify_candidate_source_manifest(
    *,
    manifest: Path,
    omni_home: Path,
    repositories: list[str],
    expected_sha256: str | None = None,
) -> dict[str, str]:
    """Fail closed unless a candidate manifest still names this clean source set."""
    names = _required_candidate_repositories(repositories)
    manifest = _candidate_manifest_path(omni_home, manifest)
    raw = manifest.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("candidate manifest digest differs from the proven floor")
    document = _strict_json_load(manifest)
    if (
        not isinstance(document, dict)
        or document.get("schema") != CANDIDATE_SOURCE_SCHEMA
    ):
        raise ValueError("candidate manifest has an unsupported schema")
    if document.get("omni_home") != str(omni_home.resolve()):
        raise ValueError("candidate manifest was generated for a different OMNI_HOME")
    rows = document.get("repositories")
    if not isinstance(rows, list):
        raise ValueError("candidate manifest repositories must be a list")
    declared: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "commit"}:
            raise ValueError(
                "candidate manifest repository rows must contain only name and commit"
            )
        name, commit = row["name"], row["commit"]
        if not isinstance(name, str) or not isinstance(commit, str):
            raise ValueError("candidate manifest repository row values must be strings")
        if name in declared:
            raise ValueError(
                f"candidate manifest contains duplicate repository {name!r}"
            )
        if not _FULL_GIT_SHA.fullmatch(commit):
            raise ValueError(f"candidate manifest has non-full commit for {name!r}")
        declared[name] = commit
    if set(declared) != set(names):
        missing = sorted(set(names) - set(declared))
        extra = sorted(set(declared) - set(names))
        raise ValueError(
            f"candidate manifest repository set differs (missing={missing}, extra={extra})"
        )
    for name in names:
        if _candidate_source_state(omni_home, name) != declared[name]:
            raise ValueError(
                f"candidate repository {name!r} no longer matches the manifest commit"
            )
    return {"manifest": str(manifest), "sha256": digest}


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
    candidate_source = None
    if args.candidate_manifest is not None:
        candidate_source = verify_candidate_source_manifest(
            manifest=Path(args.candidate_manifest),
            omni_home=Path(args.omni_home),
            repositories=args.candidate_repo,
        )
    path = write_floor(
        output=Path(args.output),
        omni_home=Path(args.omni_home),
        distributions=distributions,
        omnimarket_commit=args.omnimarket_commit,
        candidate_source=candidate_source,
    )
    print(f"floor stamped: {path}")
    return EXIT_OK


def _cmd_candidate_manifest(args: argparse.Namespace) -> int:
    path = write_candidate_source_manifest(
        output=Path(args.output),
        omni_home=Path(args.omni_home),
        repositories=args.repo,
    )
    print(f"candidate manifest stamped: {path}")
    return EXIT_OK


def _cmd_candidate_verify(args: argparse.Namespace) -> int:
    proof = verify_candidate_source_manifest(
        manifest=Path(args.manifest),
        omni_home=Path(args.omni_home),
        repositories=args.repo,
        expected_sha256=args.expected_sha256,
    )
    print(json.dumps(proof, sort_keys=True))
    return EXIT_OK


def _cmd_candidate_floor_verify(args: argparse.Namespace) -> int:
    floor = _strict_json_load(Path(args.floor))
    if not isinstance(floor, dict):
        raise ValueError("floor is not an object")
    candidate_source = floor.get("candidate_source")
    if candidate_source is None:
        raise ValueError("floor has no candidate source bound to it")
    if not isinstance(candidate_source, dict) or set(candidate_source) != {
        "manifest",
        "sha256",
    }:
        raise ValueError("floor candidate_source must contain only manifest and sha256")
    manifest, digest = candidate_source["manifest"], candidate_source["sha256"]
    if not isinstance(manifest, str) or not isinstance(digest, str):
        raise ValueError("floor candidate_source values must be strings")
    repositories = args.repo
    if args.sibling_manifest is not None:
        if repositories:
            raise ValueError(
                "candidate floor verification accepts repo names or sibling manifest, not both"
            )
        repositories = candidate_repositories_from_sibling_manifest(
            Path(args.sibling_manifest)
        )
    proof = verify_candidate_source_manifest(
        manifest=Path(manifest),
        omni_home=Path(args.omni_home),
        repositories=repositories,
        expected_sha256=digest,
    )
    if args.expected_omnimarket_commit is not None:
        expected = args.expected_omnimarket_commit
        if not _FULL_GIT_SHA.fullmatch(expected):
            raise ValueError("expected omnimarket commit must be a full SHA")
        floor_commit = floor.get("omnimarket_commit")
        if floor_commit != expected:
            raise ValueError(
                "candidate floor omnimarket commit differs from the installed build"
            )
        document = _strict_json_load(Path(manifest))
        assert isinstance(document, dict)  # verified above
        rows = document["repositories"]
        assert isinstance(rows, list)  # verified above
        market_commit = next(
            (row["commit"] for row in rows if row["name"] == "omnimarket"),
            None,
        )
        if market_commit != expected:
            raise ValueError(
                "candidate manifest omnimarket commit differs from the installed build"
            )
    print(json.dumps(proof, sort_keys=True))
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

    p = sub.add_parser("lock-targets", help="target versions from a uv.lock")
    p.add_argument("--lock", required=True)
    p.add_argument("--dist", action="append", default=[])
    p.set_defaults(func=_cmd_lock_targets)

    p = sub.add_parser("floor", help="stamp the proven floor marker")
    p.add_argument("--output", required=True)
    p.add_argument("--omni-home", required=True)
    p.add_argument("--distribution", action="append", default=[])
    p.add_argument("--omnimarket-commit", default=None)
    p.add_argument("--candidate-manifest")
    p.add_argument("--candidate-repo", action="append", default=[])
    p.set_defaults(func=_cmd_floor)

    p = sub.add_parser(
        "candidate-manifest", help="capture a clean pinned candidate source set"
    )
    p.add_argument("--output", required=True)
    p.add_argument("--omni-home", required=True)
    p.add_argument("--repo", action="append", default=[])
    p.set_defaults(func=_cmd_candidate_manifest)

    p = sub.add_parser("candidate-verify", help="verify a pinned candidate source set")
    p.add_argument("--manifest", required=True)
    p.add_argument("--omni-home", required=True)
    p.add_argument("--repo", action="append", default=[])
    p.add_argument("--expected-sha256")
    p.set_defaults(func=_cmd_candidate_verify)

    p = sub.add_parser(
        "candidate-floor-verify", help="verify the candidate source bound into a floor"
    )
    p.add_argument("--floor", required=True)
    p.add_argument("--omni-home", required=True)
    p.add_argument("--repo", action="append", default=[])
    p.add_argument("--sibling-manifest")
    p.add_argument("--expected-omnimarket-commit")
    p.set_defaults(func=_cmd_candidate_floor_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
