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
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

FLOOR_SCHEMA = "onex.workspace.floor.v1"

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
    p.set_defaults(func=_cmd_floor)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
