# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Read a venv back after a repair and answer whether it CARRIES what was asked.

OMN-18663. Every surface in the omnimarket venv-repair path decided success
from an exit status:

* ``scripts/install-node-skill-package.sh`` ran ``uv pip install`` and exited 0
  on uv's status. uv exits 0 for "already satisfied" as readily as for "3
  packages installed", so a plan that resolved to nothing at all is
  indistinguishable from one that landed.
* ``scripts/check-omnimarket-venv-drift.sh --repair`` exited 0 on that script's
  status, without ever re-running the comparison it opened with.
* ``src/omnibase_infra/cli/workspace_reconcile.py`` returned ``ok=True`` on the
  reconciler subprocess's status, for a reconciler that (by its own header,
  OMN-17307) deliberately does not prove its own work and does not necessarily
  own the interpreter the caller is running on.

The visible symptom was the drift guard's own refusal text — "A reconcile ran,
reported SUCCESS, and the venv is STILL drifted" — observed twice on 2026-09-18
against the shared plugin CLI venv (omni_home/CLAUDE.md rule 11).

This module is the readback those surfaces were missing: given a target
interpreter and the canonical omnimarket clone, it reports what is ACTUALLY
installed against what was asked for, one row per distribution, and returns
IN_SYNC only when every row matches. It mutates nothing and it is the same
comparison the in-process guard makes (installed ``direct_url.json`` VCS commit
vs the canonical clone), so a caller that gates on this cannot disagree with the
guard about what "reconciled" means.

There is deliberately no bypass flag. A readback that can be told to pass is a
readback that will be told to pass by the first lane it blocks
(omni_home/CLAUDE.md rule 10); a genuinely un-reconcilable venv is a broken
venv, and the answer to it is a repair, not an override.

Fails CLOSED. A target interpreter that cannot be probed, a clone whose ref
cannot be read, and a requirement whose specifier cannot be parsed are all
INDETERMINATE, which exits non-zero exactly as DRIFTED does. Nothing here
treats "I could not tell" as "in sync".

Usage::

    python scripts/venv_readback.py --python <venv-python> --clone <clone> \\
        [--ref <sha>] [--sibling <requirement> ...] [--label <text>]

Exit codes:
    0  IN_SYNC — every readback row matches
    1  DRIFTED — at least one row does not match
    2  bad invocation
    3  INDETERMINATE — a fact needed to decide could not be read (fail closed)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

EXIT_IN_SYNC = 0
EXIT_DRIFTED = 1
EXIT_USAGE = 2
EXIT_INDETERMINATE = 3

#: Bounded so an unreadable/hung interpreter surfaces as INDETERMINATE rather
#: than as a caller that never returns. Purely local metadata reads.
PROBE_TIMEOUT_SECONDS = 60
GIT_TIMEOUT_SECONDS = 30

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIREMENT_NAME = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)")

#: Runs INSIDE the target interpreter. One invocation, one JSON document, so a
#: caller (and a test fixture standing in for a venv) has exactly one thing to
#: answer. ``direct_url.json`` is the only place a VCS install records the
#: commit it came from; a PyPI wheel has none, which is why ``commit`` is null
#: for one and a sha for the other (OMN-14064).
_PROBE_SOURCE = """
import json, sys
from importlib.metadata import PackageNotFoundError, distribution

def _facts(name):
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return {"version": None, "commit": None}
    raw = dist.read_text("direct_url.json") or ""
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        data = {}
    commit = data.get("vcs_info", {}).get("commit_id")
    return {
        "version": dist.version,
        "commit": commit if isinstance(commit, str) else None,
    }

names = json.loads(sys.argv[1])
print(json.dumps({name: _facts(name) for name in names}))
"""


class ReadbackVerdict(str, Enum):
    """What one readback row says about one distribution."""

    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    ABSENT = "ABSENT"
    UNREADABLE = "UNREADABLE"


#: Row verdicts that are not proof the venv carries what was asked. ABSENT and
#: UNREADABLE are listed explicitly rather than derived from "not MATCH" so that
#: adding a verdict later is a decision someone has to make here.
FAILING_VERDICTS = frozenset(
    {ReadbackVerdict.MISMATCH, ReadbackVerdict.ABSENT, ReadbackVerdict.UNREADABLE}
)


class ReadbackOutcome(str, Enum):
    """The whole readback's answer."""

    IN_SYNC = "IN_SYNC"
    DRIFTED = "DRIFTED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class ReadbackRow:
    """One fact that was asked for, and what the venv actually carries."""

    subject: str
    expected: str
    installed: str | None
    verdict: ReadbackVerdict

    def render(self) -> str:
        installed = self.installed if self.installed is not None else "(absent)"
        return (
            f"  {self.verdict.value:<10} {self.subject:<28} "
            f"installed {installed}  expected {self.expected}"
        )


def normalize_name(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str | None:
    """Return the normalized distribution name a PEP 508 requirement names."""
    match = _REQUIREMENT_NAME.match(requirement)
    if match is None:
        return None
    return normalize_name(match.group("name"))


def classify_commit(installed: str | None, expected: str) -> ReadbackRow:
    """Classify the installed omnimarket VCS commit against the requested ref."""
    subject = "omnimarket (vcs commit)"
    if installed is None:
        return ReadbackRow(subject, expected, None, ReadbackVerdict.ABSENT)
    if installed == expected:
        return ReadbackRow(subject, expected, installed, ReadbackVerdict.MATCH)
    return ReadbackRow(subject, expected, installed, ReadbackVerdict.MISMATCH)


def classify_version(installed: str | None, expected: str) -> ReadbackRow:
    """Classify the installed omnimarket version against the ref's own version.

    A commit match with a version mismatch is not a contradiction to shrug at:
    it means the metadata in site-packages was written by a different build than
    the one the commit claims, which is precisely the "the install did not land"
    state an exit code cannot see.
    """
    subject = "omnimarket (version)"
    if installed is None:
        return ReadbackRow(subject, expected, None, ReadbackVerdict.ABSENT)
    if installed == expected:
        return ReadbackRow(subject, expected, installed, ReadbackVerdict.MATCH)
    return ReadbackRow(subject, expected, installed, ReadbackVerdict.MISMATCH)


def classify_sibling(requirement: str, installed: str | None) -> ReadbackRow:
    """Classify one co-installed sibling against the requirement the ref declares.

    The requirement is a PEP 508 specifier, not an equality: omnimarket declares
    ranges, and a readback that demanded ``==`` would report drift on a venv that
    is exactly right. Satisfaction is decided by ``packaging``; an unparseable
    specifier or a missing ``packaging`` yields UNREADABLE, which fails closed.
    """
    name = requirement_name(requirement) or requirement
    subject = f"{name} (sibling)"
    if installed is None:
        return ReadbackRow(subject, requirement, None, ReadbackVerdict.ABSENT)
    try:
        from packaging.requirements import InvalidRequirement, Requirement
        from packaging.version import InvalidVersion, Version
    except ImportError:
        return ReadbackRow(subject, requirement, installed, ReadbackVerdict.UNREADABLE)
    try:
        parsed = Requirement(requirement)
        version = Version(installed)
    except (InvalidRequirement, InvalidVersion):
        return ReadbackRow(subject, requirement, installed, ReadbackVerdict.UNREADABLE)
    if parsed.specifier.contains(version, prereleases=True):
        return ReadbackRow(subject, requirement, installed, ReadbackVerdict.MATCH)
    return ReadbackRow(subject, requirement, installed, ReadbackVerdict.MISMATCH)


def outcome_for(rows: list[ReadbackRow]) -> ReadbackOutcome:
    """Reduce the rows to one answer, fail-closed.

    An empty row set is INDETERMINATE, never IN_SYNC: a readback that checked
    nothing has proven nothing, and "no rows" is how a silently broken probe
    would present itself.
    """
    if not rows:
        return ReadbackOutcome.INDETERMINATE
    if any(row.verdict is ReadbackVerdict.UNREADABLE for row in rows):
        return ReadbackOutcome.INDETERMINATE
    if any(row.verdict in FAILING_VERDICTS for row in rows):
        return ReadbackOutcome.DRIFTED
    return ReadbackOutcome.IN_SYNC


def exit_code_for(outcome: ReadbackOutcome) -> int:
    """Map an outcome to this tool's exit code. Only IN_SYNC is zero."""
    if outcome is ReadbackOutcome.IN_SYNC:
        return EXIT_IN_SYNC
    if outcome is ReadbackOutcome.DRIFTED:
        return EXIT_DRIFTED
    return EXIT_INDETERMINATE


def probe_installed(
    python_bin: str, names: list[str]
) -> dict[str, dict[str, str | None]]:
    """Ask the TARGET interpreter what it actually has installed.

    Raises ``RuntimeError`` when the interpreter cannot be probed — the caller
    turns that into INDETERMINATE rather than guessing.
    """
    # PYTHONPATH is dropped for the same reason every other probe in this path
    # drops it: an ambient PYTHONPATH shadows the venv's own site-packages, so
    # the answer would describe some other tree. Everything else is inherited,
    # because a stripped environment would change what the interpreter can do.
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    try:
        result = subprocess.run(
            [python_bin, "-c", _PROBE_SOURCE, json.dumps(names)],
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"cannot probe {python_bin}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(
            f"{python_bin} exited {result.returncode} while reporting installed "
            f"packages: {detail[-1] if detail else '(no output)'}"
        )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{python_bin} did not report readable JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"{python_bin} reported {type(parsed).__name__}, not an object"
        )
    return parsed


def _git(clone: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(clone), *args],
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(
            f"git {' '.join(args)} failed in {clone}: "
            f"{detail[-1] if detail else f'exit {result.returncode}'}"
        )
    return result.stdout.strip()


def resolve_ref(clone: Path, ref: str | None) -> str:
    """Resolve the commit the venv is expected to carry.

    ``--ref`` is what the caller asked to install; without one, the canonical
    clone's own checked-out HEAD — the same reference point the in-process guard
    uses, so the two cannot disagree.
    """
    resolved = _git(clone, "rev-parse", ref or "HEAD")
    if not _SHA_RE.match(resolved):
        raise RuntimeError(f"{ref or 'HEAD'} in {clone} did not resolve to a commit")
    return resolved


def declared_version(clone: Path, ref: str) -> str:
    """Read ``[project].version`` from the pyproject.toml OF THE REF."""
    raw = _git(clone, "show", f"{ref}:pyproject.toml")
    data = tomllib.loads(raw)
    project = data.get("project")
    if not isinstance(project, dict):
        raise RuntimeError(f"{ref}:pyproject.toml has no [project] table")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError(f"{ref}:pyproject.toml declares no [project].version")
    return version


def build_rows(
    installed: dict[str, dict[str, str | None]],
    *,
    expected_commit: str,
    expected_version: str,
    siblings: list[str],
) -> list[ReadbackRow]:
    """Turn probed facts plus expectations into the full row set."""
    market = installed.get("omnimarket", {})
    rows = [
        classify_commit(market.get("commit"), expected_commit),
        classify_version(market.get("version"), expected_version),
    ]
    for requirement in siblings:
        name = requirement_name(requirement)
        facts = installed.get(name or "", {})
        rows.append(classify_sibling(requirement, facts.get("version")))
    return rows


def run(
    python_bin: str,
    clone: Path,
    *,
    ref: str | None,
    siblings: list[str],
    label: str,
) -> int:
    """Perform the readback and print it. Returns this tool's exit code."""
    print(f"== readback: {label} ==")
    print(f"  interpreter : {python_bin}")
    print(f"  clone       : {clone}")

    try:
        expected_commit = resolve_ref(clone, ref)
        expected_version = declared_version(clone, expected_commit)
    except (RuntimeError, OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"INDETERMINATE: {exc}", file=sys.stderr)
        print(
            "  Nothing was proven about the venv, so this is NOT a pass.",
            file=sys.stderr,
        )
        return EXIT_INDETERMINATE

    print(f"  expected ref: {expected_commit} (omnimarket {expected_version})")

    names = ["omnimarket"]
    for requirement in siblings:
        name = requirement_name(requirement)
        if name is not None and name not in names:
            names.append(name)

    try:
        installed = probe_installed(python_bin, names)
    except RuntimeError as exc:
        print(f"INDETERMINATE: {exc}", file=sys.stderr)
        print(
            "  Nothing was proven about the venv, so this is NOT a pass.",
            file=sys.stderr,
        )
        return EXIT_INDETERMINATE

    rows = build_rows(
        installed,
        expected_commit=expected_commit,
        expected_version=expected_version,
        siblings=siblings,
    )
    for row in rows:
        print(row.render())

    outcome = outcome_for(rows)
    if outcome is ReadbackOutcome.IN_SYNC:
        print(f"IN_SYNC: {python_bin} carries omnimarket {expected_commit[:12]}.")
        return EXIT_IN_SYNC

    failing = [row for row in rows if row.verdict in FAILING_VERDICTS]
    print(file=sys.stderr)
    print(
        f"{outcome.value}: {python_bin} does NOT carry what was installed.",
        file=sys.stderr,
    )
    for row in failing:
        installed_value = row.installed if row.installed is not None else "(absent)"
        print(
            f"  {row.subject}: installed {installed_value} != expected {row.expected} "
            f"({row.verdict.value})",
            file=sys.stderr,
        )
    print(file=sys.stderr)
    print(
        "  The install step reported success and this venv does not carry its\n"
        "  result (OMN-18663). Treat the repair as NOT done: re-run it and read\n"
        "  the uv output, rather than dispatching from this interpreter.",
        file=sys.stderr,
    )
    return exit_code_for(outcome)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="venv_readback.py",
        description=(
            "Read a venv back after a repair: report what it ACTUALLY carries "
            "against the omnimarket ref and sibling pins that were asked for. "
            "Only an exact match exits 0; there is no bypass flag."
        ),
    )
    parser.add_argument("--python", required=True, help="target venv python to read")
    parser.add_argument(
        "--clone", required=True, help="canonical omnimarket clone to compare against"
    )
    parser.add_argument(
        "--ref",
        default=None,
        help=(
            "commit the venv is expected to carry (default: the clone's own "
            "checked-out HEAD, which is what the in-process guard compares)"
        ),
    )
    parser.add_argument(
        "--sibling",
        action="append",
        default=[],
        dest="siblings",
        metavar="REQUIREMENT",
        help=(
            "a co-installed sibling requirement the ref declares, e.g. "
            "'omnibase-compat>=0.5.7,<0.6.0'; repeatable"
        ),
    )
    parser.add_argument(
        "--label", default="venv readback", help="human label for this readback"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clone = Path(args.clone)
    if not (clone / ".git").exists():
        print(f"ERROR: {clone} is not a git clone.", file=sys.stderr)
        return EXIT_USAGE
    return run(
        args.python,
        clone,
        ref=args.ref,
        siblings=list(args.siblings),
        label=args.label,
    )


if __name__ == "__main__":
    sys.exit(main())
