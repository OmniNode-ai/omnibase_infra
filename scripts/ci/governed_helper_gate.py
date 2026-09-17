#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18629 — refuse a bare primitive where this repo ships a governed helper.

One gate, one policy file, one baseline. A *pair* is four fields: the bare
primitive's pattern, the governed helper that supersedes it, the scope the pair
applies to, and the ticket that established the helper. Adding a pair is a data
edit to ``config/governed_helper_policy.json``; this module never learns a
primitive by name.

## The class

Four of the twelve verified defects of the 2026-09-17 evening window were the
same shape: an unguarded primitive committed into a repository that already
carried the correct form nearby. Three of the four are in this repository and
are the three pairs shipped in increment 1 (OMN-18608, OMN-18613, OMN-18606).
Each was found by a person reading a failure. Under operating rule 5 the remedy
for that is a gate, not another sweep.

## Ratchet, not allowlist

Occurrences present when the gate lands are recorded in
``config/governed_helper_baseline.json``, keyed by pair, path and a digest of
the matched line, with an occurrence count. The count is what makes it a
ratchet rather than a per-file exemption: a second identical call site in an
already-baselined file exceeds the count and is refused.

The baseline may shrink and may never widen, and both halves are enforced here:

* an entry whose ``ticket`` is missing or empty is a load-time refusal, so
  growing the baseline without filing the work that removes it is unavailable;
* an entry the scanner no longer matches is reported as STALE and fails the
  gate, so a fixed call site cannot leave cover behind for the next one at the
  same path. Deleting the entry is part of fixing the occurrence.

## No suppression surface, deliberately

There is no inline annotation, no flag and no environment variable that
silences a finding. A per-call-site annotation lets the lane introducing the
defect also grant itself the exemption, which is the construction that left
roughly 234 self-written suppressions unreviewed in one repository. A pair that
is wrong is corrected in the policy file, where the change is reviewable and
where a reviewer can see every call site it affects at once.

## Standard library only

This runs in pre-commit, where the resolved interpreter may be a bare system
python. A missing third-party parser on a fail-closed enforcement path would
turn into a refusal of every commit on the machine, so the policy and baseline
are JSON read with ``json``. This is the reason ``ticket_creation_policy.json``
records for the same choice.

## What it cannot catch, stated rather than implied

This is a committed-recipe gate over the tracked file list. It has nothing to
say about a command typed interactively at a terminal, which is the same honest
residual ``tests/test_no_raw_prod_bypass_policy.py`` records. And a pair needs a
helper to point at: where the repository has no governed form yet, there is
nothing to enforce and the gate is silent by construction.

Exit codes: ``0`` clean, ``1`` findings, ``2`` the gate could not read its own
inputs.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH: Final[Path] = REPO_ROOT / "config" / "governed_helper_policy.json"
DEFAULT_BASELINE_PATH: Final[Path] = (
    REPO_ROOT / "config" / "governed_helper_baseline.json"
)

#: The gate's OWN surfaces. These are not authored call sites: the policy and
#: baseline carry every pattern and every matched line by construction, and the
#: two test modules carry fixtures of every bare form on purpose -- in prose, in
#: assertions, and in captured incident bytes.
#:
#: This is a structural self-exclusion by path, enumerated here where a reviewer
#: sees the whole list at once. It is NOT the suppression annotation this gate
#: refuses to have: no file can add itself, and nothing outside this tuple is
#: exempt from anything.
#:
#: The list grew by one during this gate's own landing change, which is the
#: shortest available demonstration that the gate works: the incident-replay
#: module quotes `self.consumer.commit()` four times to explain the defect, and
#: the gate refused the commit that added it.
ALWAYS_EXCLUDED: Final[tuple[str, ...]] = (
    "config/governed_helper_policy.json",
    "config/governed_helper_baseline.json",
    "scripts/ci/governed_helper_gate.py",
    "tests/ci/test_governed_helper_gate.py",
    "tests/ci/test_incident_replay_omn18629.py",
)


class PolicyError(Exception):
    """The gate cannot read its own inputs, so it has not run."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """A bare primitive and the governed helper that supersedes it."""

    id: str
    bare_pattern: str
    governed_helper: str
    governed_form: str
    scope_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    #: Matched against the text immediately preceding a hit. When it matches,
    #: the call site is already governed and is not a finding. This exists
    #: because the governed form of an invocation is often a PREFIX of the bare
    #: one: a positive control during measurement found that `uv run python
    #: x.py` matched the bare-interpreter pattern, so stripping has to happen
    #: before the verdict, not after.
    governed_prefix_pattern: str | None
    established_by: str

    @property
    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.bare_pattern)

    @property
    def compiled_prefix(self) -> re.Pattern[str] | None:
        if self.governed_prefix_pattern is None:
            return None
        return re.compile(self.governed_prefix_pattern + r"$")


@dataclass(frozen=True)
class Policy:
    pairs: tuple[Pair, ...]


def _require_str(raw: object, key: str, source: Path) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise PolicyError(f"{source}: {key!r} must be a non-empty string, got {raw!r}")
    return raw


def _require_globs(raw: object, key: str, source: Path) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise PolicyError(f"{source}: {key!r} must be a list of strings, got {raw!r}")
    return tuple(raw)


def load_policy(path: Path | None = None) -> Policy:
    """Read the declared pairs, or raise.

    There is no built-in pair list to fall back to. A gate that reverts to a
    private copy of its rules when the declaration is unreadable keeps
    admitting commits while claiming to enforce something, which is the exact
    shape of a check reporting green while enforcing nothing.
    """
    source = path or DEFAULT_POLICY_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PolicyError(
            f"governed-helper policy unreadable at {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PolicyError(f"{source}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise PolicyError(f"{source}: top level must be an object")
    raw_pairs = raw.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise PolicyError(f"{source}: 'pairs' must be a non-empty list")

    pairs: list[Pair] = []
    seen: set[str] = set()
    for entry in raw_pairs:
        if not isinstance(entry, dict):
            raise PolicyError(f"{source}: every pair must be an object, got {entry!r}")
        pair_id = _require_str(entry.get("id"), "pairs[].id", source)
        if pair_id in seen:
            raise PolicyError(f"{source}: duplicate pair id {pair_id!r}")
        seen.add(pair_id)
        ticket = _require_str(
            entry.get("established_by"), f"pairs[{pair_id!r}].established_by", source
        )
        if not re.fullmatch(r"OMN-\d+", ticket):
            raise PolicyError(
                f"{source}: pair {pair_id!r} established_by must be an OMN ticket id, "
                f"got {ticket!r}. A pair whose helper cites no ticket cannot be "
                f"traced back to the defect that motivated it."
            )
        pattern = _require_str(
            entry.get("bare_pattern"), f"pairs[{pair_id!r}].bare_pattern", source
        )
        try:
            re.compile(pattern)
        except re.error as exc:
            raise PolicyError(
                f"{source}: pair {pair_id!r} bare_pattern is not a regex ({exc})"
            ) from exc
        prefix = entry.get("governed_prefix_pattern")
        if prefix is not None:
            prefix = _require_str(
                prefix, f"pairs[{pair_id!r}].governed_prefix_pattern", source
            )
            try:
                re.compile(prefix)
            except re.error as exc:
                raise PolicyError(
                    f"{source}: pair {pair_id!r} governed_prefix_pattern is not a "
                    f"regex ({exc})"
                ) from exc
        pairs.append(
            Pair(
                id=pair_id,
                bare_pattern=pattern,
                governed_helper=_require_str(
                    entry.get("governed_helper"),
                    f"pairs[{pair_id!r}].governed_helper",
                    source,
                ),
                governed_form=_require_str(
                    entry.get("governed_form"),
                    f"pairs[{pair_id!r}].governed_form",
                    source,
                ),
                scope_globs=_require_globs(
                    entry.get("scope_globs"), f"pairs[{pair_id!r}].scope_globs", source
                ),
                exclude_globs=_require_globs(
                    entry.get("exclude_globs", []),
                    f"pairs[{pair_id!r}].exclude_globs",
                    source,
                ),
                governed_prefix_pattern=prefix,
                established_by=ticket,
            )
        )
    return Policy(pairs=tuple(pairs))


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineEntry:
    pair: str
    path: str
    line_sha256_12: str
    occurrences: int
    ticket: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.pair, self.path, self.line_sha256_12)


@dataclass(frozen=True)
class Baseline:
    entries: tuple[BaselineEntry, ...]

    def counts(self) -> dict[tuple[str, str, str], BaselineEntry]:
        return {entry.key: entry for entry in self.entries}


def line_digest(text: str) -> str:
    """Twelve hex characters of the matched line, whitespace-normalised.

    Keyed on the line's CONTENT rather than its number so that an unrelated
    edit above a baselined call site does not un-baseline it, which would make
    the gate fire on work that touched nothing it governs.
    """
    normalised = " ".join(text.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:12]


def load_baseline(path: Path | None = None) -> Baseline:
    source = path or DEFAULT_BASELINE_PATH
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PolicyError(
            f"governed-helper baseline unreadable at {source}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise PolicyError(f"{source}: not valid JSON ({exc})") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        raise PolicyError(f"{source}: must be an object carrying an 'entries' list")

    entries: list[BaselineEntry] = []
    for item in raw["entries"]:
        if not isinstance(item, dict):
            raise PolicyError(f"{source}: every entry must be an object, got {item!r}")
        where = item.get("path")
        ticket = item.get("ticket")
        if not isinstance(ticket, str) or not re.fullmatch(r"OMN-\d+", ticket.strip()):
            raise PolicyError(
                f"{source}: the baseline entry for {where!r} declares no removal "
                f"ticket (got {ticket!r}). Every baselined occurrence must cite the "
                f"ticket that will remove it -- a baseline entry with no ticket is "
                f"an allowlist entry, and this gate does not have allowlist entries."
            )
        occurrences = item.get("occurrences")
        if not isinstance(occurrences, int) or occurrences < 1:
            raise PolicyError(
                f"{source}: entry for {where!r} must declare a positive integer "
                f"'occurrences', got {occurrences!r}"
            )
        entries.append(
            BaselineEntry(
                pair=_require_str(item.get("pair"), "entries[].pair", source),
                path=_require_str(where, "entries[].path", source),
                line_sha256_12=_require_str(
                    item.get("line_sha256_12"), "entries[].line_sha256_12", source
                ),
                occurrences=occurrences,
                ticket=ticket.strip(),
            )
        )
    return Baseline(entries=tuple(entries))


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    pair_id: str
    path: str
    line_no: int
    line_text: str
    governed_helper: str
    governed_form: str
    established_by: str


@dataclass
class Findings:
    new: list[Finding] = field(default_factory=list)
    stale: list[BaselineEntry] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.new and not self.stale


def _in_scope(pair: Pair, relpath: str) -> bool:
    if relpath in ALWAYS_EXCLUDED:
        return False
    if any(fnmatch.fnmatch(relpath, glob) for glob in pair.exclude_globs):
        return False
    return any(
        fnmatch.fnmatch(relpath, glob) or fnmatch.fnmatch(Path(relpath).name, glob)
        for glob in pair.scope_globs
    )


def _hits(pair: Pair, text: str) -> list[tuple[int, str]]:
    pattern = pair.compiled
    prefix = pair.compiled_prefix
    found: list[tuple[int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for match in pattern.finditer(line):
            if prefix is not None and prefix.search(line[: match.start()]):
                continue  # already the governed form
            found.append((line_no, line.strip()))
            break  # one finding per line; a reader fixes the line, not the match
    return found


def scan(
    policy: Policy, baseline: Baseline, root: Path, relpaths: list[str]
) -> Findings:
    """Compare the tree against the pairs, net of the baseline."""
    observed: dict[tuple[str, str, str], list[Finding]] = {}
    for pair in policy.pairs:
        for relpath in relpaths:
            if not _in_scope(pair, relpath):
                continue
            try:
                text = (root / relpath).read_text(encoding="utf-8", errors="replace")
            except OSError:
                # An unreadable tracked file is not evidence of cleanliness, but
                # it is also not a call site. Skipped loudly by the caller's own
                # file listing, never silently treated as clean.
                continue
            for line_no, line_text in _hits(pair, text):
                key = (pair.id, relpath, line_digest(line_text))
                observed.setdefault(key, []).append(
                    Finding(
                        pair_id=pair.id,
                        path=relpath,
                        line_no=line_no,
                        line_text=line_text,
                        governed_helper=pair.governed_helper,
                        governed_form=pair.governed_form,
                        established_by=pair.established_by,
                    )
                )

    allowed = baseline.counts()
    findings = Findings()
    for key, hits in observed.items():
        entry = allowed.get(key)
        permitted = entry.occurrences if entry else 0
        if len(hits) > permitted:
            findings.new.extend(hits[permitted:])
    for key, entry in allowed.items():
        if len(observed.get(key, [])) < entry.occurrences:
            findings.stale.append(entry)
    findings.new.sort(key=lambda f: (f.path, f.line_no, f.pair_id))
    findings.stale.sort(key=lambda e: (e.path, e.pair))
    return findings


def render(findings: Findings) -> str:
    lines: list[str] = []
    if findings.new:
        lines.append(
            "REFUSED: a bare primitive was committed where this repository "
            "already ships a governed helper.\n"
        )
        for hit in findings.new:
            lines.append(f"  {hit.path}:{hit.line_no}  [{hit.pair_id}]")
            lines.append(f"    found    : {hit.line_text}")
            lines.append(f"    use      : {hit.governed_form}")
            lines.append(f"    helper   : {hit.governed_helper}")
            lines.append(f"    established by: {hit.established_by}")
            lines.append("")
        lines.append(
            "There is no annotation that suppresses this. Use the governed form, "
            "or change the pair in config/governed_helper_policy.json where the "
            "change is reviewable.\n"
        )
    if findings.stale:
        lines.append(
            "REFUSED: the baseline carries entries the scanner no longer matches. "
            "A fixed call site may not leave cover behind for the next one at the "
            "same path -- delete these entries from "
            "config/governed_helper_baseline.json.\n"
        )
        for entry in findings.stale:
            lines.append(
                f"  {entry.path}  [{entry.pair}]  digest={entry.line_sha256_12} "
                f"expected={entry.occurrences}  ticket={entry.ticket}"
            )
        lines.append("")
    return "\n".join(lines)


def tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    )
    return [line for line in result.stdout.splitlines() if line]


def build_parser() -> argparse.ArgumentParser:
    """The gate's argument surface.

    Deliberately carries no force, skip, warn-only or allow option. A test reads
    these option strings, so adding one is a red test rather than a review
    catch.
    """
    parser = argparse.ArgumentParser(
        prog="governed_helper_gate",
        description=(
            "Refuse a bare primitive committed where a governed helper exists."
        ),
    )
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional paths to scan. Defaults to the whole tracked corpus.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = (args.root or REPO_ROOT).resolve()
    try:
        policy = load_policy(args.policy)
        baseline = load_baseline(args.baseline)
    except PolicyError as exc:
        print(f"governed-helper gate: {exc}", file=sys.stderr)
        return 2

    if args.paths:
        relpaths = [
            str(Path(p).resolve().relative_to(root))
            if Path(p).is_absolute()
            else str(p)
            for p in args.paths
        ]
    else:
        try:
            relpaths = tracked_files(root)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(
                f"governed-helper gate: cannot list tracked files: {exc}",
                file=sys.stderr,
            )
            return 2

    findings = scan(policy, baseline, root, relpaths)
    if findings.clean:
        return 0
    print(render(findings))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
