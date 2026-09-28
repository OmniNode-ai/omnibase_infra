#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Prove several runtime pull requests in one lab-pool run (OMN-18893 batching).

Why this exists
---------------
Operator ruling 2026-09-28T06:31:26Z: runtime-affecting omnibase_infra PRs whose
changed files are disjoint, and that carry no migration and no contract or topic
schema change, are proved together on the commit the dev merge queue would land,
merged individually on PASS, and bisected on FAIL. Operator ruling
2026-09-28T12:23:22Z ("can we start bundling prs in other repos as well?") extends
the same proof to omnimarket. Omnimarket has no merge queue: its PRs squash onto
dev one by one, and the proved tree is the tree those squashes produce when
nothing else lands between them. verify-landed reports base drift otherwise, as
it does for omnibase_infra. Until this file every runtime PR took its own pool
run, one host for 15 to 60 minutes each.

The dev merge queue squashes each entry onto the one before it, so the commit
that lands for the last member of a group is the group base with every member
squashed on in queue order. Its tree does not depend on commit metadata, so the
tree hash is the identity: this file plans the group, computes that tree with
``git merge-tree`` (no work tree, no index, nothing checked out), and later
checks that the commit which actually landed carries the same tree.
``prepr_pool_prove.sh`` builds the same commit on the pool host and prints its
tree, which must agree (``tree-agrees=yes``).

Subcommands
-----------
plan         group candidate PRs: disjoint files, no migration, no contract or
             topic schema change, not marked risky, at most --max members
build        compute the group commit and tree for a base and member heads, and
             write the params file prepr_runtime_pool.py ``run`` takes
halves       split a failed group in two, keeping queue order (bisection)
verify-landed  compare the landed commits with the proved tree

Nothing here pushes, merges, enqueues or writes the ledger. Exit status: 0 ok,
1 a verification that does not hold, 5 usage or input error.
"""

from __future__ import annotations

import argparse
import fnmatch
import itertools
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_USAGE = 5

#: A group of this size is what one run attributes a failure over; bisection
#: of eight takes three more rounds at most.
DEFAULT_MAX = 8

GROUP_REPOS = ("omnibase_infra", "omnimarket")

#: A change here migrates a database. Migrations run once, in order, on shared
#: lanes; one that fails in a group cannot be told apart cheaply, and two that
#: touch the same ledger conflict. Proved alone.
MIGRATION_PATTERNS = (
    "docker/migrations/*",
    "config/migration_classes.yaml",
    "*/migrations/*.sql",
)

#: A change here moves a contract, a topic or a wire schema: every consumer of
#: it is in scope, so the group would no longer attribute a failure.
CONTRACT_PATTERNS = (
    "*contract.yaml",
    "contracts/*",
    "*/topics/*",
    "src/omnibase_infra/topics/*",
    "src/omnimarket/topics/*",
    "*platform_topic_suffixes.py",
    "*enum_*topic*.py",
    "*/schemas/*",
    ".github/omnimarket-contract-pin.yaml",
)

#: Files every build reads and a squash merges line by line: two PRs that both
#: touch one are never disjoint in effect even when a textual merge is clean.
SHARED_BUILD_FILES = ("pyproject.toml", "uv.lock")

#: A PR carrying one of these labels is proved alone.
RISKY_LABELS = ("risky", "lab:solo", "runtime:solo")


def _matches(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


@dataclass(frozen=True)
class Candidate:
    """One runtime-affecting PR offered for grouping."""

    number: int
    head: str
    files: tuple[str, ...]
    labels: tuple[str, ...] = ()
    repo: str = "omnibase_infra"

    @property
    def key(self) -> str:
        return f"{self.repo}#{self.number}"


@dataclass
class Plan:
    groups: list[list[Candidate]] = field(default_factory=list)
    solo: list[tuple[Candidate, str]] = field(default_factory=list)

    def as_json(self) -> dict[str, object]:
        return {
            "groups": [
                [{"pr": c.number, "head": c.head} for c in g] for g in self.groups
            ],
            "solo": [
                {"pr": c.number, "head": c.head, "why": why} for c, why in self.solo
            ],
        }


def exclusion(c: Candidate) -> str | None:
    """Why this PR is proved alone, or None when it may join a group."""
    if not c.files:
        return "no changed files read"
    mig = [f for f in c.files if _matches(f, MIGRATION_PATTERNS)]
    if mig:
        return f"migration: {mig[0]}"
    con = [f for f in c.files if _matches(f, CONTRACT_PATTERNS)]
    if con:
        return f"contract or topic schema: {con[0]}"
    risky = [lb for lb in c.labels if lb in RISKY_LABELS or lb.startswith("hold:")]
    if risky:
        return f"label {risky[0]}"
    return None


def plan_groups(candidates: Sequence[Candidate], max_size: int = DEFAULT_MAX) -> Plan:
    """Greedy, in the order given (the drain's priority order).

    A candidate joins the first open group none of whose members shares a file
    with it (a shared build file counts as shared by name), else opens a new one.
    A group of one is not a group: its member is proved alone.
    """
    if max_size < 2:
        raise ValueError("a group needs room for at least two members")
    plan = Plan()
    open_groups: list[tuple[list[Candidate], set[str]]] = []
    for c in candidates:
        why = exclusion(c)
        if why is not None:
            plan.solo.append((c, why))
            continue
        mine = set(c.files)
        for members, files in open_groups:
            if len(members) < max_size and not (mine & files):
                members.append(c)
                files.update(mine)
                break
        else:
            open_groups.append(([c], set(mine)))
    for members, _files in open_groups:
        if len(members) >= 2:
            plan.groups.append(members)
        else:
            plan.solo.append((members[0], "no disjoint partner in this batch"))
    return plan


def halves(members: Sequence[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    """Bisection step: first half and second half, queue order kept."""
    if len(members) < 2:
        raise ValueError("a single PR is not bisected; it is the breaking PR")
    mid = (len(members) + 1) // 2
    return list(members[:mid]), list(members[mid:])


# --------------------------------------------------------------------------- git

#: The same identity and dates prepr_pool_prove.sh ``fetch_group`` commits with,
#: so the lane and the host compute the same commit shas for the same inputs.
GROUP_GIT_ENV = {
    "GIT_AUTHOR_NAME": "lab-pool-group",
    "GIT_AUTHOR_EMAIL": "lab-pool-group@lab.invalid",
    "GIT_COMMITTER_NAME": "lab-pool-group",
    "GIT_COMMITTER_EMAIL": "lab-pool-group@lab.invalid",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000",
}

Git = Callable[..., str]


def make_git(repo: Path) -> Git:
    def git(*args: str, env: Mapping[str, str] | None = None) -> str:
        full = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        full.update(env or {})
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            env=full,
            check=False,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args[:3])} failed rc={proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[:400]}"
            )
        return proc.stdout.strip()

    return git


@dataclass
class GroupBuild:
    base: str
    steps: list[tuple[int, str, str, str]]  # (pr, head, commit, tree)

    @property
    def commit(self) -> str:
        return self.steps[-1][2]

    @property
    def tree(self) -> str:
        return self.steps[-1][3]


def build_group(git: Git, base: str, members: Sequence[Candidate]) -> GroupBuild:
    """Squash each member onto the running commit, in order, with plumbing only.

    ``git merge-tree --write-tree`` performs the same three-way merge
    ``git merge --squash`` does (merge base of the running commit and the head),
    and ``commit-tree`` records it with one parent, as the queue's squash does.
    A conflict raises: members were planned disjoint.
    """
    current = git("rev-parse", "--verify", f"{base}^{{commit}}")
    steps: list[tuple[int, str, str, str]] = []
    for c in members:
        head = git("rev-parse", "--verify", f"{c.head}^{{commit}}")
        try:
            tree = git(
                "merge-tree", "--write-tree", "--no-messages", current, head
            ).split()[0]
        except RuntimeError as exc:
            raise RuntimeError(
                f"{c.key} does not squash cleanly onto the group: {exc}"
            ) from exc
        commit = git(
            "commit-tree",
            tree,
            "-p",
            current,
            "-m",
            f"lab-pool group member {c.key} {head}",
            env=GROUP_GIT_ENV,
        )
        steps.append((c.number, head, commit, tree))
        current = commit
    return GroupBuild(base=git("rev-parse", base), steps=steps)


def scratch_repo(
    canonical: Path, remote: str, refspecs: Sequence[str], where: Path
) -> Git:
    """A bare scratch repository borrowing the canonical clone's objects.

    Nothing is written to the canonical clone: fetched and computed objects land
    in the scratch repository only.
    """
    where.mkdir(parents=True, exist_ok=True)
    git = make_git(where)
    if not (where / "HEAD").exists():
        git("init", "-q", "--bare")
        objects = canonical / ".git" / "objects"
        if objects.is_dir():
            (where / "objects" / "info").mkdir(parents=True, exist_ok=True)
            (where / "objects" / "info" / "alternates").write_text(
                f"{objects}\n", encoding="utf-8"
            )
    git("fetch", "-q", remote, *refspecs)
    return git


def params_text(
    build: GroupBuild,
    members: Sequence[Candidate],
    extra: Mapping[str, str] | None = None,
) -> str:
    """The params file prepr_runtime_pool.py ``run`` takes for a group."""
    if not members:
        raise ValueError("a group needs at least one member")
    repo = members[0].repo
    if any(c.repo != repo for c in members):
        raise ValueError("a group cannot mix repositories")
    prefix = "INFRA" if repo == "omnibase_infra" else "MARKET"
    tests = sorted(
        {
            f"{repo}:{f}"
            for c in members
            for f in c.files
            if f.startswith("tests/") and f.endswith(".py") and "/test_" in f
        }
    )
    ids = sorted(
        {
            f"{repo}:{f[len('src/') :]}"
            for c in members
            for f in c.files
            if f.startswith("src/") and f.endswith(".py")
        }
    )
    lines = [
        f"{prefix}_GROUP="
        + shlex.quote(
            " ".join(
                f"{c.number}:{h}"
                for c, (_, h, _, _) in zip(members, build.steps, strict=True)
            )
        ),
        f"{prefix}_GROUP_BASE={build.base}",
        f"{prefix}_GROUP_TREE={build.tree}",
    ]
    if tests:
        lines.append("TESTS=" + shlex.quote(" ".join(tests)))
    if ids:
        lines.append("ID_FILES=" + shlex.quote(" ".join(ids)))
    for k, v in (extra or {}).items():
        lines.append(f"{k}={shlex.quote(v)}")
    return "\n".join(lines) + "\n"


@dataclass
class LandedVerdict:
    exact: bool
    consecutive: bool
    tip: str
    tip_tree: str
    drift: list[str]
    lines: list[str]


def verify_landed(
    git: Git,
    proved_tree: str,
    landed: Sequence[tuple[int, str]],
    member_files: Mapping[int, Sequence[str]],
    *,
    repo: str = "omnibase_infra",
) -> LandedVerdict:
    """Compare what landed with what was proved.

    ``landed`` is (pr, merge commit sha) in merge order. The proof holds for the
    tip exactly when its tree equals the proved tree. When dev moved between the
    proof and the landing, the tip differs only in paths no member touched; that
    is reported as drift (the per-PR rule's base-merge allowance), and a path a
    member touched that differs is a mismatch.
    """
    shas = [sha for _, sha in landed]
    consecutive = all(
        git("rev-parse", f"{b}^1") == git("rev-parse", a)
        for a, b in itertools.pairwise(shas)
    )
    tip = shas[-1]
    tip_tree = git("rev-parse", f"{tip}^{{tree}}")
    exact = tip_tree == proved_tree
    drift: list[str] = []
    mine = {f for fs in member_files.values() for f in fs}
    bad: list[str] = []
    if not exact:
        changed = git(
            "diff-tree", "-r", "--name-only", proved_tree, tip_tree
        ).splitlines()
        drift = [p for p in changed if p not in mine]
        bad = [p for p in changed if p in mine]
    lines = []
    for pr, sha in landed:
        lines.append(
            f"{repo}#{pr} landed {sha} group-tip {tip[:12]} "
            f"proved-tree {proved_tree[:12]} tip-tree {tip_tree[:12]} "
            f"{'exact' if exact else ('base-drift' if not bad else 'MISMATCH')}"
        )
    if not consecutive:
        lines.append(
            "note: the members did not land as consecutive commits; another merge interleaved"
        )
    if drift:
        lines.append(f"drift paths no member touched: {', '.join(drift[:10])}")
    if bad:
        lines.append(f"MISMATCH in member paths: {', '.join(bad[:10])}")
    return LandedVerdict(
        exact=exact,
        consecutive=consecutive,
        tip=tip,
        tip_tree=tip_tree,
        drift=drift,
        lines=lines,
    )


# --------------------------------------------------------------------------- cli


def _load_candidates(path: Path, repo: str) -> list[Candidate]:
    """JSON list of {pr, head, files, labels?}, in priority order."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for item in raw:
        out.append(
            Candidate(
                number=int(item["pr"]),
                head=str(item["head"]),
                files=tuple(item.get("files") or ()),
                labels=tuple(item.get("labels") or ()),
                repo=repo,
            )
        )
    return out


def _candidates_from_map(
    map_dir: Path, prs: Sequence[int], repo: str
) -> list[Candidate]:
    """Candidates from a drain map's changed-files.json (drain_map.py --out)."""
    cf = json.loads((map_dir / "changed-files.json").read_text(encoding="utf-8"))
    out = []
    for n in prs:
        entry = cf.get(f"{repo}#{n}")
        if not isinstance(entry, dict):
            raise ValueError(f"{repo}#{n} is not in {map_dir}/changed-files.json")
        out.append(
            Candidate(
                number=n,
                head=str(entry["head"]),
                files=tuple(entry.get("files") or ()),
                repo=repo,
            )
        )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    src = argparse.ArgumentParser(add_help=False)
    src.add_argument(
        "--candidates", type=Path, help="JSON list of {pr, head, files, labels}"
    )
    src.add_argument(
        "--map-dir", type=Path, help="drain map dir holding changed-files.json"
    )
    src.add_argument(
        "--pr", type=int, action="append", default=[], help="with --map-dir, in order"
    )
    src.add_argument("--repo", choices=GROUP_REPOS, default="omnibase_infra")

    p_plan = sub.add_parser("plan", parents=[src])
    p_plan.add_argument("--max", type=int, default=DEFAULT_MAX)

    p_build = sub.add_parser("build", parents=[src])
    p_build.add_argument(
        "--canonical",
        type=Path,
        required=True,
        help="the canonical clone of --repo",
    )
    p_build.add_argument(
        "--scratch", type=Path, help="scratch bare repo (default: a new temp dir)"
    )
    p_build.add_argument("--remote", default=None)
    p_build.add_argument(
        "--base", default="dev", help="base ref or sha (default: the fetched dev)"
    )
    p_build.add_argument("--params-out", type=Path, required=True)

    p_halves = sub.add_parser("halves", parents=[src])
    del p_halves

    p_ver = sub.add_parser("verify-landed", parents=[src])
    p_ver.add_argument("--canonical", type=Path, required=True)
    p_ver.add_argument("--scratch", type=Path)
    p_ver.add_argument("--remote", default=None)
    p_ver.add_argument("--proved-tree", required=True)
    p_ver.add_argument(
        "--landed",
        action="append",
        default=[],
        help="<pr>=<merge commit sha>, in merge order",
    )

    a = ap.parse_args(argv)
    try:
        if a.candidates:
            cands = _load_candidates(a.candidates, a.repo)
        elif a.map_dir and a.pr:
            cands = _candidates_from_map(a.map_dir, a.pr, a.repo)
        else:
            print("give --candidates, or --map-dir with --pr", file=sys.stderr)
            return EXIT_USAGE
    except (OSError, ValueError, KeyError) as exc:
        print(f"candidates unreadable: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if a.cmd == "plan":
        print(json.dumps(plan_groups(cands, a.max).as_json(), indent=2))
        return EXIT_OK
    if a.cmd == "halves":
        try:
            first, second = halves(cands)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE
        print(
            json.dumps(
                {
                    "first": [c.number for c in first],
                    "second": [c.number for c in second],
                }
            )
        )
        return EXIT_OK

    scratch = a.scratch or Path(tempfile.mkdtemp(prefix="lab-pool-group-"))
    remote = a.remote or f"https://github.com/OmniNode-ai/{a.repo}.git"
    if a.cmd == "build":
        if len(cands) < 2:
            print("a group needs at least two members", file=sys.stderr)
            return EXIT_USAGE
        refspecs = ["+refs/heads/dev:refs/remotes/origin/dev"] + [
            f"+refs/pull/{c.number}/head:refs/pull/{c.number}/head" for c in cands
        ]
        git = scratch_repo(a.canonical, remote, refspecs, scratch)
        for c in cands:
            got = git("rev-parse", f"refs/pull/{c.number}/head")
            if got != c.head:
                print(
                    f"{c.key}: live head {got} is not the planned head {c.head}",
                    file=sys.stderr,
                )
                return EXIT_MISMATCH
        base = "refs/remotes/origin/dev" if a.base == "dev" else a.base
        try:
            b = build_group(git, base, cands)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_MISMATCH
        a.params_out.write_text(params_text(b, cands), encoding="utf-8")
        for c, (_pr, head, commit, tree) in zip(cands, b.steps, strict=True):
            print(f"group-step {c.key} head {head} commit {commit} tree {tree}")
        print(
            f"group-commit {b.commit} tree {b.tree} base {b.base} params {a.params_out}"
        )
        return EXIT_OK

    # verify-landed
    landed: list[tuple[int, str]] = []
    for item in a.landed:
        n, _, sha = item.partition("=")
        if not n.isdigit() or len(sha) != 40:
            print(f"--landed {item!r} is not <pr>=<40-hex sha>", file=sys.stderr)
            return EXIT_USAGE
        landed.append((int(n), sha))
    if not landed:
        print("give --landed for every member, in merge order", file=sys.stderr)
        return EXIT_USAGE
    git = scratch_repo(
        a.canonical, remote, ["+refs/heads/dev:refs/remotes/origin/dev"], scratch
    )
    v = verify_landed(
        git,
        a.proved_tree,
        landed,
        {c.number: c.files for c in cands},
        repo=a.repo,
    )
    print("\n".join(v.lines))
    return (
        EXIT_OK
        if v.exact or (v.drift and not any("MISMATCH" in ln for ln in v.lines))
        else EXIT_MISMATCH
    )


if __name__ == "__main__":
    sys.exit(main())
