#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The scheduled release train's decision surface (OMN-18595).

WHY THIS EXISTS
---------------
Deciding to cut is the only manual step left in the release chain. Everything
after it is built and live: ``auto-tag-on-merge.yml`` tags a merged PR whose
title starts with the release prefix and dispatches ``release.yml``;
``release.yml`` builds, publishes to PyPI and fast-forwards ``main``;
``scripts/ci/post_release_dev_bump.py`` restores the one-version-ahead invariant;
``dependency-cascade.yml`` opens the downstream pin bumps;
``release-drift-monitor.yml`` reports the resulting drift every six hours.

Nothing decides to start it. So the decision waits for somebody to notice, and
the backlog grows: measured 2026-09-17, omniintelligence carried 67 unreleased
source commits against a tag dated 2026-05-26 and omniclaude 65 against one
dated 2026-06-02.

This module is the missing decision, and nothing else. It never tags, never
publishes and never deploys. Its whole output is a verdict per repo with a
reason, which the workflow either acts on by opening the changelog release PR
the existing tagger already recognises, or reports.

WHY A REPORTED SKIP IS A FIRST-CLASS OUTPUT
-------------------------------------------
A train that says nothing when it does not cut is indistinguishable from a train
that is not running. That is the same reasoning that makes the lab-pass receipt
emit under ``always()``: a record that only exists when everything worked cannot
tell "it failed" from "nobody ran it". So every repo gets a row every night, and
every row carries a reason a reader can act on.

THE PREMISES
------------
Two, and they are deliberately not three.

1. **Unreleased release-relevant work.** At least one commit between the latest
   ``v*`` tag and the default branch's head touches a path the policy declares
   release-relevant, and is not the train's own bookkeeping.
2. **Lab evidence**, declared per repo. ``compose-dev`` requires a PASS
   ``lab-pass-receipt-compose-dev-<sha>`` for the exact candidate sha, resolved
   through the rule-24(b) reader's own primitives in ``lab_pass_receipt.py`` --
   there is no second implementation of that query here. ``none`` is a declared
   statement that the repo has no lab lane, which is true of five of the seven
   repos in scope, and it must carry its reason.

A third premise, "dev CI concluded green on the candidate sha", is deliberately
NOT re-derived. Every repo in this registry is squash-only with no merge queue
and requires a pull request on its default branch, so a commit on that branch is
a merge commit GitHub only created after every required context reported
success. The green-ness holds by construction, and re-deriving it would add a
false-negative surface (an unrelated scheduled run failing on the same sha)
without adding a fact. This is the same reasoning ``decide_release_on_merge.py``
records for its own trigger.

WHY THE LAB PREMISE HAS FOUR FAILURE MODES AND NOT ONE
------------------------------------------------------
The rule-24(b) reader is a gate: it answers pass or refuse, which is correct for
a gate. A train has to tell a reader WHICH way the premise failed, because the
four cases belong to four different owners. A missing receipt is a question for
the rebuild trigger; a FAIL receipt is a question for the lab lane; a name and
payload that disagree is a question for the emitter; an unreadable surface is a
question for the credential. Collapsing them sends three of those four readers
to the wrong place, which is the shape OMN-18573 removed one layer down.

Deliberately stdlib-plus-pyyaml. It runs before any project install.

Usage::

    python3 scripts/ci/release_train.py plan \\
        --policy config/release_train_policy.yaml \\
        --clones-dir /tmp/train-clones \\
        --out decision.json

    python3 scripts/ci/release_train.py changelog \\
        --policy config/release_train_policy.yaml \\
        --repo omnibase_infra --clone /tmp/train-clones/omnibase_infra \\
        --version 0.38.31

Exit codes:
    0 -- every repo rendered a verdict; some may be skips, which is not failure
    2 -- a configuration error: an undeclared repo, an unreadable version, a
         malformed policy. Never reported as "nothing to do".
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
DEFAULT_POLICY_PATH = _REPO_ROOT / "config" / "release_train_policy.yaml"


def _load_sibling(name: str) -> Any:
    """Import a sibling script by path.

    These modules are scripts rather than an installed package, and this one is
    itself loaded by path in tests, so a plain ``import`` resolves differently
    depending on the caller. Loading by path resolves identically everywhere.
    """
    module_name = f"_release_train_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _HERE / f"{name}.py")
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load sibling module {name}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


#: The rule-24(b) receipt reader. Its primitives are the ONLY path to a receipt
#: here; this module classifies what they return and never queries in parallel.
lab_pass_receipt = _load_sibling("lab_pass_receipt")

_FINAL_SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_RELEASE_TAG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

#: Subjects the train itself produces. The post-release bump edits
#: ``pyproject.toml``, which is a release-relevant path, so counted naively it is
#: one unreleased commit the morning after every release and the train cuts an
#: empty release every night forever. Anchored to the start of the subject's
#: type-and-scope prefix rather than matched loosely, so a commit that merely
#: MENTIONS a release is still real work (rule 15, in the other direction).
_BOOKKEEPING_SUBJECT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^chore(\([^)]*\))?:\s*post-release dev version bump\b"),
    re.compile(r"^chore(\([^)]*\))?:\s*release\b"),
    re.compile(r"^release:\s"),
)


class ReleaseTrainConfigError(Exception):
    """Operator-facing misuse: an undeclared repo, a malformed policy."""


class EnumLabEvidence(StrEnum):
    """How a repo's candidate sha is proven to have run on a lab lane."""

    #: A PASS ``lab-pass-receipt-compose-dev-<sha>`` in the repo the sha belongs
    #: to. True of omnibase_infra and omnimarket, the only two repos that carry
    #: a rebuild trigger.
    COMPOSE_DEV = "compose-dev"
    #: The repo has no lab lane of its own. A declaration, with its reason, not
    #: a way around the bar -- the reason is printed on every decision it makes.
    NONE = "none"


class EnumTrainMode(StrEnum):
    """What the train is permitted to do for a repo."""

    CUT = "cut"
    REPORT_ONLY = "report_only"


class EnumTrainVerdict(StrEnum):
    CUT = "CUT"
    SKIP = "SKIP"
    REFUSE = "REFUSE"


class EnumTrainReason(StrEnum):
    """Why the train reached its verdict. Every value is actionable by itself."""

    UNRELEASED_WORK_LAB_PROVEN = "unreleased_work_lab_proven"
    UNRELEASED_WORK_NO_LAB_SURFACE_DECLARED = "unreleased_work_no_lab_surface_declared"
    NO_UNRELEASED_RELEASE_RELEVANT_WORK = "no_unreleased_release_relevant_work"
    MODE_REPORT_ONLY = "mode_report_only"
    LAB_RECEIPT_ABSENT = "lab_receipt_absent"
    LAB_RECEIPT_FAIL = "lab_receipt_fail"
    LAB_RECEIPT_NAME_PAYLOAD_DISAGREE = "lab_receipt_name_payload_disagree"
    LAB_RECEIPT_UNREADABLE = "lab_receipt_unreadable"
    VERSION_UNREADABLE = "version_unreadable"
    FACTS_UNREADABLE = "facts_unreadable"


@dataclass(frozen=True)
class ModelRepoReleasePolicy:
    """One repo's declared release-train policy."""

    repo: str
    package: str
    default_branch: str
    release_relevant_paths: tuple[str, ...]
    lab_evidence: EnumLabEvidence
    lab_evidence_note: str
    mode: EnumTrainMode
    mode_note: str


@dataclass(frozen=True)
class ModelRepoFacts:
    """What the repo's own git history says, at one moment."""

    repo: str
    latest_tag: str
    dev_version: str
    dev_head_sha: str
    unreleased_count: int
    unreleased_subjects: tuple[str, ...]


@dataclass(frozen=True)
class ModelTrainDecision:
    """One repo's verdict, renderable as JSON and as a human row."""

    repo: str
    verdict: EnumTrainVerdict
    reason: EnumTrainReason
    detail: str
    # The branch a release PR for this repo opens against. Carried on the
    # decision, not baked into the workflow, because the train fans out across
    # repositories and a branch name in the automation is correct only for as
    # long as every repository agrees (OMN-18588, same repo, same day).
    base_branch: str
    candidate_sha: str
    candidate_version: str
    latest_tag: str
    dev_version: str
    unreleased_count: int
    needs_bump: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "verdict": self.verdict.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "base_branch": self.base_branch,
            "candidate_sha": self.candidate_sha,
            "candidate_version": self.candidate_version,
            "latest_tag": self.latest_tag,
            "dev_version": self.dev_version,
            "unreleased_count": self.unreleased_count,
            "needs_bump": self.needs_bump,
        }


# --------------------------------------------------------------------------- #
# Version arithmetic. Ported from decide_release_on_merge.py rather than         #
# re-derived, so the two callers cannot disagree about what "ahead" means.       #
# --------------------------------------------------------------------------- #
def parse_final_version(raw: str, *, label: str) -> tuple[int, int, int]:
    """Parse a strict ``X.Y.Z``, tolerating a leading ``v``.

    A pre-release, rc or post version is an error rather than a skip: a repo
    whose version string cannot be read is a repo whose next release is already
    broken, and reporting that as "nothing to do" is the original disease.
    """
    candidate = raw.strip()
    if candidate.startswith("v"):
        candidate = candidate[1:]
    match = _FINAL_SEMVER.match(candidate)
    if match is None:
        msg = (
            f"{label} must be a final X.Y.Z version (got {raw!r}); a "
            "pre-release version never drives an automatic release"
        )
        raise ReleaseTrainConfigError(msg)
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def next_patch(version: str) -> str:
    major, minor, patch = parse_final_version(version, label="version")
    return f"{major}.{minor}.{patch + 1}"


def highest_published(tags: Sequence[str]) -> str:
    """Highest ``vX.Y.Z`` tag, ordered numerically rather than lexically.

    ``v0.4.9`` sorts BEFORE ``v0.4.18``, which a string sort gets backwards and
    which would compare the candidate against the wrong release.
    """
    parsed: list[tuple[tuple[int, int, int], str]] = []
    for tag in tags:
        match = _RELEASE_TAG.match(tag.strip())
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        parsed.append((key, tag.strip()))
    if not parsed:
        return ""
    return max(parsed)[1]


def is_release_bookkeeping_subject(subject: str) -> bool:
    """True when a commit is the release machinery's own bookkeeping."""
    text = subject.strip()
    return any(pattern.match(text) for pattern in _BOOKKEEPING_SUBJECT_PATTERNS)


# --------------------------------------------------------------------------- #
# Policy.                                                                       #
# --------------------------------------------------------------------------- #
def load_policy(path: Path) -> dict[str, ModelRepoReleasePolicy]:
    """Read and VALIDATE the policy file. A malformed entry refuses at load.

    Validated here rather than at use, because a policy that cannot be read is
    a train that would otherwise run against a built-in default -- and a default
    is exactly the silent wrong answer rule 8 exists to prevent.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"release-train policy {path} is unreadable: {exc}"
        raise ReleaseTrainConfigError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"release-train policy {path} is not valid YAML: {exc}"
        raise ReleaseTrainConfigError(msg) from exc

    if not isinstance(raw, dict) or not isinstance(raw.get("repos"), dict):
        msg = f"release-train policy {path} carries no 'repos' mapping"
        raise ReleaseTrainConfigError(msg)

    policies: dict[str, ModelRepoReleasePolicy] = {}
    for repo, entry in raw["repos"].items():
        policies[str(repo)] = _parse_entry(str(repo), entry, path)
    if not policies:
        msg = f"release-train policy {path} declares no repos"
        raise ReleaseTrainConfigError(msg)
    return policies


def _parse_entry(repo: str, entry: Any, path: Path) -> ModelRepoReleasePolicy:
    if not isinstance(entry, dict):
        msg = f"{path}: repo {repo} is not a mapping"
        raise ReleaseTrainConfigError(msg)

    def _required(key: str) -> str:
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            msg = f"{path}: repo {repo} is missing a non-empty {key!r}"
            raise ReleaseTrainConfigError(msg)
        return value.strip()

    paths = entry.get("release_relevant_paths")
    if not isinstance(paths, list) or not paths:
        msg = (
            f"{path}: repo {repo} must declare a non-empty "
            "'release_relevant_paths' list; an empty one would make every "
            "commit releasable or none of them, silently"
        )
        raise ReleaseTrainConfigError(msg)

    try:
        lab_evidence = EnumLabEvidence(_required("lab_evidence"))
        mode = EnumTrainMode(_required("mode"))
    except ValueError as exc:
        msg = f"{path}: repo {repo} declares an unknown value: {exc}"
        raise ReleaseTrainConfigError(msg) from exc

    lab_note = str(entry.get("lab_evidence_note", "") or "").strip()
    if lab_evidence is EnumLabEvidence.NONE and not lab_note:
        msg = (
            f"{path}: repo {repo} declares lab_evidence 'none' with no "
            "'lab_evidence_note'. The declaration IS the record of the gap, so "
            "an empty one records nothing."
        )
        raise ReleaseTrainConfigError(msg)

    mode_note = str(entry.get("mode_note", "") or "").strip()
    if mode is EnumTrainMode.REPORT_ONLY and not mode_note:
        msg = (
            f"{path}: repo {repo} is 'report_only' with no 'mode_note'. A repo "
            "the train will never cut has a reason, and the reason is what a "
            "reader needs in order to change it."
        )
        raise ReleaseTrainConfigError(msg)

    return ModelRepoReleasePolicy(
        repo=repo,
        package=_required("package"),
        default_branch=_required("default_branch"),
        release_relevant_paths=tuple(str(p) for p in paths),
        lab_evidence=lab_evidence,
        lab_evidence_note=lab_note,
        mode=mode,
        mode_note=mode_note,
    )


def policy_for(
    policies: dict[str, ModelRepoReleasePolicy], repo: str
) -> ModelRepoReleasePolicy:
    """Look up a repo, refusing rather than inventing a default for it."""
    try:
        return policies[repo]
    except KeyError as exc:
        known = ", ".join(sorted(policies)) or "(none)"
        msg = (
            f"repo {repo!r} is not declared in the release-train policy "
            f"(declared: {known}). Refusing to run a train against a repo whose "
            "release-relevant paths and lab-evidence rule nobody has stated."
        )
        raise ReleaseTrainConfigError(msg) from exc


# --------------------------------------------------------------------------- #
# Fact collection, from a local clone.                                          #
# --------------------------------------------------------------------------- #
def _git(clone: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(clone), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip().splitlines()
        detail = stderr[-1] if stderr else f"exit {completed.returncode}"
        msg = f"git {' '.join(args)} in {clone}: {detail}"
        raise ReleaseTrainConfigError(msg)
    return completed.stdout


def collect_repo_facts(policy: ModelRepoReleasePolicy, clone: Path) -> ModelRepoFacts:
    """Read the repo's own history. Every failure raises rather than returning 0.

    A collector that reports zero unreleased commits because it could not read
    the repository is a clean bill of health for a repo nobody looked at.
    """
    head_ref = f"origin/{policy.default_branch}"
    head_sha = _git(clone, "rev-parse", head_ref).strip()

    tags = [line.strip() for line in _git(clone, "tag", "--list", "v*").splitlines()]
    latest_tag = highest_published(tags)

    pyproject_raw = _git(clone, "show", f"{head_ref}:pyproject.toml")
    try:
        parsed = tomllib.loads(pyproject_raw)
    except tomllib.TOMLDecodeError as exc:
        msg = f"{policy.repo}: pyproject.toml at {head_ref} does not parse: {exc}"
        raise ReleaseTrainConfigError(msg) from exc
    version = parsed.get("project", {}).get("version")
    if not isinstance(version, str) or not version.strip():
        msg = (
            f"{policy.repo}: pyproject.toml at {head_ref} declares no [project].version"
        )
        raise ReleaseTrainConfigError(msg)

    span = f"{latest_tag}..{head_ref}" if latest_tag else head_ref
    log = _git(
        clone,
        "log",
        "--no-merges",
        "--format=%s",
        span,
        "--",
        *policy.release_relevant_paths,
    )
    subjects = tuple(
        line.strip()
        for line in log.splitlines()
        if line.strip() and not is_release_bookkeeping_subject(line)
    )

    return ModelRepoFacts(
        repo=policy.repo,
        latest_tag=latest_tag,
        dev_version=version.strip(),
        dev_head_sha=head_sha,
        unreleased_count=len(subjects),
        unreleased_subjects=subjects,
    )


# --------------------------------------------------------------------------- #
# The lab-evidence premise.                                                     #
# --------------------------------------------------------------------------- #
def classify_lab_receipt(
    repo: str,
    sha: str,
    *,
    list_artifacts: Callable[[str, str], list[dict[str, Any]]],
    download_receipt: Callable[[str, int], Any],
) -> tuple[EnumTrainReason | None, str]:
    """Resolve the compose-dev receipt for one sha into a reason, or None on PASS.

    ``None`` means the premise holds. Every other return names WHICH way it did
    not, because the four cases belong to four different owners.

    The receipt is looked up in the repo the sha belongs to, by exact artifact
    name, because a paginated listing returns a false zero and a failed rebuild
    is precisely when a receipt should exist.
    """
    lane = lab_pass_receipt.EnumLabLane.COMPOSE_DEV
    full_repo = f"OmniNode-ai/{repo}"
    name = lab_pass_receipt.artifact_name(lane, sha)

    try:
        artifacts = list_artifacts(full_repo, name)
    except Exception as exc:  # noqa: BLE001 - rule 16: an unread surface is a finding
        return (
            EnumTrainReason.LAB_RECEIPT_UNREADABLE,
            f"the artifact query for {name} failed: {exc}",
        )
    if not artifacts:
        return (
            EnumTrainReason.LAB_RECEIPT_ABSENT,
            f"no artifact named {name} exists in {full_repo}; the sha has not "
            "been exercised on the compose dev lane, or its rebuild never emitted",
        )

    # Newest first: a re-run of the emitting job supersedes an earlier attempt,
    # the same ordering the rule-24(b) reader applies one layer down.
    artifacts = sorted(
        artifacts, key=lambda a: str(a.get("created_at", "")), reverse=True
    )
    try:
        receipt = download_receipt(full_repo, int(artifacts[0]["id"]))
    except Exception as exc:  # noqa: BLE001 - same reasoning
        return (
            EnumTrainReason.LAB_RECEIPT_UNREADABLE,
            f"artifact {name} could not be read: {exc}",
        )

    if receipt.sha != sha:
        return (
            EnumTrainReason.LAB_RECEIPT_NAME_PAYLOAD_DISAGREE,
            f"artifact {name} carries sha {receipt.sha}; the name and the "
            "payload disagree, so neither can be trusted about this commit",
        )
    if receipt.lane is not lane:
        return (
            EnumTrainReason.LAB_RECEIPT_NAME_PAYLOAD_DISAGREE,
            f"artifact {name} carries lane {receipt.lane.value}; the name and "
            "the payload disagree",
        )
    if receipt.result is not lab_pass_receipt.EnumLabPassResult.PASS:
        failing = ", ".join(
            check.name
            for check in receipt.checks
            if check.outcome is not lab_pass_receipt.EnumLabPassCheckOutcome.PASS
        )
        return (
            EnumTrainReason.LAB_RECEIPT_FAIL,
            f"the compose dev lane recorded {receipt.result.value} for this sha; "
            f"checks not passing: {failing or '(none named)'}",
        )
    return None, f"{name} records PASS on the compose dev lane"


# --------------------------------------------------------------------------- #
# The decision.                                                                 #
# --------------------------------------------------------------------------- #
def decide(
    *,
    policy: ModelRepoReleasePolicy,
    facts: ModelRepoFacts,
    list_artifacts: Callable[[str, str], list[dict[str, Any]]] | None = None,
    download_receipt: Callable[[str, int], Any] | None = None,
) -> ModelTrainDecision:
    """Decide whether this repo cuts tonight, and say why either way.

    The version is resolved FIRST so a malformed ``[project].version`` refuses
    rather than being masked by a later skip.
    """
    list_artifacts = list_artifacts or lab_pass_receipt.list_artifacts
    download_receipt = download_receipt or lab_pass_receipt.download_receipt

    def _build(
        verdict: EnumTrainVerdict,
        reason: EnumTrainReason,
        detail: str,
        *,
        candidate_version: str = "",
        needs_bump: bool = False,
    ) -> ModelTrainDecision:
        return ModelTrainDecision(
            repo=policy.repo,
            verdict=verdict,
            reason=reason,
            detail=detail,
            base_branch=policy.default_branch,
            candidate_sha=facts.dev_head_sha,
            candidate_version=candidate_version,
            latest_tag=facts.latest_tag,
            dev_version=facts.dev_version,
            unreleased_count=facts.unreleased_count,
            needs_bump=needs_bump,
        )

    try:
        dev_tuple = parse_final_version(facts.dev_version, label="[project].version")
    except ReleaseTrainConfigError as exc:
        return _build(
            EnumTrainVerdict.REFUSE, EnumTrainReason.VERSION_UNREADABLE, str(exc)
        )

    dev_normalized = ".".join(str(part) for part in dev_tuple)
    if facts.latest_tag:
        try:
            latest_tuple = parse_final_version(
                facts.latest_tag, label="latest published tag"
            )
        except ReleaseTrainConfigError as exc:
            return _build(
                EnumTrainVerdict.REFUSE, EnumTrainReason.VERSION_UNREADABLE, str(exc)
            )
        needs_bump = dev_tuple <= latest_tuple
        candidate_version = (
            next_patch(facts.latest_tag) if needs_bump else dev_normalized
        )
    else:
        needs_bump = False
        candidate_version = dev_normalized

    if facts.unreleased_count == 0:
        return _build(
            EnumTrainVerdict.SKIP,
            EnumTrainReason.NO_UNRELEASED_RELEASE_RELEVANT_WORK,
            f"no commit between {facts.latest_tag or '(no tag)'} and "
            f"{facts.dev_head_sha[:12]} touches "
            f"{', '.join(policy.release_relevant_paths)}",
            candidate_version=candidate_version,
            needs_bump=needs_bump,
        )

    if policy.mode is EnumTrainMode.REPORT_ONLY:
        return _build(
            EnumTrainVerdict.SKIP,
            EnumTrainReason.MODE_REPORT_ONLY,
            f"{facts.unreleased_count} unreleased release-relevant commit(s); "
            f"the train reports this repo and does not cut it: {policy.mode_note}",
            candidate_version=candidate_version,
            needs_bump=needs_bump,
        )

    if policy.lab_evidence is EnumLabEvidence.NONE:
        return _build(
            EnumTrainVerdict.CUT,
            EnumTrainReason.UNRELEASED_WORK_NO_LAB_SURFACE_DECLARED,
            f"{facts.unreleased_count} unreleased release-relevant commit(s); "
            f"this repo declares no lab-evidence surface: {policy.lab_evidence_note}",
            candidate_version=candidate_version,
            needs_bump=needs_bump,
        )

    reason, detail = classify_lab_receipt(
        policy.repo,
        facts.dev_head_sha,
        list_artifacts=list_artifacts,
        download_receipt=download_receipt,
    )
    if reason is not None:
        return _build(
            EnumTrainVerdict.SKIP,
            reason,
            detail,
            candidate_version=candidate_version,
            needs_bump=needs_bump,
        )
    return _build(
        EnumTrainVerdict.CUT,
        EnumTrainReason.UNRELEASED_WORK_LAB_PROVEN,
        f"{facts.unreleased_count} unreleased release-relevant commit(s); {detail}",
        candidate_version=candidate_version,
        needs_bump=needs_bump,
    )


# --------------------------------------------------------------------------- #
# Reporting.                                                                    #
# --------------------------------------------------------------------------- #
def render_report_json(decisions: Sequence[ModelTrainDecision]) -> str:
    payload = {
        "report_version": "release_train_decision.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "cut_count": sum(1 for d in decisions if d.verdict is EnumTrainVerdict.CUT),
        "refuse_count": sum(
            1 for d in decisions if d.verdict is EnumTrainVerdict.REFUSE
        ),
        "decisions": [d.to_dict() for d in decisions],
    }
    return json.dumps(payload, indent=2)


def render_report_human(decisions: Sequence[ModelTrainDecision]) -> str:
    """One row per repo, every row naming its commit.

    A verdict with no commit named is unactionable, and the whole point of a
    sha-keyed premise is that the answer is about one commit.
    """
    lines = [
        "| repo | verdict | version | sha | reason |",
        "| -- | -- | -- | -- | -- |",
    ]
    for d in decisions:
        lines.append(
            f"| {d.repo} | {d.verdict.value} | {d.candidate_version or '-'} | "
            f"`{d.candidate_sha[:12]}` | {d.reason.value} |"
        )
    lines.append("")
    for d in decisions:
        lines.append(f"**{d.repo}** — {d.verdict.value} ({d.reason.value}): {d.detail}")
    return "\n".join(lines)


def render_changelog_entry(
    *, package: str, version: str, previous_tag: str, subjects: Sequence[str]
) -> str:
    """The release PR's whole diff: one prepended changelog block.

    Mirrors the shape the hand-cut releases use, because the release PR is read
    by people and a generated one that looks different reads as a different kind
    of change.
    """
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    since = previous_tag or "the first commit"
    lines = [
        f"## v{version} ({today})",
        "",
        "### Release",
        f"- Cut {package} from dev at {version} by the scheduled release train.",
        f"- {len(subjects)} release-relevant commit(s) merged since {since}.",
        "- Opened by the release train, which cuts only when the repo's declared "
        "lab-evidence premise holds for the exact candidate commit.",
        "",
        f"### Included Since {since}",
    ]
    lines.extend(f"- {subject}" for subject in subjects)
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI.                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="decide, for every declared repo")
    plan.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    plan.add_argument(
        "--clones-dir",
        type=Path,
        required=True,
        help="directory holding one clone per repo, named after the repo",
    )
    plan.add_argument(
        "--repos",
        default="",
        help="comma-separated subset to decide; empty means every declared repo",
    )
    plan.add_argument("--out", type=Path, default=None)

    # The workflow needs the declared set before it can clone anything, and it
    # needs it from THIS module so the listing and the decision cannot disagree.
    # It is a subcommand rather than an inline importlib heredoc in the YAML
    # because that heredoc was a second, subtly different loader for a module
    # shipped right beside it: it omitted `sys.modules[name] = module`, so
    # `@dataclass` resolved its defining module to None and the first dispatched
    # run of the train (35247876101) died at import before deciding anything.
    # A CLI surface is exercised by the same tests as the rest of the module.
    declared = sub.add_parser(
        "declared", help="print every declared repo name, one per line"
    )
    declared.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)

    changelog = sub.add_parser("changelog", help="render one release changelog block")
    changelog.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    changelog.add_argument("--repo", required=True)
    changelog.add_argument("--clone", type=Path, required=True)
    changelog.add_argument("--version", required=True)

    return parser


def _selected(policies: dict[str, ModelRepoReleasePolicy], raw: str) -> list[str]:
    if not raw.strip():
        return sorted(policies)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    for name in names:
        policy_for(policies, name)
    return names


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        policies = load_policy(args.policy)
    except ReleaseTrainConfigError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    if args.command == "declared":
        # Sorted, so the clone order is stable run to run and a diff of two
        # runs' logs is about what changed rather than about dict ordering.
        for name in sorted(policies):
            print(name)
        return 0

    if args.command == "changelog":
        try:
            policy = policy_for(policies, args.repo)
            facts = collect_repo_facts(policy, args.clone)
        except ReleaseTrainConfigError as exc:
            print(f"::error::{exc}", file=sys.stderr)
            return 2
        print(
            render_changelog_entry(
                package=policy.package,
                version=args.version,
                previous_tag=facts.latest_tag,
                subjects=facts.unreleased_subjects,
            )
        )
        return 0

    try:
        names = _selected(policies, args.repos)
    except ReleaseTrainConfigError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    decisions: list[ModelTrainDecision] = []
    for name in names:
        policy = policies[name]
        clone = args.clones_dir / name
        try:
            facts = collect_repo_facts(policy, clone)
        except ReleaseTrainConfigError as exc:
            decisions.append(
                ModelTrainDecision(
                    repo=name,
                    verdict=EnumTrainVerdict.REFUSE,
                    reason=EnumTrainReason.FACTS_UNREADABLE,
                    detail=str(exc),
                    base_branch=policy.default_branch,
                    candidate_sha="",
                    candidate_version="",
                    latest_tag="",
                    dev_version="",
                    unreleased_count=0,
                    needs_bump=False,
                )
            )
            continue
        decisions.append(decide(policy=policy, facts=facts))

    report = render_report_json(decisions)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
    print(report)
    print(render_report_human(decisions), file=sys.stderr)

    return 2 if any(d.verdict is EnumTrainVerdict.REFUSE for d in decisions) else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
