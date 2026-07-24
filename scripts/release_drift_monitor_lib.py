#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure drift-evaluation logic for the RT-7 release/deploy drift monitor (OMN-14466).

Given already-gathered facts about each monitored repo -- its PyPI-published
version, latest ``v*`` tag, ``main``/``dev`` pyproject versions, ``[tool.uv.sources]``
git overrides on each branch, dev-ahead-of-main commit count, and the last
``release.yml`` / ``runtime-rebuild-trigger.yml`` workflow run -- this module
decides which drift findings fire.

DEPLOY-GAP CHECK (OMN-14994): a second, independent fact family -- ``DeployFacts``
-- covers the "merged but never deployed" class that ``RepoFacts`` does not: a
cloud deploy workflow (e.g. ``omninode_infra``'s ``deploy-onex-dev.yml``) whose
last SUCCESSFUL run is stale relative to deploy-affecting commits already sitting
on the target branch. This is the exact incident class that left three
production-blocking fixes (PRs #619/#620/#622) merged to ``omninode_infra`` dev
for ~28 hours with nothing red anywhere -- the repo's push-trigger only fires on
``main``/``m7/**``, but ``dev`` is the actual default/everyday branch (OMN-14198).
``evaluate_deploy_target`` / ``evaluate_deploy_targets`` are additive: they reuse
the same ``DriftFinding``/``DriftReport`` shapes and the same friction-emit path
as the PyPI/tag checks above, so no new store or schema is introduced.

WHY A SEPARATE PURE MODULE
--------------------------
The fire/no-fire decision has **no I/O**: every input is a plain frozen
dataclass the CLI shell (``release_drift_monitor.py``) populates from ``gh api``
and the PyPI JSON API. That separation makes the decision unit-testable against
both **EXISTS-but-WRONG** facts (must fire) and **aligned** facts (must stay
green) -- the negative control an enforcement check needs so a green result
proves *"checked and matched"*, not *"couldn't see anything"*.

DESIGN (taxonomy Class-3 / Class-6, docs/plans/2026-07-12-mechanical-release-trains.md
§4 RT-7): every stage between "merged" and "released/running" is a silent
producer -- green and unread. This monitor reads the observable end-state
(PyPI vs tag vs main vs dev, and the workflow that should have moved them) and
fires the moment they diverge, so "PyPI stuck at 0.36.1 for six weeks" is a
same-day RED, not a six-week silence.

FRESHNESS / FAIL-LOUD: a fact the shell could not gather is carried as a
``probe_error`` on the RepoFacts, never dropped. ``exit_code_for`` returns 2
when the monitor could not see (probe errors, no findings) so a blind run is
never mistaken for a clean run -- "stale -> explicit, never silently served as
current".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from packaging.version import InvalidVersion, Version

# --------------------------------------------------------------------------- #
# Severity ordering (used for sorting / summary only; not a gate threshold).   #
# --------------------------------------------------------------------------- #
_SEVERITY_ORDER = {"P1": 0, "P2": 1, "P3": 2}


def parse_version(raw: str | None) -> Version | None:
    """Parse a version string (with or without a leading ``v``); None if unparseable."""
    if not raw:
        return None
    candidate = raw[1:] if raw.startswith("v") else raw
    try:
        return Version(candidate)
    except InvalidVersion:
        return None


def _age_days(iso_ts: str | None, now: datetime) -> float | None:
    """Age in days of an ISO-8601 timestamp relative to ``now``; None if unparseable."""
    if not iso_ts:
        return None
    text = iso_ts.replace("Z", "+00:00")
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return (now - when).total_seconds() / 86400.0


def _age_hours(iso_ts: str | None, now: datetime) -> float | None:
    """Age in hours of an ISO-8601 timestamp relative to ``now``; None if unparseable."""
    days = _age_days(iso_ts, now)
    return None if days is None else days * 24.0


# --------------------------------------------------------------------------- #
# Fact model -- populated by the I/O shell, consumed here with zero I/O.       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkflowRun:
    """The last run of a release/deploy workflow for one repo."""

    name: str
    exists: bool = True
    conclusion: str | None = None  # "success" | "failure" | "cancelled" | None
    created_at: str | None = None  # ISO-8601
    head_sha: str | None = None  # commit the run executed against


@dataclass(frozen=True)
class RepoFacts:
    """Everything the monitor knows about one repo this cycle.

    ``pypi_package`` is None for repos that are intentionally not published to
    PyPI (e.g. onex_change_control) -- PyPI-based checks are then skipped, not
    treated as drift.
    """

    repo: str
    pypi_package: str | None = None
    pypi_version: str | None = None
    latest_tag: str | None = None
    main_version: str | None = None
    dev_version: str | None = None
    dev_ahead_commits: int | None = None
    main_git_overrides: tuple[str, ...] = ()
    dev_git_overrides: tuple[str, ...] = ()
    release_run: WorkflowRun | None = None
    trigger_run: WorkflowRun | None = None
    probe_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DeployFacts:
    """Everything the monitor knows about one deploy target this cycle.

    A "deploy target" is a (repo, branch, workflow) triple whose successful
    runs are the only mechanism that moves code from "merged" to "running" --
    e.g. ``omninode_infra`` dev -> ``deploy-onex-dev.yml`` -> the onex-dev k3s
    namespace. Distinct from ``RepoFacts``, which is about PyPI/tag/main-vs-dev
    lineage, not a live runtime target.
    """

    repo: str
    target: str  # human label, e.g. "onex-dev"
    branch: str  # branch the target deploys from, e.g. "dev"
    workflow: str  # workflow filename, e.g. "deploy-onex-dev.yml"
    last_success: WorkflowRun | None = None
    branch_head_sha: str | None = None
    branch_head_at: str | None = None  # ISO-8601 commit date of branch HEAD
    deploy_affecting_paths_changed: tuple[str, ...] = ()
    probe_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class DriftFinding:
    """One divergence the monitor is firing on."""

    code: str
    severity: str  # "P1" | "P2" | "P3"
    repo: str
    signature: str  # stable dedup key -> friction filename; repeated runs overwrite
    summary: str
    detail: str


@dataclass(frozen=True)
class DriftThresholds:
    """Tunable thresholds for the count/age-based findings."""

    dev_ahead_warn: int = 25  # dev commits ahead of main before DEV_AHEAD_OF_MAIN fires
    release_stale_days: float = (
        7.0  # release-run age (main-ahead-of-tag) before STALLED fires
    )
    deploy_stale_hours: float = (
        4.0  # hours since the last successful deploy run before
        # DEPLOY_STALE_VS_DEV fires (OMN-14998: keyed on last_success.created_at,
        # not branch-HEAD-commit recency) -- a buffer so the check does not nag
        # seconds after a fresh success; the incident this guards against
        # (OMN-14994) sat undeployed ~28h, so 4h surfaces it same-day with ample
        # margin below the incident window.
    )


@dataclass(frozen=True)
class DriftReport:
    """Aggregate result for one monitor cycle."""

    findings: tuple[DriftFinding, ...]
    repos_checked: int
    probe_errors: tuple[str, ...]
    generated_at: str
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)

    @property
    def diverged(self) -> bool:
        return len(self.findings) > 0

    @property
    def blind(self) -> bool:
        """True when the monitor could not gather some facts (partial visibility)."""
        return len(self.probe_errors) > 0


# --------------------------------------------------------------------------- #
# The checks. Each is a small pure predicate over RepoFacts.                   #
# --------------------------------------------------------------------------- #
def evaluate_repo(
    facts: RepoFacts,
    thresholds: DriftThresholds,
    now: datetime | None = None,
) -> list[DriftFinding]:
    """Return every drift finding that fires for one repo."""
    now = now or datetime.now(tz=UTC)
    findings: list[DriftFinding] = []

    pypi = parse_version(facts.pypi_version)
    tag = parse_version(facts.latest_tag)
    main = parse_version(facts.main_version)

    # 1) PyPI behind the latest tag -> a tag was cut but the publish never
    #    landed (stuck/failed publish). This is the "PyPI stuck at 0.36.1 while
    #    v0.38.3 is tagged" case -- the six-week silence.
    if pypi is not None and tag is not None and pypi < tag:
        findings.append(
            DriftFinding(
                code="PYPI_BEHIND_TAG",
                severity="P1",
                repo=facts.repo,
                signature=f"{facts.repo}:pypi-behind-tag",
                summary=(
                    f"{facts.pypi_package} PyPI {pypi} is behind latest tag v{tag}"
                ),
                detail=(
                    f"A v{tag} tag exists but PyPI still serves {pypi}. The "
                    "publish step never landed for at least one release -- "
                    "downstream pins resolving from PyPI run stale code."
                ),
            )
        )

    # 2) main bumped past the latest tag -> a version was set on main that was
    #    never tagged/released (the auto-tag chain stopped firing).
    if main is not None and tag is not None and main > tag:
        findings.append(
            DriftFinding(
                code="MAIN_AHEAD_OF_TAG",
                severity="P2",
                repo=facts.repo,
                signature=f"{facts.repo}:main-ahead-of-tag",
                summary=f"main pyproject {main} was never tagged (latest tag v{tag})",
                detail=(
                    f"main declares {main} but the highest release tag is v{tag}. "
                    "A release was merged as a version bump but no tag/publish "
                    "followed -- the release train did not advance."
                ),
            )
        )

    # 3) main behind the highest released artifact (PyPI or tag) -> stale main
    #    lineage running behind what has already shipped.
    released_candidates = [v for v in (pypi, tag) if v is not None]
    released = max(released_candidates) if released_candidates else None
    if main is not None and released is not None and main < released:
        findings.append(
            DriftFinding(
                code="MAIN_BEHIND_RELEASED",
                severity="P2",
                repo=facts.repo,
                signature=f"{facts.repo}:main-behind-released",
                summary=f"main pyproject {main} is behind released {released}",
                detail=(
                    f"main is at {main} but the highest released version "
                    f"(max of PyPI {pypi} / tag {tag}) is {released}. main "
                    "lineage has not caught up to what shipped."
                ),
            )
        )

    # 4) large unreleased dev->main backlog (the "six weeks of merged, never
    #    released" signal at commit granularity).
    if (
        facts.dev_ahead_commits is not None
        and facts.dev_ahead_commits >= thresholds.dev_ahead_warn
    ):
        findings.append(
            DriftFinding(
                code="DEV_AHEAD_OF_MAIN",
                severity="P3",
                repo=facts.repo,
                signature=f"{facts.repo}:dev-ahead-of-main",
                summary=(
                    f"dev is {facts.dev_ahead_commits} commits ahead of main "
                    f"(>= {thresholds.dev_ahead_warn})"
                ),
                detail=(
                    f"{facts.dev_ahead_commits} commits merged to dev have not "
                    "promoted to main -- a growing unreleased backlog."
                ),
            )
        )

    # 5) a [tool.uv.sources] git override still present on main that dev already
    #    dropped -> main resolves a pinned git SHA while dev runs the released
    #    dep (the OCC omnibase-core rev=f325e759 case).
    stale_overrides = tuple(
        pkg for pkg in facts.main_git_overrides if pkg not in facts.dev_git_overrides
    )
    for pkg in stale_overrides:
        findings.append(
            DriftFinding(
                code="MAIN_STALE_UV_OVERRIDE",
                severity="P1",
                repo=facts.repo,
                signature=f"{facts.repo}:stale-uv-override:{pkg}",
                summary=(
                    f"main carries a stale [tool.uv.sources] git override for "
                    f"{pkg} that dev has removed"
                ),
                detail=(
                    f"main's pyproject pins {pkg} to a git rev via "
                    "[tool.uv.sources] while dev already dropped it. main builds "
                    "resolve the pinned SHA, not the released dep -- the "
                    '"fix merged but not live" residual.'
                ),
            )
        )

    # 6) the release workflow's last run failed outright.
    if (
        facts.release_run is not None
        and facts.release_run.exists
        and facts.release_run.conclusion == "failure"
    ):
        findings.append(
            DriftFinding(
                code="RELEASE_WORKFLOW_FAILED",
                severity="P1",
                repo=facts.repo,
                signature=f"{facts.repo}:release-workflow-failed",
                summary=f"{facts.release_run.name} last run concluded 'failure'",
                detail=(
                    f"The most recent {facts.release_run.name} run "
                    f"({facts.release_run.created_at}) failed. The release chain "
                    "is broken; no publish has succeeded since."
                ),
            )
        )

    # 7) release workflow stale while main is ahead of the tag -> the train has
    #    stalled (a bump landed, but no release run has moved it in N days).
    if (
        main is not None
        and tag is not None
        and main > tag
        and facts.release_run is not None
    ):
        age = _age_days(facts.release_run.created_at, now)
        if (
            facts.release_run.exists
            and age is not None
            and age >= thresholds.release_stale_days
        ):
            findings.append(
                DriftFinding(
                    code="RELEASE_TRAIN_STALLED",
                    severity="P2",
                    repo=facts.repo,
                    signature=f"{facts.repo}:release-train-stalled",
                    summary=(
                        f"main {main} > tag v{tag} but {facts.release_run.name} "
                        f"last ran {age:.0f}d ago (>= {thresholds.release_stale_days:.0f}d)"
                    ),
                    detail=(
                        f"main is ahead of the latest tag yet the release "
                        f"workflow has not run in {age:.0f} days -- the release "
                        "train has stalled with unreleased bumps pending."
                    ),
                )
            )

    return findings


def evaluate_deploy_target(
    facts: DeployFacts,
    thresholds: DriftThresholds,
    now: datetime | None = None,
) -> list[DriftFinding]:
    """Return every deploy-gap finding that fires for one deploy target.

    Two independent checks:

    1. ``MISSING_SUCCESSFUL_DEPLOY`` -- the workflow has never had a
       successful run at all (or does not exist). Fires unconditionally; a
       target with zero known-good deploys is worse than a stale one.
    2. ``DEPLOY_STALE_VS_DEV`` -- deploy-affecting paths changed on the
       target branch since the last successful run's ``head_sha``, and the
       last successful run itself is older than ``thresholds.deploy_stale_hours``.
       Requires ``branch_head_sha != last_success.head_sha`` (something is
       actually ahead) AND a non-empty ``deploy_affecting_paths_changed``
       (the ahead commits actually touch deploy-relevant paths, not just
       docs/tests) AND the age buffer, so a merge from five minutes after a
       fresh success does not nag before the workflow has had a chance to run.

       OMN-14998: the age clock is keyed on ``last_success.created_at``, NOT
       ``branch_head_at`` (the target branch's current HEAD commit date).
       Keying on branch-HEAD recency was the original (wrong) implementation --
       it is reset to ~0h by *any* commit landing on the branch, deploy-affecting
       or not, as long as at least one ahead commit touches a deploy-affecting
       path. That under-reports exactly the failure this check exists to catch:
       a deploy workflow that has not succeeded in months while ordinary commits
       keep landing on ``dev`` looks perpetually fresh under a branch-HEAD clock.
       Keying on ``last_success.created_at`` instead makes staleness a direct,
       monotonically-growing function of "how long has it been since we know
       this target was actually deployed" -- which is the fact this check is
       named for.
    """
    now = now or datetime.now(tz=UTC)
    findings: list[DriftFinding] = []

    if facts.last_success is None or not facts.last_success.exists:
        findings.append(
            DriftFinding(
                code="MISSING_SUCCESSFUL_DEPLOY",
                severity="P1",
                repo=facts.repo,
                signature=f"{facts.repo}:{facts.target}:missing-successful-deploy",
                summary=(f"{facts.workflow} has no successful run for {facts.target}"),
                detail=(
                    f"No successful run of {facts.workflow} was found for the "
                    f"{facts.target} deploy target. Either the workflow has never "
                    "run clean, or the monitor cannot see any run -- either way "
                    "there is no known-good deploy to compare the branch against."
                ),
            )
        )
        return findings

    if (
        facts.branch_head_sha is not None
        and facts.branch_head_sha != facts.last_success.head_sha
        and facts.deploy_affecting_paths_changed
    ):
        # OMN-14998: clock on time-since-last-success, not branch-HEAD-commit
        # recency -- see the OMN-14998 docstring note above.
        age = _age_hours(facts.last_success.created_at, now)
        if age is not None and age >= thresholds.deploy_stale_hours:
            paths_preview = ", ".join(facts.deploy_affecting_paths_changed[:5])
            more = len(facts.deploy_affecting_paths_changed) - 5
            if more > 0:
                paths_preview += f" (+{more} more)"
            findings.append(
                DriftFinding(
                    code="DEPLOY_STALE_VS_DEV",
                    severity="P1",
                    repo=facts.repo,
                    signature=f"{facts.repo}:{facts.target}:deploy-stale-vs-{facts.branch}",
                    summary=(
                        f"{facts.target}: last successful {facts.workflow} run is "
                        f"{age:.1f}h old with deploy-affecting changes on "
                        f"{facts.branch} since then "
                        f"(>= {thresholds.deploy_stale_hours:.0f}h threshold)"
                    ),
                    detail=(
                        f"{facts.branch} HEAD ({facts.branch_head_sha}) is ahead of "
                        f"the last successful {facts.workflow} run "
                        f"({facts.last_success.head_sha}, {facts.last_success.created_at}) "
                        f"and the change set touches deploy-affecting paths: "
                        f"{paths_preview}. The last successful run is {age:.1f}h old "
                        "with no successful run since. Merged code is not running "
                        '-- this is the "merged but never deployed" class '
                        f"(OMN-14994): dispatch {facts.workflow} and confirm via a "
                        "live HTTP readback, not a green apply step alone."
                    ),
                )
            )

    return findings


def evaluate_deploy_targets(
    facts: list[DeployFacts],
    thresholds: DriftThresholds,
    now: datetime | None = None,
) -> list[DriftFinding]:
    """Evaluate every deploy target; findings are NOT pre-sorted (caller sorts)."""
    now = now or datetime.now(tz=UTC)
    findings: list[DriftFinding] = []
    for target_facts in facts:
        findings.extend(evaluate_deploy_target(target_facts, thresholds, now))
    return findings


def evaluate_all(
    facts: list[RepoFacts],
    thresholds: DriftThresholds | None = None,
    now: datetime | None = None,
    deploy_facts: list[DeployFacts] | None = None,
) -> DriftReport:
    """Evaluate every repo (+ every deploy target) and aggregate into one report."""
    thresholds = thresholds or DriftThresholds()
    now = now or datetime.now(tz=UTC)
    deploy_facts = deploy_facts or []
    all_findings: list[DriftFinding] = []
    all_probe_errors: list[str] = []
    for repo_facts in facts:
        all_findings.extend(evaluate_repo(repo_facts, thresholds, now))
        all_probe_errors.extend(
            f"{repo_facts.repo}: {err}" for err in repo_facts.probe_errors
        )
    all_findings.extend(evaluate_deploy_targets(deploy_facts, thresholds, now))
    for target_facts in deploy_facts:
        all_probe_errors.extend(
            f"{target_facts.repo}:{target_facts.target}: {err}"
            for err in target_facts.probe_errors
        )
    all_findings.sort(
        key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.repo, f.code)
    )
    return DriftReport(
        findings=tuple(all_findings),
        repos_checked=len(facts) + len(deploy_facts),
        probe_errors=tuple(all_probe_errors),
        generated_at=now.isoformat(),
        thresholds=thresholds,
    )


def exit_code_for(report: DriftReport) -> int:
    """Map a report to a process exit code (fail-loud).

    * 1 -- divergence found (RED, the fail-loud drift signal)
    * 2 -- the monitor could not see (probe errors) and found no divergence;
           a blind run must never be mistaken for a clean run
    * 0 -- clean: every fact gathered, nothing diverged
    """
    if report.diverged:
        return 1
    if report.blind:
        return 2
    return 0


__all__ = [
    "DeployFacts",
    "DriftFinding",
    "DriftReport",
    "DriftThresholds",
    "RepoFacts",
    "WorkflowRun",
    "evaluate_all",
    "evaluate_deploy_target",
    "evaluate_deploy_targets",
    "evaluate_repo",
    "exit_code_for",
    "parse_version",
]
