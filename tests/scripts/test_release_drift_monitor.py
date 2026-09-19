# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the RT-7 release/deploy drift monitor pure evaluator (OMN-14466).

Two obligations, per feedback_prove_red_against_exists_but_wrong:

1. RED reproduces on the CURRENT live drift. Each ``test_live_*`` feeds the
   evaluator the EXACT facts observed on 2026-07-12 (infra PyPI 0.36.1 vs tag
   v0.38.3 vs main 0.38.4; core main 0.46.6 behind PyPI/tag 0.46.7; OCC main
   still pinning the omnibase-core git override while dev is clean; infra
   release.yml last run = failure) and asserts the monitor FIRES.
2. GREEN is discriminating, not vacuous. ``test_aligned_repo_is_green`` proves
   an aligned repo produces zero findings, and its sibling flips one field to
   confirm the same check goes RED -- a green here means "checked and matched",
   not "couldn't see anything".

Deploy-gap tests (OMN-14994) apply the same two obligations to the deploy-gap
check: ``test_live_omninode_infra_*`` reconstructs the 2026-07-22/23 incident
(three fixes -- omninode_infra PRs #619/#620/#622 -- merged to dev and left
undeployed for ~28h because deploy-onex-dev.yml's push trigger only fires on
main/m7/**, never dev) and asserts it fires; the aligned/threshold tests prove
the green is discriminating.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = _REPO_ROOT / "scripts" / "release_drift_monitor_lib.py"

# A fixed "now" so the age-based findings are deterministic. Matches the
# 2026-07-12 observation date used for the live-drift fixtures below.
_NOW = datetime(2026, 7, 12, 17, 0, tzinfo=UTC)


def _load_lib() -> Any:
    mod_name = "release_drift_monitor_lib_under_test"
    spec = importlib.util.spec_from_file_location(mod_name, _LIB)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


_lib = _load_lib()

RepoFacts = _lib.RepoFacts
WorkflowRun = _lib.WorkflowRun
DriftThresholds = _lib.DriftThresholds
DeployFacts = _lib.DeployFacts
evaluate_repo = _lib.evaluate_repo
evaluate_all = _lib.evaluate_all
evaluate_deploy_target = _lib.evaluate_deploy_target
evaluate_deploy_targets = _lib.evaluate_deploy_targets
exit_code_for = _lib.exit_code_for
blind_verdicts_for_deploy_target = _lib.blind_verdicts_for_deploy_target


def _codes(facts: Any, thresholds: DriftThresholds | None = None) -> set[str]:
    findings = evaluate_repo(facts, thresholds or DriftThresholds(), now=_NOW)
    return {f.code for f in findings}


# --------------------------------------------------------------------------- #
# 1. RED reproduces on the CURRENT live drift (2026-07-12 observations).       #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_live_infra_drift_fires() -> None:
    """infra: PyPI 0.36.1 behind tag v0.38.3; main 0.38.4 never tagged; release failed."""
    facts = RepoFacts(
        repo="omnibase_infra",
        pypi_package="omnibase-infra",
        pypi_version="0.36.1",
        latest_tag="v0.38.3",
        main_version="0.38.4",
        dev_version="0.38.4",
        dev_ahead_commits=294,
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="failure",
            created_at="2026-05-27T21:12:23Z",
        ),
    )
    codes = _codes(facts)
    # The six-week-stuck-PyPI signal and its siblings must all fire.
    assert "PYPI_BEHIND_TAG" in codes  # 0.36.1 < v0.38.3
    assert "MAIN_AHEAD_OF_TAG" in codes  # 0.38.4 never tagged
    assert "RELEASE_WORKFLOW_FAILED" in codes  # last release.yml run failed
    assert "RELEASE_TRAIN_STALLED" in codes  # main>tag and release run 46d stale
    assert "DEV_AHEAD_OF_MAIN" in codes  # 294 >= 25
    # main is AHEAD of the released set here, so this must NOT fire.
    assert "MAIN_BEHIND_RELEASED" not in codes


@pytest.mark.unit
def test_live_occ_stale_override_fires() -> None:
    """OCC: main still pins the omnibase-core git override that dev removed."""
    facts = RepoFacts(
        repo="onex_change_control",
        pypi_package=None,  # not published to PyPI -> no PyPI checks
        pypi_version=None,
        latest_tag="v0.5.1",
        main_version="0.5.1",
        dev_version="0.5.1",
        main_git_overrides=("omnibase-core",),
        dev_git_overrides=(),
        release_run=WorkflowRun(name="release.yml", exists=False),
    )
    findings = evaluate_repo(facts, DriftThresholds(), now=_NOW)
    codes = {f.code for f in findings}
    assert "MAIN_STALE_UV_OVERRIDE" in codes
    stale = next(f for f in findings if f.code == "MAIN_STALE_UV_OVERRIDE")
    assert stale.signature == "onex_change_control:stale-uv-override:omnibase-core"
    # No PyPI package -> no PyPI-based finding, no crash.
    assert "PYPI_BEHIND_TAG" not in codes


@pytest.mark.unit
def test_live_core_main_behind_released_fires() -> None:
    """core: main 0.46.6 is behind PyPI 0.46.7 / tag v0.46.7 (stale main lineage)."""
    facts = RepoFacts(
        repo="omnibase_core",
        pypi_package="omnibase-core",
        pypi_version="0.46.7",
        latest_tag="v0.46.7",
        main_version="0.46.6",
        dev_version="0.46.8",
        dev_ahead_commits=170,
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="success",
            created_at="2026-07-10T00:00:00Z",
        ),
    )
    codes = _codes(facts)
    assert "MAIN_BEHIND_RELEASED" in codes  # 0.46.6 < 0.46.7
    # PyPI equals the tag here, so the publish is NOT behind the tag.
    assert "PYPI_BEHIND_TAG" not in codes
    # main is not ahead of the tag, so no missing-tag / stalled finding.
    assert "MAIN_AHEAD_OF_TAG" not in codes
    assert "RELEASE_TRAIN_STALLED" not in codes


# --------------------------------------------------------------------------- #
# 2. GREEN is discriminating (aligned passes; a one-field break fires).        #
# --------------------------------------------------------------------------- #
def _aligned_facts(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "repo": "omnibase_example",
        "pypi_package": "omnibase-example",
        "pypi_version": "1.4.0",
        "latest_tag": "v1.4.0",
        "main_version": "1.4.0",
        "dev_version": "1.4.0",
        "dev_ahead_commits": 0,
        "main_git_overrides": (),
        "dev_git_overrides": (),
        "release_run": WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="success",
            created_at="2026-07-11T00:00:00Z",
        ),
    }
    base.update(overrides)
    return RepoFacts(**base)


@pytest.mark.unit
def test_aligned_repo_is_green() -> None:
    """Negative control: fully aligned repo produces zero findings."""
    assert _codes(_aligned_facts()) == set()


@pytest.mark.unit
def test_aligned_repo_goes_red_when_pypi_slips() -> None:
    """Same aligned fixture, PyPI dropped one patch behind the tag -> fires.

    Proves the green above is 'checked and matched', not vacuous absence.
    """
    assert "PYPI_BEHIND_TAG" in _codes(_aligned_facts(pypi_version="1.3.9"))


@pytest.mark.unit
def test_aligned_repo_goes_red_when_override_added() -> None:
    assert "MAIN_STALE_UV_OVERRIDE" in _codes(
        _aligned_facts(main_git_overrides=("omnibase-core",), dev_git_overrides=())
    )


# --------------------------------------------------------------------------- #
# Threshold / edge behaviour                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_dev_ahead_backlog_threshold() -> None:
    below = _aligned_facts(dev_ahead_commits=24)
    at = _aligned_facts(dev_ahead_commits=25)
    assert "DEV_AHEAD_OF_MAIN" not in _codes(below)
    assert "DEV_AHEAD_OF_MAIN" in _codes(at)


@pytest.mark.unit
def test_release_stalled_needs_main_ahead_and_stale_run() -> None:
    # main ahead of tag but the release run is RECENT -> not stalled.
    recent = _aligned_facts(
        main_version="1.5.0",
        latest_tag="v1.4.0",
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="success",
            created_at="2026-07-11T00:00:00Z",  # 1d before _NOW
        ),
    )
    assert "RELEASE_TRAIN_STALLED" not in _codes(recent)
    # main ahead of tag AND the run is stale -> stalled.
    stale = _aligned_facts(
        main_version="1.5.0",
        latest_tag="v1.4.0",
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="success",
            created_at="2026-06-01T00:00:00Z",  # ~41d before _NOW
        ),
    )
    assert "RELEASE_TRAIN_STALLED" in _codes(stale)


@pytest.mark.unit
def test_shared_override_on_both_branches_is_not_drift() -> None:
    """An override present on BOTH main and dev is intentional, not stale."""
    facts = _aligned_facts(
        main_git_overrides=("omnibase-core",),
        dev_git_overrides=("omnibase-core",),
    )
    assert "MAIN_STALE_UV_OVERRIDE" not in _codes(facts)


# --------------------------------------------------------------------------- #
# Aggregation + fail-loud exit codes                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_evaluate_all_aggregates_and_exit_code_is_red() -> None:
    infra = RepoFacts(
        repo="omnibase_infra",
        pypi_package="omnibase-infra",
        pypi_version="0.36.1",
        latest_tag="v0.38.3",
        main_version="0.38.4",
        dev_version="0.38.4",
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="failure",
            created_at="2026-05-27T21:12:23Z",
        ),
    )
    report = evaluate_all([infra, _aligned_facts()], DriftThresholds(), now=_NOW)
    assert report.diverged is True
    assert report.repos_checked == 2
    assert exit_code_for(report) == 1
    # Findings are sorted P1 before P2.
    severities = [f.severity for f in report.findings]
    assert severities == sorted(severities, key={"P1": 0, "P2": 1, "P3": 2}.get)


@pytest.mark.unit
def test_aligned_only_exit_code_is_zero() -> None:
    report = evaluate_all([_aligned_facts()], DriftThresholds(), now=_NOW)
    assert report.diverged is False
    assert report.blind is False
    assert exit_code_for(report) == 0


@pytest.mark.unit
def test_probe_error_is_not_silently_green() -> None:
    """A blind run (probe errors, no findings) must exit 2, never 0."""
    blind = _aligned_facts(probe_errors=("tags: gh api failed",))
    report = evaluate_all([blind], DriftThresholds(), now=_NOW)
    assert report.diverged is False
    assert report.blind is True
    assert exit_code_for(report) == 2  # blind, not clean


# --------------------------------------------------------------------------- #
# Deploy-gap check (OMN-14994): "merged but never deployed".                   #
# --------------------------------------------------------------------------- #
# A fixed "now" matching the OMN-14938/OMN-14198 incident window (2026-07-23,
# ~28h after the #619/#620/#622 fixes started landing on 2026-07-22).
_DEPLOY_NOW = datetime(2026, 7, 23, 22, 0, tzinfo=UTC)


def _deploy_codes(facts: Any, thresholds: DriftThresholds | None = None) -> set[str]:
    findings = evaluate_deploy_target(
        facts, thresholds or DriftThresholds(), now=_DEPLOY_NOW
    )
    return {f.code for f in findings}


def _deploy_target(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "repo": "omninode_infra",
        "target": "onex-dev",
        "branch": "dev",
        "workflow": "deploy-onex-dev.yml",
        "last_success": WorkflowRun(
            name="deploy-onex-dev.yml",
            exists=True,
            conclusion="success",
            created_at="2026-07-20T09:00:00Z",
            head_sha="aaaaaaa1111111111111111111111111111111",
        ),
        "branch_head_sha": "aaaaaaa1111111111111111111111111111111",
        "branch_head_at": "2026-07-20T09:00:00Z",
        "deploy_affecting_paths_changed": (),
        "probe_errors": (),
    }
    base.update(overrides)
    return DeployFacts(**base)


@pytest.mark.unit
def test_live_omninode_infra_deploy_gap_fires() -> None:
    """Reconstructs the OMN-14938/OMN-14198 incident: #619/#620/#622 merged to
    dev, touching deploy-onex-dev.yml itself, and sat ~28h with no successful
    deploy run past their head -- the last successful run predates all three.
    """
    facts = _deploy_target(
        last_success=WorkflowRun(
            name="deploy-onex-dev.yml",
            exists=True,
            conclusion="success",
            created_at="2026-07-20T09:00:00Z",
            head_sha="aaaaaaa1111111111111111111111111111111",
        ),
        branch_head_sha="ddddddd4444444444444444444444444444444",  # #622 merge
        branch_head_at="2026-07-22T18:00:00Z",  # commit landed ~28h before _DEPLOY_NOW
        # last_success stays the fixture default (2026-07-20T09:00:00Z, ~85h
        # before _DEPLOY_NOW) -- OMN-14998: the fire decision is keyed on
        # that, not branch_head_at, so branch_head_at recency is deliberately
        # NOT what makes this fire (see test_deploy_gap_fires_on_stale_clock_
        # even_with_recent_commits below for the discriminating case).
        deploy_affecting_paths_changed=(
            ".github/workflows/deploy-onex-dev.yml",
            "tests/k8s/test_deploy_onex_dev_workflow.py",
        ),
    )
    codes = _deploy_codes(facts)
    assert "DEPLOY_STALE_VS_DEV" in codes
    finding = next(
        f
        for f in evaluate_deploy_target(facts, DriftThresholds(), now=_DEPLOY_NOW)
        if f.code == "DEPLOY_STALE_VS_DEV"
    )
    assert finding.severity == "P1"
    assert finding.signature == "omninode_infra:onex-dev:deploy-stale-vs-dev"
    assert "deploy-onex-dev.yml" in finding.detail


@pytest.mark.unit
def test_live_omninode_infra_missing_successful_deploy_fires() -> None:
    """A stretch of the same incident: multiple dispatches (OMN-14929 kustomize
    wall, OMN-14938 valueFrom wall) failed outright -- zero successful runs to
    compare against. MISSING_SUCCESSFUL_DEPLOY must fire unconditionally.
    """
    facts = _deploy_target(
        last_success=WorkflowRun(name="deploy-onex-dev.yml", exists=False)
    )
    codes = _deploy_codes(facts)
    assert codes == {"MISSING_SUCCESSFUL_DEPLOY"}


@pytest.mark.unit
def test_aligned_deploy_target_is_green() -> None:
    """Negative control: branch HEAD == last successful run's head_sha -> clean."""
    assert _deploy_codes(_deploy_target()) == set()


@pytest.mark.unit
def test_deploy_gap_requires_deploy_affecting_paths() -> None:
    """Branch is ahead of the last successful run, but only docs/tests changed
    (no deploy-affecting path) -> must NOT fire. Proves the green above is
    discriminating, not vacuous absence of any ahead-commits.
    """
    facts = _deploy_target(
        branch_head_sha="bbbbbbb2222222222222222222222222222222",
        branch_head_at="2026-07-22T18:00:00Z",
        deploy_affecting_paths_changed=(),  # e.g. only README.md changed
    )
    assert _deploy_codes(facts) == set()


@pytest.mark.unit
def test_deploy_gap_threshold_is_age_of_last_success() -> None:
    """Below the stale-hours threshold -> no fire (buffer against nagging
    seconds after a fresh success); at/above it -> fires.

    OMN-14998: the clock is time-since-last-success, NOT the age of the
    newest commit -- this test varies ``last_success.created_at`` (the
    load-bearing input) while holding ``branch_head_at`` fixed, proving
    branch-HEAD recency is no longer what the threshold responds to.
    """

    def _target(last_success_created_at: str) -> Any:
        return _deploy_target(
            branch_head_sha="ccccccc3333333333333333333333333333333",
            branch_head_at="2026-07-20T09:00:00Z",  # fixed; irrelevant to the fire decision
            last_success=WorkflowRun(
                name="deploy-onex-dev.yml",
                exists=True,
                conclusion="success",
                created_at=last_success_created_at,
                head_sha="aaaaaaa1111111111111111111111111111111",
            ),
            deploy_affecting_paths_changed=(".github/workflows/deploy-onex-dev.yml",),
        )

    below = _target("2026-07-23T19:00:00Z")  # last success 3h before _DEPLOY_NOW
    at = _target("2026-07-23T18:00:00Z")  # last success 4h before _DEPLOY_NOW
    assert "DEPLOY_STALE_VS_DEV" not in _deploy_codes(below)
    assert "DEPLOY_STALE_VS_DEV" in _deploy_codes(at)


@pytest.mark.unit
def test_deploy_gap_fires_on_stale_clock_even_with_recent_commits() -> None:
    """Regression for the OMN-14998 blind spot, reconstructed from the live
    facts this monitor's own diagnosis found (2026-07-23/24): the last
    successful ``deploy-onex-dev.yml`` run was 2026-03-12, but commits --
    including deploy-affecting ones (the OMN-14938/OMN-14198 chain) -- kept
    landing on ``omninode_infra`` dev right up through 2026-07-23, hours
    before this fixture's "now".

    A staleness clock keyed on "age of the branch's current HEAD commit"
    (``branch_head_at``) is reset by every such commit and would report ~1h
    old here, permanently masking a 4+ month deploy gap. Keying on time
    since the last SUCCESSFUL run instead must fire regardless of how
    recently the newest commit landed. This is the exact case the pre-fix
    implementation gets wrong -- it stays green here.
    """
    facts = _deploy_target(
        last_success=WorkflowRun(
            name="deploy-onex-dev.yml",
            exists=True,
            conclusion="success",
            created_at="2026-03-12T12:00:00Z",
            head_sha="0000000000000000000000000000000000000f",
        ),
        branch_head_sha="373e650c6c93239b205025c5eecfe094f3da931",  # live dev HEAD, 2026-07-23
        branch_head_at="2026-07-23T21:00:00Z",  # commit landed 1h before _DEPLOY_NOW
        deploy_affecting_paths_changed=(
            "k8s/onex-dev/networkpolicies.yaml",
            ".github/workflows/deploy-onex-dev.yml",
        ),
    )
    codes = _deploy_codes(facts)
    assert "DEPLOY_STALE_VS_DEV" in codes, (
        "last successful deploy-onex-dev run was 2026-03-12 (>4 months before "
        "_DEPLOY_NOW) with deploy-affecting commits still landing on dev -- "
        "this must fire regardless of how recently the newest commit landed. "
        "A branch-HEAD-keyed clock (age of branch_head_at, 1h old here) "
        "misses this entirely -- that is the OMN-14998 bug this test guards."
    )


@pytest.mark.unit
def test_evaluate_deploy_targets_aggregates_multiple() -> None:
    gapped = _deploy_target(
        branch_head_sha="ddddddd4444444444444444444444444444444",
        branch_head_at="2026-07-22T18:00:00Z",
        deploy_affecting_paths_changed=(".github/workflows/deploy-onex-dev.yml",),
    )
    clean = _deploy_target(repo="other_infra", target="other-dev")
    findings = evaluate_deploy_targets(
        [gapped, clean], DriftThresholds(), now=_DEPLOY_NOW
    )
    assert {f.code for f in findings} == {"DEPLOY_STALE_VS_DEV"}
    assert findings[0].repo == "omninode_infra"


@pytest.mark.unit
def test_evaluate_all_includes_deploy_targets_in_exit_code() -> None:
    """evaluate_all aggregates repo findings + deploy-target findings into one
    report, and a deploy-only gap is enough to flip the exit code RED.
    """
    gapped = _deploy_target(
        branch_head_sha="ddddddd4444444444444444444444444444444",
        branch_head_at="2026-07-22T18:00:00Z",
        deploy_affecting_paths_changed=(".github/workflows/deploy-onex-dev.yml",),
    )
    report = evaluate_all(
        [_aligned_facts()],
        DriftThresholds(),
        now=_DEPLOY_NOW,
        deploy_facts=[gapped],
    )
    assert report.diverged is True
    assert report.repos_checked == 2  # 1 RepoFacts + 1 DeployFacts
    assert exit_code_for(report) == 1
    assert any(f.code == "DEPLOY_STALE_VS_DEV" for f in report.findings)


# --------------------------------------------------------------------------- #
# OMN-18348: a monitor that cannot READ a surface must say so as its own       #
# verdict, never as a finding ABOUT that surface.                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_unreadable_deploy_probe_is_not_a_surface_finding() -> None:
    """Reproduces the live 2026-09-18 report (run 35380688946).

    The minted App token carried ``contents`` only, so reading
    ``deploy-onex-dev.yml`` runs on the PRIVATE omninode_infra repo returned
    ``HTTP 403 Resource not accessible by integration``. The monitor turned
    that unreadable probe into a P1 ``MISSING_SUCCESSFUL_DEPLOY`` -- a claim
    about the deploy target that the monitor had no facts to support.
    """
    facts = _deploy_target(
        last_success=None,
        last_success_readable=False,
        probe_errors=(
            "deploy-onex-dev.yml successful runs: gh: Resource not accessible "
            "by integration (HTTP 403)",
        ),
    )
    assert _deploy_codes(facts) == set()

    verdicts = blind_verdicts_for_deploy_target(facts)
    assert [v.code for v in verdicts] == ["PROBE_UNREADABLE"]
    assert "403" in verdicts[0].detail


@pytest.mark.unit
def test_readable_probe_with_zero_successful_runs_still_fires_missing() -> None:
    """AC1 falsifier: a genuinely absent successful deploy, on a surface the
    monitor CAN read, must still be reported as MISSING_SUCCESSFUL_DEPLOY.

    The read succeeded and returned no successful run at all (the workflow
    exists, has runs, none of them green) -- there is no head_sha to compare
    the branch against. That is a fact about the target, not a visibility gap.
    """
    facts = _deploy_target(
        last_success=WorkflowRun(
            name="deploy-onex-dev.yml", exists=True, conclusion=None, head_sha=None
        ),
        last_success_readable=True,
    )
    assert _deploy_codes(facts) == {"MISSING_SUCCESSFUL_DEPLOY"}
    assert blind_verdicts_for_deploy_target(facts) == []


@pytest.mark.unit
def test_blind_verdict_makes_the_report_blind_and_exits_2() -> None:
    """A blind verdict is never a silent green: the report reports BLIND and
    the process exits 2 even though no surface finding fired.
    """
    facts = _deploy_target(
        last_success=None,
        last_success_readable=False,
        probe_errors=("successful runs: HTTP 403",),
    )
    report = evaluate_all([], DriftThresholds(), now=_NOW, deploy_facts=[facts])
    assert report.findings == ()
    assert report.diverged is False
    assert [v.code for v in report.blind_verdicts] == ["PROBE_UNREADABLE"]
    assert report.blind is True
    assert exit_code_for(report) == 2


# --------------------------------------------------------------------------- #
# OMN-18348: a failed release RUN is not the same fact as a failed PUBLISH.    #
# --------------------------------------------------------------------------- #
_INFRA_0_38_32 = {
    "repo": "omnibase_infra",
    "pypi_package": "omnibase-infra",
    "pypi_version": "0.38.32",
    "latest_tag": "v0.38.32",
    "main_version": "0.38.32",
    "dev_version": "0.38.32",
    "dev_ahead_commits": 0,
}


@pytest.mark.unit
def test_failed_release_run_whose_publish_landed_is_a_cascade_failure() -> None:
    """Reproduces the live 2026-09-18 omnibase_infra finding.

    Release run 35329990280 (v0.38.32) concluded ``failure``, so the monitor
    fired P1 ``RELEASE_WORKFLOW_FAILED`` claiming "The release chain is broken;
    no publish has succeeded since". The publish HAD succeeded: the ``release``
    job was green and PyPI serves 0.38.32, matching tag v0.38.32 -- facts this
    same monitor collects. Only the downstream ``Dependency Cascade / Bump
    omniintelligence`` job failed (OMN-18634 / OMN-18673).
    """
    facts = RepoFacts(
        **_INFRA_0_38_32,
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="failure",
            created_at="2026-09-18T09:31:39Z",
            failed_jobs=("Dependency Cascade / Bump omniintelligence",),
        ),
    )
    findings = evaluate_repo(facts, DriftThresholds(), now=_NOW)
    codes = {f.code for f in findings}
    assert codes == {"RELEASE_CASCADE_FAILED"}
    (finding,) = findings
    assert finding.severity == "P2"
    assert "Dependency Cascade / Bump omniintelligence" in finding.detail
    # The false claim must be gone.
    assert "no publish has succeeded" not in finding.detail


@pytest.mark.unit
def test_failed_release_run_with_pypi_behind_tag_is_still_p1() -> None:
    """Falsifier: when the publish genuinely did NOT land (PyPI behind the
    tag), the P1 RELEASE_WORKFLOW_FAILED verdict must survive unchanged.
    """
    facts = RepoFacts(
        repo="omnibase_infra",
        pypi_package="omnibase-infra",
        pypi_version="0.38.31",
        latest_tag="v0.38.32",
        main_version="0.38.32",
        dev_version="0.38.32",
        dev_ahead_commits=0,
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="failure",
            created_at="2026-09-18T09:31:39Z",
            failed_jobs=("release",),
        ),
    )
    codes = {f.code for f in evaluate_repo(facts, DriftThresholds(), now=_NOW)}
    assert "RELEASE_WORKFLOW_FAILED" in codes
    assert "RELEASE_CASCADE_FAILED" not in codes


@pytest.mark.unit
def test_failed_release_run_without_pypi_facts_is_still_p1() -> None:
    """Fail-loud default: with no published-artifact fact to derive from, the
    monitor cannot conclude the publish landed, so the P1 verdict stands.
    """
    facts = RepoFacts(
        repo="onex_change_control",
        pypi_package=None,
        latest_tag="v0.5.3",
        main_version="0.5.3",
        dev_version="0.5.3",
        dev_ahead_commits=0,
        release_run=WorkflowRun(
            name="release.yml",
            exists=True,
            conclusion="failure",
            created_at="2026-09-18T09:31:39Z",
        ),
    )
    codes = {f.code for f in evaluate_repo(facts, DriftThresholds(), now=_NOW)}
    assert "RELEASE_WORKFLOW_FAILED" in codes


# --------------------------------------------------------------------------- #
# OMN-18348: release-lineage checks presume main tracks what shipped.          #
# --------------------------------------------------------------------------- #
_OCC_LIVE = {
    "repo": "onex_change_control",
    "pypi_package": None,
    "pypi_version": None,
    "latest_tag": "v0.5.3",
    "main_version": "0.5.1",
    "dev_version": "0.5.4",
    "dev_ahead_commits": 7019,
}


@pytest.mark.unit
def test_repo_that_is_not_release_synced_skips_lineage_checks() -> None:
    """onex_change_control carries no release.yml on either branch and its
    ``main`` is a deliberately separate lineage (the CODEOWNERS-gated grants
    anchor), so "main is behind what shipped" and "dev is ahead of main" are
    not drift there -- they are the repo's design. Live on 2026-09-18:
    main 0.5.1, tag v0.5.3, dev 7019 commits ahead.
    """
    facts = RepoFacts(
        **_OCC_LIVE,
        release_synced=False,
        release_synced_note="no release.yml on dev or main; main is a separate lineage",
    )
    codes = {f.code for f in evaluate_repo(facts, DriftThresholds(), now=_NOW)}
    assert "MAIN_BEHIND_RELEASED" not in codes
    assert "DEV_AHEAD_OF_MAIN" not in codes


@pytest.mark.unit
def test_release_synced_repo_with_the_same_facts_still_fires() -> None:
    """Falsifier for the test above: the identical facts on a release-synced
    repo must still fire both lineage findings. The skip is scoped to the
    repo's release model, not a severity floor.
    """
    facts = RepoFacts(**{**_OCC_LIVE, "repo": "omnibase_core"}, release_synced=True)
    codes = {f.code for f in evaluate_repo(facts, DriftThresholds(), now=_NOW)}
    assert "MAIN_BEHIND_RELEASED" in codes
    assert "DEV_AHEAD_OF_MAIN" in codes


@pytest.mark.unit
def test_not_release_synced_repo_still_fires_non_lineage_checks() -> None:
    """The skip is narrow: a stale [tool.uv.sources] override on main is not a
    lineage claim and must still fire on a non-release-synced repo.
    """
    facts = RepoFacts(
        **_OCC_LIVE,
        release_synced=False,
        main_git_overrides=("omnibase-core",),
        dev_git_overrides=(),
    )
    codes = {f.code for f in evaluate_repo(facts, DriftThresholds(), now=_NOW)}
    assert "MAIN_STALE_UV_OVERRIDE" in codes


@pytest.mark.unit
def test_report_records_which_repos_skipped_lineage_and_why() -> None:
    """The skip must be VISIBLE in the report, never silent."""
    facts = RepoFacts(
        **_OCC_LIVE,
        release_synced=False,
        release_synced_note="no release.yml on dev or main",
    )
    report = evaluate_all([facts], DriftThresholds(), now=_NOW)
    assert report.lineage_skipped == (
        "onex_change_control: no release.yml on dev or main",
    )
