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
        branch_head_at="2026-07-22T18:00:00Z",  # ~28h before _DEPLOY_NOW
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
def test_deploy_gap_threshold_is_age_of_newest_commit() -> None:
    """Below the stale-hours threshold -> no fire (buffer against nagging
    seconds after a merge); at/above it -> fires.
    """
    below = _deploy_target(
        branch_head_sha="ccccccc3333333333333333333333333333333",
        branch_head_at="2026-07-23T19:00:00Z",  # 3h before _DEPLOY_NOW
        deploy_affecting_paths_changed=(".github/workflows/deploy-onex-dev.yml",),
    )
    at = _deploy_target(
        branch_head_sha="ccccccc3333333333333333333333333333333",
        branch_head_at="2026-07-23T18:00:00Z",  # 4h before _DEPLOY_NOW
        deploy_affecting_paths_changed=(".github/workflows/deploy-onex-dev.yml",),
    )
    assert "DEPLOY_STALE_VS_DEV" not in _deploy_codes(below)
    assert "DEPLOY_STALE_VS_DEV" in _deploy_codes(at)


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
