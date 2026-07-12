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
evaluate_repo = _lib.evaluate_repo
evaluate_all = _lib.evaluate_all
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
