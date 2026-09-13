# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18350: a baseline-derived performance threshold, not a magic literal.

`test_benchmark_warm_cache_hit` failed the nightly `Performance Microbenchmarks`
job twice on the omnibase-ci shared runner fleet -- run 34666515741 (p99=36.004ms
vs a 3.0ms threshold) and run 34782693843 (p99=3.006ms vs the same 3.0ms
threshold). Both failures trace to the SAME defect: `p99` computed over exactly
100 samples degenerates to `max(samples)` (index ``int(0.99 * 100) == 99``, the
last element), so a single GC-pause or scheduler-preemption sample on a shared
runner -- not a sustained regression -- fails the gate. The old threshold was
also two unexplained magic literals multiplied together
(`PERF_THRESHOLD_CACHE_HIT_MS = 1.0` times a `PERF_MULTIPLIER` of 2.0 or 3.0
with no stated rationale for either number).

This module is the RED-first proof for the fix: a committed, documented
baseline (`perf_baselines.json`) with a stated margin, plus a median-of-trials
aggregate that a single noisy trial cannot drag past the threshold while a
regression present in every trial still trips it.
"""

from __future__ import annotations

import pytest

from tests.unit.mixins.perf_threshold import (
    load_baseline,
    median_of_trial_percentiles,
    threshold_ms,
)

pytestmark = pytest.mark.unit


def test_baseline_file_has_a_documented_margin_not_a_magic_literal() -> None:
    entry = load_baseline("cache_hit_p99_ms")
    assert entry["baseline_ms"] > 0
    assert entry["margin"] > 0
    assert entry["source"], (
        "the baseline entry must document why the margin is what it is -- "
        "a bare number with no source is the exact magic-literal shape this "
        "ticket removes"
    )


def test_unknown_baseline_name_fails_closed() -> None:
    with pytest.raises(KeyError):
        load_baseline("no_such_metric_ms")


def test_threshold_rejects_a_real_2x_regression() -> None:
    entry = load_baseline("cache_hit_p99_ms")
    threshold = threshold_ms("cache_hit_p99_ms")
    regressed = entry["baseline_ms"] * 2.0
    assert regressed > threshold, (
        f"a genuine 2x regression ({regressed}ms) must exceed the threshold "
        f"({threshold}ms) -- otherwise the gate cannot catch a real regression"
    )


def test_threshold_admits_a_5_percent_wobble() -> None:
    entry = load_baseline("cache_hit_p99_ms")
    threshold = threshold_ms("cache_hit_p99_ms")
    wobble = entry["baseline_ms"] * 1.05
    assert wobble < threshold, (
        f"ordinary 5 percent runner wobble ({wobble}ms) must stay under the "
        f"threshold ({threshold}ms) -- otherwise the gate flakes on noise"
    )


def test_median_of_trials_survives_a_single_noisy_trial_outlier() -> None:
    """Reproduces run 34666515741 and 34782693843's shape: 100 clean samples
    plus one single-sample spike, in exactly one of five trials."""
    clean_trial = [0.02] * 99 + [0.05]
    noisy_trial_like_36ms_run = [0.02] * 99 + [36.0]
    noisy_trial_like_3ms_run = [0.019] * 99 + [3.006]

    for noisy_trial in (noisy_trial_like_36ms_run, noisy_trial_like_3ms_run):
        trials = [clean_trial, clean_trial, clean_trial, clean_trial, noisy_trial]
        result = median_of_trial_percentiles(trials, 99)
        assert result < 1.0, (
            f"the median across 5 trials ({result}ms) must not be dragged up "
            "by one noisy trial's single-sample outlier"
        )


def test_median_of_trials_still_catches_a_regression_present_in_every_trial() -> None:
    """A sustained 2x-baseline slowdown in every trial must not be averaged away."""
    regressed_trial = [0.5] * 100
    trials = [regressed_trial] * 5
    result = median_of_trial_percentiles(trials, 99)
    assert result == pytest.approx(0.5)


def test_median_of_trials_needs_at_least_one_trial() -> None:
    with pytest.raises(ValueError):
        median_of_trial_percentiles([], 99)
