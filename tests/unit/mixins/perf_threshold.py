# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18350: baseline-derived performance thresholds for microbenchmarks.

Replaces the magic-literal `PERF_THRESHOLD_CACHE_HIT_MS * PERF_MULTIPLIER`
pattern in `test_mixin_node_introspection.py` with a threshold computed from a
committed, documented baseline entry in `perf_baselines.json`. The margin is a
property of the baseline entry -- stated and sourced -- not an unexplained
runtime constant.

`median_of_trial_percentiles` is the companion fix for the actual defect that
caused OMN-18350's two nightly failures: a percentile computed over exactly
100 samples degenerates to `max(samples)`, so a single GC-pause or
scheduler-preemption sample on a shared CI runner reads as a regression.
Running several independent trials and taking the median of each trial's
percentile removes a single noisy trial without hiding a regression that is
present in every trial.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

BASELINE_FILE = Path(__file__).parent / "perf_baselines.json"


def load_baseline(name: str) -> dict[str, Any]:
    """Read one named baseline entry from the committed baseline file.

    Raises ``KeyError`` (fail-fast, no silent default) if the name is not
    recorded.
    """
    data: dict[str, Any] = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    if name not in data:
        raise KeyError(
            f"no recorded performance baseline named {name!r} in {BASELINE_FILE}"
        )
    entry: dict[str, Any] = data[name]
    return entry


def threshold_ms(name: str) -> float:
    """The assertion threshold for a named baseline: ``baseline * (1 + margin)``."""
    entry = load_baseline(name)
    baseline_ms: float = entry["baseline_ms"]
    margin: float = entry["margin"]
    return baseline_ms * (1.0 + margin)


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    sorted_values = sorted(values)
    index = min(int(len(sorted_values) * pct / 100), len(sorted_values) - 1)
    return sorted_values[index]


def median_of_trial_percentiles(trials: list[list[float]], percentile: int) -> float:
    """The median, across independent trials, of each trial's own percentile.

    A single trial wrecked by one noisy sample (the OMN-18350 shape: 99 clean
    samples plus one GC-pause spike) contributes one outlying trial-percentile;
    the median across several trials is unmoved by it. A regression present in
    every trial raises every trial's percentile and is still caught.
    """
    if not trials:
        raise ValueError("median_of_trial_percentiles needs at least one trial")
    per_trial = [_percentile(trial, percentile) for trial in trials]
    return median(per_trial)
