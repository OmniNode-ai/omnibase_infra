# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18542: shard count is sized from the test population, not the path count.

Why this module exists, stated so a later reader does not have to reconstruct it
from a diff.

`_split_count_for()` used to map the NUMBER OF SELECTED PATH STRINGS to a shard
count (``n <= 2 -> 1``, ``n <= 5 -> 2``, ``n <= 10 -> 3`` ...). Nothing in that
ladder related a path to the amount of work behind it, and one string can be the
entire unit tree. Two live populations on 2026-09-16, both cancelled within one
second of ``timeout-minutes: 15`` and therefore neither a hang:

* ``omnibase_infra#3652`` ``Tests (Split 1/3)``, cancelled twice at 15m14s and
  15m13s. The selection was the ten paths in TEN_PATH_WHOLE_SUITE_SELECTION
  below -- effectively the whole tree -- and the ladder gave it three shards.
  The job log recorded ``2 workers [12264 items]`` and steady progress to 59% at
  the wall, so the shard needed about 25 minutes.
* The conservative fallback ``["tests/unit/"]``, one shard for 28,455 cases,
  measured at 11.9, 14.5, 14.7 and 15.0 minutes on four runs the same day. The
  15.0 passed by seconds.

The same tests on the full-suite path are given 15 shards, so the two branches of
one module disagreed with each other by up to fifteen times on identical input.

The rule these cases pin is therefore a PARITY rule rather than a magic number:
no narrowed selection may be denser per shard than the full-suite run of the
whole tree, give or take the one shard reserved so a narrowed run can never mint
the full-suite denominator (see `_MAX_NARROWED_SPLIT_COUNT`). Both sides of that
comparison are counted from the working tree at selection time, so the rule
cannot go stale as the suite grows.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts.ci.detect_test_paths import (
    _FULL_SUITE_SPLIT_COUNT,
    _MAX_NARROWED_SPLIT_COUNT,
    collectable_test_file_count,
    compute_selection,
    full_suite_test_file_count,
    split_count_for_selection,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
ADJ = REPO_ROOT / "scripts/ci/test_selection_adjacency.yaml"

# The exact selection `detect-changes` emitted for omnibase_infra#3652, read
# from run 35155416362's Detect Changes job log. Reproduced verbatim: it is the
# live fixture, not an illustrative one.
TEN_PATH_WHOLE_SUITE_SELECTION = [
    "scripts/ci/tests/",
    "scripts/tests/",
    "tests/ci/",
    "tests/integration/",
    "tests/scripts/",
    "tests/test_no_redaction_placeholder_in_docker.py",
    "tests/unit/",
    "tests/unit/docker/",
    "tests/unit/infra/",
    "tests/unit/scripts/",
]


def _target_files_per_split() -> int:
    """The per-shard file population the full-suite path itself accepts."""
    return math.ceil(full_suite_test_file_count(REPO_ROOT) / _FULL_SUITE_SPLIT_COUNT)


# ---------------------------------------------------------------------------
# Counting the population
# ---------------------------------------------------------------------------


def test_full_suite_population_is_counted_from_the_working_tree() -> None:
    """The denominator is measured, not a constant that ages."""
    total = full_suite_test_file_count(REPO_ROOT)
    # Measured 2026-09-16 at 2,231 collectable modules across the five
    # pyproject `testpaths` roots. The bound is deliberately wide: this asserts
    # the count is real, and a narrow bound would turn ordinary suite growth
    # into a red test, which is how a control gets suppressed.
    assert total > 1000, f"full-suite population implausibly small: {total}"


def test_overlapping_selected_paths_are_counted_once() -> None:
    """`tests/unit/docker/` inside `tests/unit/` must not be double counted."""
    broad = collectable_test_file_count(["tests/unit/"], REPO_ROOT)
    with_redundant_children = collectable_test_file_count(
        ["tests/unit/", "tests/unit/docker/", "tests/unit/scripts/"], REPO_ROOT
    )
    assert broad == with_redundant_children
    # Positive control on that equality: the child alone is a strict, non-zero
    # subset, so the equality above is not two zeros agreeing.
    child = collectable_test_file_count(["tests/unit/docker/"], REPO_ROOT)
    assert 0 < child < broad


def test_a_path_that_is_not_on_disk_contributes_nothing() -> None:
    """Fail-closed on the denominator, not on the numerator."""
    assert collectable_test_file_count(["tests/does_not_exist/"], REPO_ROOT) == 0


# ---------------------------------------------------------------------------
# AC1 / AC2 -- the two live failure populations
# ---------------------------------------------------------------------------


def test_whole_suite_selection_gets_full_suite_shard_parity() -> None:
    """AC1: omnibase_infra#3652's ten-path selection is not given three shards.

    Red before the fix: the path-count ladder returned 3 for ten paths, and the
    shard needed ~25 minutes against a 15-minute ceiling.
    """
    count = split_count_for_selection(TEN_PATH_WHOLE_SUITE_SELECTION, REPO_ROOT)
    assert count == _MAX_NARROWED_SPLIT_COUNT, (
        f"selection {TEN_PATH_WHOLE_SUITE_SELECTION} covers "
        f"{collectable_test_file_count(TEN_PATH_WHOLE_SUITE_SELECTION, REPO_ROOT)} "
        f"of {full_suite_test_file_count(REPO_ROOT)} collectable modules but was "
        f"given {count} shards; the full-suite path gives the same work "
        f"{_FULL_SUITE_SPLIT_COUNT}, so a narrowed selection this wide belongs "
        f"at the narrowed ceiling {_MAX_NARROWED_SPLIT_COUNT}"
    )


def test_unit_root_fallback_is_sized_to_its_population() -> None:
    """AC2: the conservative `["tests/unit/"]` fallback is not one shard."""
    selection = compute_selection(
        changed_files=["docker/catalog/services/some_new_service.yaml"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    # The fallback itself is unchanged -- this module does not touch _resolve().
    assert selection.selected_paths == ["tests/unit/"]

    files = collectable_test_file_count(["tests/unit/"], REPO_ROOT)
    target = _target_files_per_split()
    assert selection.split_count == math.ceil(files / target), (
        f"tests/unit/ carries {files} collectable modules against a per-shard "
        f"target of {target}, so it needs {math.ceil(files / target)} shards; "
        f"the selector returned {selection.split_count}"
    )
    assert selection.split_count > 1
    assert selection.matrix == list(range(1, selection.split_count + 1))


def test_no_selection_is_denser_per_shard_than_the_full_suite_run() -> None:
    """The parity invariant itself, over every root a selection can name.

    The tolerance is the one the reserved denominator costs and nothing more:
    a narrowed selection is capped one shard below the full-suite count, so the
    widest one is at most `15/14` as dense. A selection that exceeds that is
    over-packed for a reason other than the reservation.
    """
    target = (
        _target_files_per_split() * _FULL_SUITE_SPLIT_COUNT / _MAX_NARROWED_SPLIT_COUNT
    )
    for paths in (
        ["tests/unit/"],
        ["tests/integration/"],
        ["tests/ci/"],
        ["tests/scripts/"],
        TEN_PATH_WHOLE_SUITE_SELECTION,
    ):
        files = collectable_test_file_count(paths, REPO_ROOT)
        shards = split_count_for_selection(paths, REPO_ROOT)
        per_shard = math.ceil(files / shards)
        assert per_shard <= target, (
            f"{paths} puts {per_shard} modules on a shard, denser than the "
            f"{target:.0f} the reserved-denominator cap allows"
        )


# ---------------------------------------------------------------------------
# AC3 -- the positive control
# ---------------------------------------------------------------------------


def test_a_small_selection_still_runs_on_one_shard() -> None:
    """AC3: a fix that mints fifteen shards for every change is not a fix."""
    selection = compute_selection(
        changed_files=["src/omnibase_infra/cli/foo.py"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert "tests/unit/cli/" in selection.selected_paths
    assert selection.split_count == 1, (
        f"a single leaf module selection carries "
        f"{collectable_test_file_count(selection.selected_paths, REPO_ROOT)} "
        f"collectable modules and must stay on one shard; got "
        f"{selection.split_count}"
    )


def test_a_docs_only_diff_selects_nothing_and_stays_on_one_shard() -> None:
    """An empty selection is a legitimate zero, not a sizing input."""
    selection = compute_selection(
        changed_files=["docs/runbooks/some-new-runbook.md"],
        adjacency_path=ADJ,
        ref_name="pr-branch",
    )
    assert selection.selected_paths == []
    assert selection.split_count == 1


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_a_narrowed_selection_can_never_mint_the_full_suite_denominator() -> None:
    """The structural gap `prepush_remote_verify.py` binding 3 rests on.

    That binding reads the shard denominator out of the CI job names to decide
    whether a green run was the full suite. It is sound only while no narrowed
    selection can reach `_FULL_SUITE_SPLIT_COUNT`. Probed here with REAL paths:
    the same assertion over paths that are not on disk would count zero modules,
    return 1, and pass while measuring nothing.
    """
    widest = max(
        split_count_for_selection(paths, REPO_ROOT)
        for paths in (
            ["tests/"],
            ["tests/unit/"],
            TEN_PATH_WHOLE_SUITE_SELECTION,
            ["tests/unit/", "tests/integration/", "tests/ci/", "tests/scripts/"],
        )
    )
    # Positive control on the bound: the probe reaches the ceiling, so the
    # inequality below is measuring the cap rather than a trivially small number.
    assert widest == _MAX_NARROWED_SPLIT_COUNT
    assert widest < _FULL_SUITE_SPLIT_COUNT


def test_an_unmeasurable_repo_root_falls_closed_to_the_full_suite_count(
    tmp_path: Path,
) -> None:
    """A tree with no collectable modules cannot be sized, so it is not narrowed.

    Sizing divides by a population read off disk. If that read comes back empty
    -- a wrong root, a checkout that has not materialised -- returning 1 would
    put an unknown amount of work on one shard, which is the exact failure this
    module exists to remove. It fails closed to the widest count a narrowed run
    may take instead.
    """
    assert (
        split_count_for_selection(["tests/unit/"], tmp_path)
        == _MAX_NARROWED_SPLIT_COUNT
    )
