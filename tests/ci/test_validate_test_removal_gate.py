# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for OMN-8732 test removal gate counting logic.

Covers the two CR bypasses flagged on PR #1336:
  1. Modified integration tests must NOT count as replacements for deletions.
  2. Renames OUT of tests/integration/ must count as deletions; renames INTO
     must count as additions.
"""

from __future__ import annotations

import io
import sys

import pytest

from scripts.validation.validate_test_removal_gate import (
    count_deletions_and_additions,
    main,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# count_deletions_and_additions
# ---------------------------------------------------------------------------


def test_pure_deletion_counts_as_deletion() -> None:
    diff = "D\ttests/integration/test_foo.py\n"
    assert count_deletions_and_additions(diff) == (1, 0)


def test_pure_addition_counts_as_addition() -> None:
    diff = "A\ttests/integration/test_bar.py\n"
    assert count_deletions_and_additions(diff) == (0, 1)


def test_modified_does_not_count_as_addition() -> None:
    """Regression: the previous counter treated M as a replacement, so
    modifying an unrelated integration test could cancel a deletion."""
    diff = "D\ttests/integration/test_old.py\nM\ttests/integration/test_unrelated.py\n"
    # Deletion is unmatched — modification MUST NOT offset it.
    assert count_deletions_and_additions(diff) == (1, 0)


def test_modification_only_counts_nothing() -> None:
    diff = "M\ttests/integration/test_foo.py\n"
    assert count_deletions_and_additions(diff) == (0, 0)


def test_rename_out_of_integration_counts_as_deletion() -> None:
    """Regression: `git mv tests/integration/test_foo.py tests/unit/test_foo.py`
    previously produced an R* record the old awk parser ignored, letting the
    file escape the integration suite without triggering the gate."""
    diff = "R100\ttests/integration/test_foo.py\ttests/unit/test_foo.py\n"
    assert count_deletions_and_additions(diff) == (1, 0)


def test_rename_into_integration_counts_as_addition() -> None:
    diff = "R100\ttests/unit/test_foo.py\ttests/integration/test_foo.py\n"
    assert count_deletions_and_additions(diff) == (0, 1)


def test_rename_within_integration_counts_as_neither() -> None:
    diff = "R090\ttests/integration/test_old.py\ttests/integration/test_new.py\n"
    # File count inside tests/integration/ is unchanged, so the gate should
    # neither fire nor consume a replacement slot.
    assert count_deletions_and_additions(diff) == (0, 0)


def test_rename_out_plus_unrelated_modification_still_blocks() -> None:
    """End-to-end bypass attempt: rename a test out of integration/ and also
    modify a different integration test to try to offset the deletion."""
    diff = (
        "R100\ttests/integration/test_old.py\ttests/unit/test_old.py\n"
        "M\ttests/integration/test_unrelated.py\n"
    )
    assert count_deletions_and_additions(diff) == (1, 0)


def test_copy_into_integration_counts_as_addition() -> None:
    diff = "C075\ttests/unit/test_foo.py\ttests/integration/test_foo.py\n"
    assert count_deletions_and_additions(diff) == (0, 1)


def test_copy_out_of_integration_counts_as_nothing() -> None:
    # Copy preserves the source, so it is not a deletion.
    diff = "C075\ttests/integration/test_foo.py\ttests/unit/test_foo.py\n"
    assert count_deletions_and_additions(diff) == (0, 0)


def test_non_integration_paths_ignored() -> None:
    diff = (
        "D\ttests/unit/test_foo.py\nA\tsrc/module/new.py\nR100\tdocs/a.md\tdocs/b.md\n"
    )
    assert count_deletions_and_additions(diff) == (0, 0)


def test_nested_integration_subdir_ignored() -> None:
    # Regex anchors to a single-level file directly under tests/integration/
    diff = "D\ttests/integration/subdir/test_foo.py\n"
    assert count_deletions_and_additions(diff) == (0, 0)


def test_non_python_file_ignored() -> None:
    diff = "D\ttests/integration/conftest.yaml\n"
    assert count_deletions_and_additions(diff) == (0, 0)


def test_multiple_mixed_records() -> None:
    diff = (
        "D\ttests/integration/test_alpha.py\n"
        "D\ttests/integration/test_beta.py\n"
        "A\ttests/integration/test_gamma.py\n"
        "M\ttests/integration/test_delta.py\n"
        "R100\ttests/integration/test_epsilon.py\ttests/unit/test_epsilon.py\n"
        "R100\ttests/unit/test_zeta.py\ttests/integration/test_zeta.py\n"
    )
    # Deletions: alpha, beta, epsilon (renamed out) = 3
    # Additions: gamma, zeta (renamed in) = 2
    # Modifications ignored.
    assert count_deletions_and_additions(diff) == (3, 2)


def test_blank_and_crlf_lines_tolerated() -> None:
    diff = "D\ttests/integration/test_foo.py\r\n\nA\ttests/integration/test_bar.py\n"
    assert count_deletions_and_additions(diff) == (1, 1)


# ---------------------------------------------------------------------------
# main() exit codes (via --stdin)
# ---------------------------------------------------------------------------


def _run_main_with_stdin(diff: str, monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setattr(sys, "stdin", io.StringIO(diff))
    return main(["--stdin"])


def test_main_passes_when_deletions_fully_matched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diff = "D\ttests/integration/test_old.py\nA\ttests/integration/test_new.py\n"
    assert _run_main_with_stdin(diff, monkeypatch) == 0


def test_main_fails_when_modification_was_the_only_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key regression: the pre-fix gate would have passed this PR."""
    diff = "D\ttests/integration/test_old.py\nM\ttests/integration/test_other.py\n"
    assert _run_main_with_stdin(diff, monkeypatch) == 1


def test_main_fails_when_rename_out_is_unmatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diff = "R100\ttests/integration/test_old.py\ttests/unit/test_old.py\n"
    assert _run_main_with_stdin(diff, monkeypatch) == 1


def test_main_passes_on_empty_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_main_with_stdin("", monkeypatch) == 0


def test_main_passes_on_rename_within_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diff = "R090\ttests/integration/test_old.py\ttests/integration/test_new.py\n"
    assert _run_main_with_stdin(diff, monkeypatch) == 0
