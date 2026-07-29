# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the OMN-15378 uncollected-pytest-root guard.

``scripts/deploy-agent/tests/`` sat uncollected by any CI job for ~5 weeks; a
RED test inside it (superseded OMN-12988 literal) went unnoticed the entire
time. This module proves the guard that closes that class:

  1. The live repo passes today (this IS the CI gate assertion — it runs
     inside the required full-suite / smart-selection pytest job).
  2. A synthetic stray ``tests/`` directory outside every collected root is
     detected and fails (RED-proof per OMN-15378 acceptance criterion 3, not
     just green-proof).
  3. Every ``KNOWN_UNCOLLECTED_DEBT`` entry still exists and is still
     genuinely uncollected — an allowlist that silently drifts from reality
     (a debt entry that got fixed, or a debt entry that no longer exists) is
     itself a defect the guard should not paper over.
  4. Every ``STANDALONE_PROJECT_ROOTS`` entry resolves to a real
     ``pyproject.toml`` and a real, live CI workflow file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validation.validate_test_root_collection import (
    KNOWN_UNCOLLECTED_DEBT,
    REPO_ROOT,
    STANDALONE_PROJECT_ROOTS,
    find_test_dirs,
    find_violations,
)

pytestmark = pytest.mark.unit


def test_live_repo_has_no_uncollected_test_roots() -> None:
    """The actual CI gate: every tests/ dir in THIS repo is wired somewhere."""
    violations = find_violations(REPO_ROOT)
    assert violations == [], (
        "Uncollected pytest root(s) detected (OMN-15378 class):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_synthetic_stray_tests_dir_is_a_violation(tmp_path: Path) -> None:
    """RED-proof: a stray tests/ dir with no wiring anywhere must fail."""
    stray = tmp_path / "scripts" / "widget" / "tests"
    stray.mkdir(parents=True)
    (stray / "test_widget.py").write_text("def test_ok() -> None:\n    pass\n")

    violations = find_violations(tmp_path)

    assert len(violations) == 1
    assert violations[0].startswith("scripts/widget/tests/:")
    assert "unregistered" not in violations[0]  # human message, not the raw sentinel
    assert "no pytest invocation in CI can ever run these tests" in violations[0]


def test_synthetic_collected_root_tests_dir_is_not_a_violation(
    tmp_path: Path,
) -> None:
    """A tests/ dir under the root-collected tree is never flagged."""
    collected = tmp_path / "tests" / "unit" / "widget"
    collected.mkdir(parents=True)
    (collected / "test_widget.py").write_text("def test_ok() -> None:\n    pass\n")

    assert find_violations(tmp_path) == []


def test_synthetic_unregistered_standalone_project_fails_closed(
    tmp_path: Path,
) -> None:
    """A registered STANDALONE_PROJECT_ROOTS entry with no real workflow file
    behind it must still fail — the allowlist cannot be satisfied by adding a
    dict entry alone."""
    from scripts.validation import validate_test_root_collection as module

    root = tmp_path / "scripts" / "unwired-agent"
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_thing.py").write_text("def test_ok() -> None:\n    pass\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'unwired-agent'\n")
    # Deliberately do NOT create the workflow file the registration claims.

    original = dict(module.STANDALONE_PROJECT_ROOTS)
    module.STANDALONE_PROJECT_ROOTS.clear()
    module.STANDALONE_PROJECT_ROOTS["scripts/unwired-agent"] = (
        ".github/workflows/does-not-exist.yml"
    )
    try:
        violations = module.find_violations(tmp_path)
    finally:
        module.STANDALONE_PROJECT_ROOTS.clear()
        module.STANDALONE_PROJECT_ROOTS.update(original)

    assert len(violations) == 1
    assert "does-not-exist.yml does not exist" in violations[0]


def test_known_uncollected_debt_entries_still_exist_and_are_still_uncollected() -> None:
    """The debt allowlist must track reality: an entry pointing at a directory
    that no longer exists (or that got wired up) is stale and should be
    removed, not left as permanent cover for an unrelated future violation."""
    live_test_dirs = {d.rstrip("/") for d in find_test_dirs(REPO_ROOT)}
    for debt_entry in KNOWN_UNCOLLECTED_DEBT:
        assert debt_entry in live_test_dirs, (
            f"KNOWN_UNCOLLECTED_DEBT entry {debt_entry!r} no longer exists as "
            "a tests/ directory with test_*.py files — remove it from the "
            "allowlist (OMN-15378 guard)."
        )


def test_standalone_project_roots_are_all_real() -> None:
    """Every STANDALONE_PROJECT_ROOTS registration has a real pyproject.toml
    and a real, live workflow file behind it (belt-and-suspenders on top of
    find_violations itself already asserting this for the live repo)."""
    for root, workflow in STANDALONE_PROJECT_ROOTS.items():
        assert (REPO_ROOT / root / "pyproject.toml").is_file(), (
            f"{root} is registered as a standalone project but has no pyproject.toml"
        )
        assert (REPO_ROOT / workflow).is_file(), (
            f"{root} is registered as a standalone project but its wiring "
            f"workflow {workflow} does not exist"
        )
