# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the OMN-15378/OMN-15410 uncollected-pytest-root guard.

``scripts/deploy-agent/tests/`` sat uncollected by any CI job for ~5 weeks; a
RED test inside it (superseded OMN-12988 literal) went unnoticed the entire
time. OMN-15410 then collected the four remaining roots of the same class and
closed the two seams that made the class possible. This module proves all of
it:

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
  5. OMN-15410: collected roots come from ``pyproject.toml`` ``testpaths``,
     ci.yml's full suite passes no positional path that would override them,
     and every collocated root is selectable by the change-aware selector.
     Each has a synthetic RED-proof alongside the live-repo green assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.ci.detect_test_paths import COLLOCATED_TEST_ROOTS
from scripts.validation.validate_test_root_collection import (
    FULL_SUITE_STEP_NAME,
    KNOWN_UNCOLLECTED_DEBT,
    REPO_ROOT,
    STANDALONE_PROJECT_ROOTS,
    check_collocated_selector_coverage,
    check_full_suite_invocation,
    collected_roots,
    find_test_dirs,
    find_violations,
    positional_pytest_args,
)

pytestmark = pytest.mark.unit

# The four roots OMN-15410 moved out of KNOWN_UNCOLLECTED_DEBT and into
# collection. Named literally so a silent removal from testpaths reddens here
# rather than quietly dropping 366 tests again.
OMN_15410_COLLECTED_ROOTS = (
    "scripts/ci/tests/",
    "scripts/tests/",
    "scripts/runtime_build/tests/",
    "src/omnibase_infra/services/observability/agent_actions/tests/",
)


def _synthetic_repo(tmp_path: Path, testpaths: str = '["tests"]') -> Path:
    """A tmp_path repo root with just enough config for the guard to run.

    The guard reads ``testpaths`` from pyproject.toml rather than assuming
    ``tests/`` (OMN-15410), so a synthetic fixture must declare its own —
    fail-closed by design: no pyproject means no answer, not a guessed default.
    """
    (tmp_path / "pyproject.toml").write_text(
        f"[tool.pytest.ini_options]\ntestpaths = {testpaths}\n"
    )
    return tmp_path


def test_live_repo_has_no_uncollected_test_roots() -> None:
    """The actual CI gate: every tests/ dir in THIS repo is wired somewhere."""
    violations = find_violations(REPO_ROOT)
    assert violations == [], (
        "Uncollected pytest root(s) detected (OMN-15378 class):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_synthetic_stray_tests_dir_is_a_violation(tmp_path: Path) -> None:
    """RED-proof: a stray tests/ dir with no wiring anywhere must fail."""
    repo = _synthetic_repo(tmp_path)
    (repo / "tests").mkdir()
    stray = repo / "scripts" / "widget" / "tests"
    stray.mkdir(parents=True)
    (stray / "test_widget.py").write_text("def test_ok() -> None:\n    pass\n")

    violations = find_violations(repo)

    assert len(violations) == 1
    assert violations[0].startswith("scripts/widget/tests/:")
    assert "unregistered" not in violations[0]  # human message, not the raw sentinel
    assert "no pytest invocation in CI can ever run these tests" in violations[0]


def test_synthetic_collected_root_tests_dir_is_not_a_violation(
    tmp_path: Path,
) -> None:
    """A tests/ dir under the root-collected tree is never flagged."""
    repo = _synthetic_repo(tmp_path)
    collected = repo / "tests" / "unit" / "widget"
    collected.mkdir(parents=True)
    (collected / "test_widget.py").write_text("def test_ok() -> None:\n    pass\n")

    assert find_violations(repo) == []


def test_synthetic_root_named_in_testpaths_is_collected(tmp_path: Path) -> None:
    """OMN-15410: a collocated root becomes collected by declaring it in
    testpaths — the mechanism the four real roots now use."""
    repo = _synthetic_repo(tmp_path, testpaths='["tests", "scripts/widget/tests"]')
    (repo / "tests").mkdir()
    widget = repo / "scripts" / "widget" / "tests"
    widget.mkdir(parents=True)
    (widget / "test_widget.py").write_text("def test_ok() -> None:\n    pass\n")

    assert find_violations(repo) == []


def test_synthetic_missing_testpaths_entry_is_a_violation(tmp_path: Path) -> None:
    """RED-proof: a testpaths entry with no directory behind it would abort
    pytest collection with exit 5, so the guard rejects it."""
    repo = _synthetic_repo(tmp_path, testpaths='["tests", "scripts/gone/tests"]')
    (repo / "tests").mkdir()

    violations = find_violations(repo)

    assert len(violations) == 1
    assert violations[0].startswith("scripts/gone/tests/:")
    assert "exit 5" in violations[0]


def test_collected_roots_fails_closed_on_empty_testpaths(tmp_path: Path) -> None:
    """RED-proof: an empty testpaths would make bare `pytest` collect the whole
    repository (including .venv), so reading it must raise, not return ()."""
    repo = _synthetic_repo(tmp_path, testpaths="[]")

    with pytest.raises(ValueError, match="declares no"):
        collected_roots(repo)


def test_collected_roots_matches_live_testpaths() -> None:
    """The live repo collects tests/ plus the four OMN-15410 roots."""
    roots = collected_roots(REPO_ROOT)

    assert "tests/" in roots
    for root in OMN_15410_COLLECTED_ROOTS:
        assert root in roots, (
            f"{root} dropped out of pyproject.toml testpaths — the OMN-15410 "
            "roots would silently stop being collected again."
        )


def test_synthetic_unregistered_standalone_project_fails_closed(
    tmp_path: Path,
) -> None:
    """A registered STANDALONE_PROJECT_ROOTS entry with no real workflow file
    behind it must still fail — the allowlist cannot be satisfied by adding a
    dict entry alone."""
    from scripts.validation import validate_test_root_collection as module

    repo = _synthetic_repo(tmp_path)
    (repo / "tests").mkdir()
    root = repo / "scripts" / "unwired-agent"
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
        violations = module.find_violations(repo)
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
    collected = collected_roots(REPO_ROOT)
    for debt_entry in KNOWN_UNCOLLECTED_DEBT:
        assert debt_entry in live_test_dirs, (
            f"KNOWN_UNCOLLECTED_DEBT entry {debt_entry!r} no longer exists as "
            "a tests/ directory with test_*.py files — remove it from the "
            "allowlist (OMN-15378 guard)."
        )
        assert not any(f"{debt_entry}/".startswith(root) for root in collected), (
            f"KNOWN_UNCOLLECTED_DEBT entry {debt_entry!r} IS collected now — "
            "remove it from the allowlist rather than leaving dead cover "
            "(OMN-15410)."
        )


def test_omn15410_roots_are_no_longer_uncollected_debt() -> None:
    """The OMN-15410 deliverable: the baseline shrank by exactly these four."""
    for root in OMN_15410_COLLECTED_ROOTS:
        assert root.rstrip("/") not in KNOWN_UNCOLLECTED_DEBT, (
            f"{root} is collected via testpaths but still listed as "
            "KNOWN_UNCOLLECTED_DEBT — the two cannot both be true."
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


# =============================================================================
# OMN-15410 seam 2: ci.yml must not override testpaths with a positional path
# =============================================================================


def test_live_full_suite_step_passes_no_positional_path() -> None:
    """The live gate: ci.yml's full suite inherits testpaths verbatim."""
    violations = check_full_suite_invocation(REPO_ROOT)
    assert violations == [], "\n".join(violations)


def test_positional_pytest_args_ignores_options_and_gh_expressions() -> None:
    """The real full-suite command shape yields zero positional paths.

    Guards the parser itself: ``--splits ${{ ... }}``, a quoted ``-m`` marker
    expression, ``-n 2 --dist loadgroup`` and
    ``--junitxml=junit-${{ matrix.split }}.xml`` must not be read as paths.
    """
    run_block = (
        "uv run pytest \\\n"
        "  --ignore=tests/integration/docker \\\n"
        '  -m "not slow and not chaos and not kafka and not performance" \\\n'
        "  --splits ${{ needs.detect-changes.outputs.split_count }} \\\n"
        "  --group ${{ matrix.split }} \\\n"
        "  -n 2 --dist loadgroup \\\n"
        "  --timeout=60 \\\n"
        "  --timeout-method=thread \\\n"
        "  --tb=short \\\n"
        "  --store-durations \\\n"
        "  --junitxml=junit-${{ matrix.split }}.xml\n"
    )

    assert positional_pytest_args(run_block) == []


def test_positional_pytest_args_detects_reintroduced_path() -> None:
    """RED-proof: re-adding `tests/` is exactly the OMN-15410 regression."""
    run_block = "uv run pytest tests/ --ignore=tests/integration/docker -n 2\n"

    assert positional_pytest_args(run_block) == ["tests/"]


def test_full_suite_check_reports_reintroduced_positional_path(
    tmp_path: Path,
) -> None:
    """RED-proof end to end: a synthetic ci.yml with `pytest tests/` fails."""
    repo = _synthetic_repo(tmp_path)
    (repo / "tests").mkdir()
    workflow_dir = repo / ".github" / "workflows"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "ci.yml").write_text(
        "jobs:\n"
        "  test:\n"
        "    steps:\n"
        f"      - name: {FULL_SUITE_STEP_NAME}\n"
        "        run: |\n"
        "          uv run pytest tests/ --tb=short\n"
    )

    violations = check_full_suite_invocation(repo)

    assert len(violations) == 1
    assert "overriding pyproject.toml testpaths" in violations[0]
    assert "tests/" in violations[0]


# =============================================================================
# OMN-15410 seam 3: collocated roots must be selectable by the selector
# =============================================================================


def test_live_collocated_roots_are_selector_reachable() -> None:
    """The live gate: testpaths <-> COLLOCATED_TEST_ROOTS parity holds."""
    violations = check_collocated_selector_coverage(REPO_ROOT)
    assert violations == [], "\n".join(violations)


def test_every_omn15410_root_is_mapped_from_its_own_source_prefix() -> None:
    """Each collocated root is reachable from a diff touching its own code, not
    only from an unrelated full-suite escalation."""
    mapped = set(COLLOCATED_TEST_ROOTS.values())
    for root in OMN_15410_COLLECTED_ROOTS:
        assert root in mapped, (
            f"{root} has no COLLOCATED_TEST_ROOTS mapping — a narrowed "
            "smart-selection run could never select it (OMN-15410)."
        )
    for source_prefix, root in COLLOCATED_TEST_ROOTS.items():
        assert root.startswith(source_prefix), (
            f"COLLOCATED_TEST_ROOTS maps {source_prefix!r} -> {root!r}, but the "
            "root does not live under that prefix; the mapping would not fire "
            "for a change to the code it covers."
        )
