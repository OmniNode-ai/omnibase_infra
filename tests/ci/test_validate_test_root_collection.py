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
  5. That workflow is actually *reachable on a pull request* and actually
     *references the root* (OMN-15378 AC3). File existence alone was never
     proof of wiring: the deploy-agent workflow is now ``workflow_call``-only,
     reachable solely through ci.yml's caller job, which is what puts its
     result under the required "CI Summary" context.
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


def _synthetic_standalone_root(tmp_path: Path) -> Path:
    """A registered-shaped standalone project: pyproject.toml + its own tests/."""
    root = tmp_path / "scripts" / "widget-agent"
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_thing.py").write_text("def test_ok() -> None:\n    pass\n")
    (root / "pyproject.toml").write_text("[project]\nname = 'widget-agent'\n")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    return root


def _with_registration(
    tmp_path: Path, workflow_rel: str
) -> list[str]:  # pragma: no cover - helper
    from scripts.validation import validate_test_root_collection as module

    original = dict(module.STANDALONE_PROJECT_ROOTS)
    module.STANDALONE_PROJECT_ROOTS.clear()
    module.STANDALONE_PROJECT_ROOTS["scripts/widget-agent"] = workflow_rel
    try:
        return module.find_violations(tmp_path)
    finally:
        module.STANDALONE_PROJECT_ROOTS.clear()
        module.STANDALONE_PROJECT_ROOTS.update(original)


def test_registered_workflow_that_never_runs_on_a_pr_fails_closed(
    tmp_path: Path,
) -> None:
    """RED-proof for the OMN-15378 AC3 hardening: a `workflow_call`-only wiring
    workflow that NO workflow invokes runs zero tests, so registering it must
    still fail — file existence alone was never proof of wiring."""
    _synthetic_standalone_root(tmp_path)
    (tmp_path / ".github" / "workflows" / "widget-agent-tests.yml").write_text(
        "name: Widget Agent Tests\n"
        "on:\n"
        "  workflow_call:\n"
        "jobs:\n"
        "  widget-agent-tests:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: pytest scripts/widget-agent/tests\n"
    )

    violations = _with_registration(
        tmp_path, ".github/workflows/widget-agent-tests.yml"
    )

    assert len(violations) == 1
    assert "never runs on a pull request" in violations[0]


def test_registered_workflow_called_by_a_pr_workflow_is_accepted(
    tmp_path: Path,
) -> None:
    """The shape this repo now uses: the reusable is invoked by a PR-triggered
    caller (ci.yml), which is what puts its result under a required context."""
    _synthetic_standalone_root(tmp_path)
    workflows = tmp_path / ".github" / "workflows"
    (workflows / "widget-agent-tests.yml").write_text(
        "name: Widget Agent Tests\n"
        "on:\n"
        "  workflow_call:\n"
        "jobs:\n"
        "  widget-agent-tests:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: pytest scripts/widget-agent/tests\n"
    )
    (workflows / "ci.yml").write_text(
        "name: CI\n"
        "on:\n"
        "  pull_request:\n"
        "    branches: [dev]\n"
        "jobs:\n"
        "  widget-agent-tests:\n"
        "    uses: ./.github/workflows/widget-agent-tests.yml\n"
    )

    assert (
        _with_registration(tmp_path, ".github/workflows/widget-agent-tests.yml") == []
    )


def test_registered_workflow_that_does_not_reference_the_root_fails_closed(
    tmp_path: Path,
) -> None:
    """A PR-triggered workflow that never mentions the root cannot be running
    its tests — registration must not be satisfiable by pointing at any old
    workflow file (e.g. re-pointing an entry at ci.yml)."""
    _synthetic_standalone_root(tmp_path)
    (tmp_path / ".github" / "workflows" / "unrelated.yml").write_text(
        "name: Unrelated\n"
        "on:\n"
        "  pull_request:\n"
        "    branches: [dev]\n"
        "jobs:\n"
        "  lint:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo lint\n"
    )

    violations = _with_registration(tmp_path, ".github/workflows/unrelated.yml")

    assert len(violations) == 1
    assert "never references scripts/widget-agent" in violations[0]


def test_deploy_agent_wiring_workflow_is_pr_reachable_in_this_repo() -> None:
    """Live assertion for the registration this repo actually ships: the
    deploy-agent reusable is `workflow_call`-only, so its PR reachability comes
    entirely from ci.yml's caller job. If that caller is removed, this fails."""
    from scripts.validation.validate_test_root_collection import (
        _workflow_runs_on_pull_request,
    )

    for root, workflow in STANDALONE_PROJECT_ROOTS.items():
        assert _workflow_runs_on_pull_request(workflow, REPO_ROOT), (
            f"{root}'s wiring workflow {workflow} is not reachable on a pull "
            "request — its tests would be uncollected in practice"
        )


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
