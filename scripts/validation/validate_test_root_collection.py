#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15378: fail closed on a `tests/` directory pytest can never collect.

Why this exists
----------------
``scripts/deploy-agent/tests/`` sat completely uncollected for ~5 weeks: no
entry in ``pyproject.toml`` ``testpaths``, no CI job, no governed selector
mapping reached it. A test asserting a literal superseded by OMN-12990 was RED
that entire time with zero signal, found only by hand in an unrelated PR
(OMN-14968 / #2536). This is a false-green class, not a stale-test nit — any
regression covered by an uncollected root is unenforced.

What this checks
-----------------
Every directory containing at least one ``test_*.py`` file must be reachable
by SOME real, wired pytest invocation:

1. Under the root-collected tree (``tests/`` — see ``pyproject.toml``
   ``[tool.pytest.ini_options] testpaths``), collected by the full-suite and
   governed-selector CI jobs in ``.github/workflows/ci.yml``; or
2. A registered :data:`STANDALONE_PROJECT_ROOTS` entry — a directory that owns
   its own ``pyproject.toml`` (a genuinely separate uv (sub-)project, e.g.
   ``scripts/deploy-agent``) AND has a live GitHub Actions workflow that runs
   its tests. All three legs are verified: the ``pyproject.toml`` exists, the
   workflow file exists and references the root, and the workflow is reachable
   on a pull request (its own ``pull_request`` trigger, or a PR-reachable
   workflow calls it via ``uses:``). An allowlist entry with no wiring behind
   it IS the OMN-15378 defect; or
3. Named in :data:`KNOWN_UNCOLLECTED_DEBT` — pre-existing uncollected roots
   present at the time this guard landed, called out explicitly rather than
   silently grandfathered. Do NOT add a new root here to make a violation
   pass — wire its collection (path 1 or 2 above) instead. Every entry here
   is a live defect; see the OMN-15378 PR body for the follow-up-ticket ask.

Deliberately excluded: ``scripts/deploy-agent/tests/`` cannot simply be added
to ``pyproject.toml`` ``testpaths`` or folded into the main ``tests/`` tree —
both trees declare a top-level ``tests`` package (``tests/__init__.py`` vs
``scripts/deploy-agent/tests/__init__.py``), so pytest raises
``ImportPathMismatchError`` when both are collected in one session (verified
locally 2026-07-29). That is why it is a registered standalone project, not a
folded-in directory.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# The canonical, root-collected pytest tree — everything under here is
# reachable via `uv run pytest tests/` (ci.yml's full-suite / smart-selection
# invocations), per pyproject.toml's `testpaths = ["tests"]`.
COLLECTED_ROOT = "tests/"

# Directories that own a private pyproject.toml (a genuine separate uv
# (sub-)project) and are exercised by a DEDICATED CI workflow rather than
# folded into the main `tests/` tree. The pyproject.toml and the workflow file
# must exist on disk, the workflow must REFERENCE the root, and the workflow
# must be reachable on a pull request -- either it triggers on `pull_request`
# itself, or a workflow that does calls it via `uses:` (OMN-15378 AC3: the
# deploy-agent workflow is `workflow_call`-only and is invoked by ci.yml's
# `deploy-agent-tests` job so its result lands under the required "CI Summary"
# context). An unwired allowlist entry -- a workflow file that exists but never
# runs on a PR, or runs but does not touch the root -- is exactly the class of
# defect this guard exists to catch.
STANDALONE_PROJECT_ROOTS: dict[str, str] = {
    "scripts/deploy-agent": ".github/workflows/deploy-agent-tests.yml",
}

# Pre-existing debt at the time OMN-15378 landed this guard (2026-07-29).
# These roots are REAL, currently-uncollected test directories with zero CI
# wiring -- flagged, not silently forgiven. Do not add to this set to make a
# NEW violation pass; that defeats the guard. Fixing these is out of scope for
# OMN-15378 (see PR body RESIDUALS); each needs its own follow-up ticket.
KNOWN_UNCOLLECTED_DEBT: frozenset[str] = frozenset(
    {
        "scripts/ci/tests",
        "scripts/tests",
        "scripts/runtime_build/tests",
        "src/omnibase_infra/services/observability/agent_actions/tests",
    }
)

_IGNORED_DIR_PARTS = (".git", "__pycache__", ".venv", "node_modules")


def find_test_dirs(repo_root: Path) -> list[str]:
    """Return every repo-relative directory literally named ``tests`` (POSIX,
    trailing slash) that directly contains at least one ``test_*.py`` file.

    Scoped to directories named ``tests`` -- not "any file matching
    ``test_*.py`` anywhere" -- because several modules in this repo are named
    ``test_selection_*.py`` as PRODUCT CODE (the change-aware selector,
    ``scripts/ci/test_selection_models.py`` / ``test_selection_loader.py``),
    not pytest tests; a bare ``test_*.py`` glob false-positives on those.
    """
    found: set[str] = set()
    for tests_dir in repo_root.rglob("tests"):
        if not tests_dir.is_dir():
            continue
        rel = tests_dir.relative_to(repo_root)
        if any(part in _IGNORED_DIR_PARTS for part in rel.parts):
            continue
        # Recursive: scripts/deploy-agent/tests/unit/test_*.py must count as
        # content of the scripts/deploy-agent/tests/ root, not be missed
        # because the test files sit one level below the `tests` dir itself.
        if any(tests_dir.rglob("test_*.py")):
            found.add(rel.as_posix() + "/")
    return sorted(found)


def _is_collected(test_dir: str) -> bool:
    return test_dir.startswith(COLLECTED_ROOT)


def _load_workflow(path: Path) -> dict[str, Any]:
    """Parse a workflow file; an unparseable file yields an empty mapping."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _workflow_triggers(workflow: dict[str, Any]) -> set[str]:
    """Return the workflow's trigger names.

    PyYAML resolves the bare ``on:`` key to the boolean ``True`` (YAML 1.1), so
    both spellings are checked. ``on: [push, pull_request]`` (list form) and
    ``on: push`` (scalar form) are normalized alongside the mapping form.
    """
    raw = workflow.get(True, workflow.get("on"))
    if isinstance(raw, dict):
        return {str(key) for key in raw}
    if isinstance(raw, list):
        return {str(item) for item in raw}
    if isinstance(raw, str):
        return {raw}
    return set()


def _workflow_runs_on_pull_request(
    workflow_rel: str, repo_root: Path, _seen: frozenset[str] = frozenset()
) -> bool:
    """True when ``workflow_rel`` actually executes on a pull request.

    Either it declares a ``pull_request`` trigger itself, or it is a reusable
    (``workflow_call``) workflow invoked via ``uses:`` by another workflow that
    is itself PR-reachable. A registered wiring workflow that no event and no
    caller can ever reach runs zero tests -- the OMN-15378 defect wearing a
    workflow file as a disguise.
    """
    if workflow_rel in _seen:  # cycle guard
        return False
    workflow = _load_workflow(repo_root / workflow_rel)
    triggers = _workflow_triggers(workflow)
    if "pull_request" in triggers or "pull_request_target" in triggers:
        return True
    if "workflow_call" not in triggers:
        return False
    reference = f"./{workflow_rel}"
    workflows_dir = repo_root / ".github" / "workflows"
    candidates = sorted(workflows_dir.glob("*.yml")) + sorted(
        workflows_dir.glob("*.yaml")
    )
    for candidate in candidates:
        candidate_rel = candidate.relative_to(repo_root).as_posix()
        if candidate_rel == workflow_rel:
            continue
        jobs = _load_workflow(candidate).get("jobs")
        if not isinstance(jobs, dict):
            continue
        calls_it = any(
            str((body or {}).get("uses") or "") == reference
            for body in jobs.values()
            if isinstance(body, dict)
        )
        if calls_it and _workflow_runs_on_pull_request(
            candidate_rel, repo_root, _seen | {workflow_rel}
        ):
            return True
    return False


def _standalone_problem(test_dir: str, repo_root: Path) -> str | None:
    """Return None if legitimately wired, 'unregistered' if no match, else a
    description of what is wrong with the registered entry."""
    for root, workflow in STANDALONE_PROJECT_ROOTS.items():
        root_prefix = root.rstrip("/") + "/"
        if test_dir == root_prefix or test_dir.startswith(root_prefix):
            pyproject = repo_root / root / "pyproject.toml"
            workflow_path = repo_root / workflow
            if not pyproject.is_file():
                return (
                    f"registered as a STANDALONE_PROJECT_ROOTS entry but "
                    f"{root}/pyproject.toml does not exist"
                )
            if not workflow_path.is_file():
                return (
                    f"registered as a STANDALONE_PROJECT_ROOTS entry but "
                    f"the wiring workflow {workflow} does not exist"
                )
            if root.rstrip("/") not in workflow_path.read_text(encoding="utf-8"):
                return (
                    f"registered as a STANDALONE_PROJECT_ROOTS entry but the "
                    f"wiring workflow {workflow} never references {root} — it "
                    f"cannot be running those tests"
                )
            if not _workflow_runs_on_pull_request(workflow, repo_root):
                return (
                    f"registered as a STANDALONE_PROJECT_ROOTS entry but the "
                    f"wiring workflow {workflow} never runs on a pull request "
                    f"(no pull_request trigger, and no PR-reachable workflow "
                    f"calls it via uses:) — the tests are invisible again"
                )
            return None
    return "unregistered"


def find_violations(repo_root: Path = REPO_ROOT) -> list[str]:
    """Return a human-readable violation string per uncollected test root."""
    violations: list[str] = []
    for test_dir in find_test_dirs(repo_root):
        if _is_collected(test_dir):
            continue
        if test_dir.rstrip("/") in KNOWN_UNCOLLECTED_DEBT:
            continue
        problem = _standalone_problem(test_dir, repo_root)
        if problem is None:
            continue
        if problem == "unregistered":
            violations.append(
                f"{test_dir}: not under '{COLLECTED_ROOT}', not a registered "
                "STANDALONE_PROJECT_ROOTS entry, and not in "
                "KNOWN_UNCOLLECTED_DEBT — no pytest invocation in CI can ever "
                "run these tests (OMN-15378 class). Wire it into tests/, "
                "register it in STANDALONE_PROJECT_ROOTS with real CI wiring, "
                "or (last resort, requires a follow-up ticket) add it to "
                "KNOWN_UNCOLLECTED_DEBT with a comment explaining why."
            )
        else:
            violations.append(f"{test_dir}: {problem}")
    return violations


def main() -> int:
    violations = find_violations()
    if violations:
        print("FAIL: uncollected pytest root(s) detected (OMN-15378 class):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("OK: every tests/ directory is reachable by a wired pytest invocation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
