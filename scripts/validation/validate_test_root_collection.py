#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-15378 / OMN-15410: fail closed on a `tests/` directory CI can never run.

Why this exists
----------------
``scripts/deploy-agent/tests/`` sat completely uncollected for ~5 weeks: no
entry in ``pyproject.toml`` ``testpaths``, no CI job, no governed selector
mapping reached it. A test asserting a literal superseded by OMN-12990 was RED
that entire time with zero signal, found only by hand in an unrelated PR
(OMN-14968 / #2536). This is a false-green class, not a stale-test nit — any
regression covered by an uncollected root is unenforced.

OMN-15410 collected the four remaining roots of that class
(``scripts/ci/tests``, ``scripts/tests``, ``scripts/runtime_build/tests``,
``src/omnibase_infra/services/observability/agent_actions/tests`` — 366 tests,
4 of them RED since 2026-04-02) and hardened this guard against the two seams
that let the class exist in the first place.

What this checks
-----------------
1. **Reachability** (:func:`find_violations`). Every directory containing at
   least one ``test_*.py`` file must sit under a root pytest actually
   collects, i.e. under an entry of ``[tool.pytest.ini_options] testpaths`` in
   ``pyproject.toml``; or be a registered :data:`STANDALONE_PROJECT_ROOTS`
   entry — a directory that owns its own ``pyproject.toml`` (a genuinely
   separate uv (sub-)project, e.g. ``scripts/deploy-agent``) AND has a live
   GitHub Actions workflow that runs its tests. All three legs are verified
   (OMN-15378 / #2553): the ``pyproject.toml`` exists, the workflow file
   exists and references the root, and the workflow is reachable on a pull
   request (its own ``pull_request`` trigger, or a PR-reachable workflow calls
   it via ``uses:``). An allowlist entry with no wiring behind it IS the
   OMN-15378 defect; or be named in :data:`KNOWN_UNCOLLECTED_DEBT`, which is
   EMPTY as of OMN-15410. Do NOT add a root there to make a violation pass —
   wire its collection (path 1 or 2 above) instead; every entry would be a
   live defect.

2. **The ci.yml seam** (:func:`check_full_suite_invocation`). ``testpaths``
   only governs if CI lets it. Before OMN-15410 the full-suite step ran
   ``uv run pytest tests/``, a hardcoded positional path that made the
   *workflow*, not ``pyproject.toml``, the real definition of the suite — a
   root added to ``testpaths`` would still never have run. The step now passes
   no positional path at all, and this check fails closed if one reappears.

3. **The selector seam** (:func:`check_collocated_selector_coverage`). The
   full suite is only one of two pytest steps; a NARROWED smart-selection run
   reaches only what ``scripts/ci/detect_test_paths.py`` maps. A collocated
   root present in ``testpaths`` but absent from that mapping would run only
   on unrelated full-suite escalations. Parity is asserted in both directions.

Why the four OMN-15410 roots are ``testpaths`` entries rather than a move into
``tests/``: they are collocated with the code they cover, two of them declare
their own packages, and once ``testpaths`` is the single source of truth a
move buys no collection guarantee this list does not already provide.

Deliberately excluded: ``scripts/deploy-agent/tests/`` cannot be added to
``testpaths`` or folded into ``tests/`` — both trees declare a top-level
``tests`` package (``tests/__init__.py`` vs
``scripts/deploy-agent/tests/__init__.py``), so pytest raises
``ImportPathMismatchError`` when both are collected in one session (verified
locally 2026-07-29). That is why it is a registered standalone project. The
four OMN-15410 roots have no such conflict — a combined collection of all five
``testpaths`` entries was verified clean (27,436 tests, 2026-07-30).
"""

from __future__ import annotations

import re
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

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

# Uncollected roots explicitly tolerated. EMPTY as of OMN-15410: the four
# entries OMN-15378 grandfathered in (scripts/ci/tests, scripts/tests,
# scripts/runtime_build/tests, and the agent_actions root) are now collected
# via pyproject.toml `testpaths`, so the guard has converged to zero known
# uncollected roots -- its stated OMN-15378 goal.
#
# Do NOT add an entry here to make a violation pass; that is the exact move
# this guard exists to prevent. Wire the root's collection instead (add it to
# `testpaths` plus a COLLOCATED_TEST_ROOTS mapping, or register a standalone
# project with real CI wiring). An entry here is only ever legitimate as a
# same-PR, ticket-bearing record of debt that genuinely cannot be wired yet.
KNOWN_UNCOLLECTED_DEBT: frozenset[str] = frozenset()

_IGNORED_DIR_PARTS = (".git", "__pycache__", ".venv", "node_modules")

# The ci.yml step whose pytest invocation defines the full suite.
FULL_SUITE_STEP_NAME = "Run pytest (full suite)"
CI_WORKFLOW = ".github/workflows/ci.yml"

# GitHub Actions expressions are substituted out before shell tokenization:
# `--junitxml=junit-${{ matrix.split }}.xml` would otherwise tokenize into
# three words, one of which looks like a positional argument.
_GH_EXPRESSION = re.compile(r"\$\{\{.*?\}\}", re.DOTALL)
_GH_EXPRESSION_PLACEHOLDER = "GH_EXPR"

# pytest options that take their value as a SEPARATE token. Needed so
# `--splits 15` is not misread as the positional path `15`. Options written
# `--opt=value` need no entry here.
_PYTEST_VALUE_OPTIONS = frozenset(
    {
        "-c",
        "-k",
        "-m",
        "-n",
        "-o",
        "-p",
        "--deselect",
        "--dist",
        "--group",
        "--ignore",
        "--junitxml",
        "--maxfail",
        "--rootdir",
        "--splits",
        "--timeout",
        "--timeout-method",
    }
)


def collected_roots(repo_root: Path = REPO_ROOT) -> tuple[str, ...]:
    """Return ``testpaths`` from pyproject.toml, POSIX with trailing slash.

    Fails closed: a missing pyproject.toml, a missing ``testpaths`` key, or an
    empty list all raise. An empty ``testpaths`` is especially dangerous — bare
    ``pytest`` would then collect from the rootdir, i.e. the entire repository
    including ``.venv`` — and the full-suite CI step now relies on this list.
    """
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        raise FileNotFoundError(
            f"{pyproject} does not exist; cannot determine collected test roots"
        )
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    ini_options = data.get("tool", {}).get("pytest", {}).get("ini_options", {})
    paths = ini_options.get("testpaths")
    if not paths:
        raise ValueError(
            f"{pyproject} declares no [tool.pytest.ini_options] testpaths; bare "
            "`pytest` would collect the whole repository (OMN-15410)"
        )
    return tuple(str(p).rstrip("/") + "/" for p in paths)


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


def _is_collected(test_dir: str, roots: tuple[str, ...]) -> bool:
    return any(test_dir.startswith(root) for root in roots)


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
    roots = collected_roots(repo_root)
    violations: list[str] = []

    # A testpaths entry that does not exist on disk aborts collection with
    # pytest exit 5 ("no tests ran"), reddening the whole suite for a
    # bookkeeping error. Catch it here, not in a 40-minute CI job.
    for root in roots:
        if not (repo_root / root).is_dir():
            violations.append(
                f"{root}: listed in pyproject.toml testpaths but is not a "
                "directory on disk — pytest would abort collection with exit 5. "
                "Remove the entry or restore the directory."
            )

    for test_dir in find_test_dirs(repo_root):
        if _is_collected(test_dir, roots):
            continue
        if test_dir.rstrip("/") in KNOWN_UNCOLLECTED_DEBT:
            continue
        problem = _standalone_problem(test_dir, repo_root)
        if problem is None:
            continue
        if problem == "unregistered":
            violations.append(
                f"{test_dir}: not under any pyproject.toml testpaths root "
                f"({', '.join(roots)}), not a registered "
                "STANDALONE_PROJECT_ROOTS entry, and not in "
                "KNOWN_UNCOLLECTED_DEBT — no pytest invocation in CI can ever "
                "run these tests (OMN-15378 class). Add it to testpaths (plus a "
                "COLLOCATED_TEST_ROOTS mapping in "
                "scripts/ci/detect_test_paths.py if it lives outside tests/), or "
                "register it in STANDALONE_PROJECT_ROOTS with real CI wiring."
            )
        else:
            violations.append(f"{test_dir}: {problem}")
    return violations


def _full_suite_run_block(repo_root: Path) -> str:
    """Return the shell body of ci.yml's full-suite pytest step."""
    workflow_path = repo_root / CI_WORKFLOW
    if not workflow_path.is_file():
        raise FileNotFoundError(f"{CI_WORKFLOW} does not exist")
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    for job in (workflow.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            if isinstance(step, dict) and step.get("name") == FULL_SUITE_STEP_NAME:
                return str(step.get("run", ""))
    raise LookupError(
        f"{CI_WORKFLOW} has no step named {FULL_SUITE_STEP_NAME!r}; the "
        "full-suite invocation cannot be verified (OMN-15410)"
    )


def positional_pytest_args(run_block: str) -> list[str]:
    """Return the positional (non-option) arguments a run block passes pytest.

    GitHub expressions are replaced with a placeholder first, then the block is
    tokenized as shell so a quoted marker expression stays one token.
    """
    text = _GH_EXPRESSION.sub(_GH_EXPRESSION_PLACEHOLDER, run_block)
    text = text.replace("\\\n", " ")
    tokens = shlex.split(text)
    if "pytest" not in tokens:
        return []
    positionals: list[str] = []
    skip_next = False
    for token in tokens[tokens.index("pytest") + 1 :]:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            skip_next = token in _PYTEST_VALUE_OPTIONS
            continue
        positionals.append(token)
    return positionals


def check_full_suite_invocation(repo_root: Path = REPO_ROOT) -> list[str]:
    """Fail closed when ci.yml's full suite names its own paths (OMN-15410).

    A positional path there silently overrides ``testpaths``, which is how four
    collectable roots went unrun for months while ``pyproject.toml`` looked
    correct.
    """
    positionals = positional_pytest_args(_full_suite_run_block(repo_root))
    if not positionals:
        return []
    return [
        f"{CI_WORKFLOW} step {FULL_SUITE_STEP_NAME!r} passes positional path(s) "
        f"{positionals} to pytest, overriding pyproject.toml testpaths. The "
        "full suite must pass NO positional path so it inherits every collected "
        "root (OMN-15410); use --ignore to exclude, never a positional include."
    ]


def check_collocated_selector_coverage(repo_root: Path = REPO_ROOT) -> list[str]:
    """Assert testpaths <-> COLLOCATED_TEST_ROOTS parity (OMN-15410).

    A collocated root collected only by the full suite is reachable solely via
    unrelated escalations; the change-aware selector must be able to select it
    from a diff that touches the code it covers.
    """
    sys.path.insert(0, str(repo_root))
    try:
        from scripts.ci.detect_test_paths import (
            COLLOCATED_TEST_ROOTS,
            TESTS_PREFIX,
        )
    finally:
        sys.path.remove(str(repo_root))

    violations: list[str] = []
    mapped = set(COLLOCATED_TEST_ROOTS.values())
    declared = {root for root in collected_roots(repo_root) if root != TESTS_PREFIX}

    for root in sorted(declared - mapped):
        violations.append(
            f"{root}: collected via pyproject.toml testpaths but no "
            "COLLOCATED_TEST_ROOTS entry in scripts/ci/detect_test_paths.py maps "
            "any source prefix to it — a narrowed smart-selection run can never "
            "select it (OMN-15410)."
        )
    for root in sorted(mapped - declared):
        violations.append(
            f"{root}: mapped by COLLOCATED_TEST_ROOTS in "
            "scripts/ci/detect_test_paths.py but absent from pyproject.toml "
            "testpaths — the selector would hand pytest a path the full suite "
            "never collects (OMN-15410)."
        )
    return violations


def main() -> int:
    violations = (
        find_violations()
        + check_full_suite_invocation()
        + check_collocated_selector_coverage()
    )
    if violations:
        print("FAIL: test-collection defect(s) detected (OMN-15378/OMN-15410 class):")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print(
        "OK: every tests/ directory is reachable by a wired pytest invocation, the "
        "full suite inherits pyproject testpaths, and every collocated root is "
        "selector-reachable."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
