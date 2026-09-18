# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A workflow's inline Python may not import a name its module does not export (OMN-18684).

What this exists to prevent
---------------------------
``dev-lane-liveness.yml``'s ``saturation-record`` job is the G7 saturation
monitor the operator consented to on 2026-09-07. Its first step is an inline
heredoc that did::

    from runner_route_decision import probe_fleet, load_route_policy

``OMN-18412`` (squash ``341ac00cf``) rewrote the routing decision as a COMPUTE
node and renamed that loader to ``load_contract_policy``. The workflow was not
part of that change and nothing connected the two, so the step began failing on
its very first run after the rename and on every run after that::

    ImportError: cannot import name 'load_route_policy' from 'runner_route_decision'
    ##[error]Process completed with exit code 1.

The failure is at the FIRST step, before any fleet or lab data is gathered, so
the monitor produced no record at all -- not a stale one. Three consecutive
scheduled runs (35337065964, 35336042566, 35335179281) were red this way before
anyone read the log.

Why a checker rather than a reviewer
------------------------------------
An inline heredoc is invisible to every tool that makes a rename safe. Ruff and
mypy see a YAML string; an IDE rename touches call sites in ``.py`` files and
stops there; the import only resolves on a hosted runner, minutes into a
scheduled run nobody watches. CLAUDE.md rule 5: a rename that cannot be
mechanically checked will keep being made.

Contract
--------
For every ``run:`` step in ``.github/workflows/*.y[a]ml`` and
``.github/actions/*/action.y[a]ml`` that contains a ``from <module> import
<names>`` statement, where ``<module>`` resolves to a repository file under a
directory the same step puts on ``sys.path``, every imported name must be a
module-level name of that file.

The target module is parsed with :mod:`ast` and never imported, so this checker
costs nothing and cannot be defeated by a module whose import has side effects
or unavailable dependencies -- which is precisely the situation of the modules
it guards.

Exit codes: ``0`` clean, ``1`` violations found (printed one per line).

Scope, and why it is drawn here
-------------------------------
ONLY imports whose module resolves to a file in this repository are checked. A
``from json import loads`` in a heredoc resolves to no repository file and is
skipped rather than guessed at -- a checker that reported third-party and
stdlib imports it could not see would be noise, and noisy checkers get turned
off (rule 5 again).

SEARCH PATHS ARE READ FROM THE STEP, not assumed. A step reaches its module by
``sys.path.insert(0, "scripts/ci")``; that literal is what this reads. A step
that inserts nothing is checked against no search path and therefore resolves
nothing, which is correct: it cannot import a repository module either.

Wired as a pre-commit hook AND as a CI step, and asserted by
``tests/ci/test_workflow_inline_python_imports_omn18684.py``.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: ``sys.path.insert(0, "scripts/ci")`` -- the literal a step uses to reach a
#: repository module. Only string literals are read; a computed path is not a
#: search path this checker claims to know.
_SYS_PATH_INSERT = re.compile(
    r"""sys\.path\.insert\(\s*\d+\s*,\s*["']([^"']+)["']\s*\)"""
)

#: ``from <module> import a, b`` / ``from <module> import (a, b)``. Relative
#: imports (a leading dot) are not repository-module imports in a heredoc run
#: from the repository root and are left alone.
_FROM_IMPORT = re.compile(
    r"^\s*from\s+([A-Za-z_][A-Za-z0-9_]*)\s+import\s+\(?([^)\n#]+)\)?\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Violation:
    """One imported name that its target module does not define."""

    workflow: Path
    job: str
    step: str
    module: Path
    name: str

    def render(self) -> str:
        return (
            f"{self.workflow}: job '{self.job}' step '{self.step}' imports "
            f"'{self.name}' from '{self.module}', which does not define it"
        )


def module_level_names(source: str) -> frozenset[str]:
    """Every name a module binds at module level, without importing it.

    Covers the four shapes an inline heredoc can legitimately import: function
    and class definitions, module-level assignments (constants), and re-exported
    imports.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


def imported_names(clause: str) -> tuple[str, ...]:
    """The bound names of an import clause, honouring ``as`` aliases.

    The name that must EXIST in the target module is the one on the left of
    ``as``, so this returns the original names rather than the local aliases.
    """
    found: list[str] = []
    for part in clause.split(","):
        candidate = part.strip().split(" as ")[0].strip()
        if candidate and candidate != "*":
            found.append(candidate)
    return tuple(found)


def check_run_block(
    script: str, *, root: Path, workflow: Path, job: str, step: str
) -> list[Violation]:
    """Resolve every repository-module import in one ``run:`` block."""
    search_paths = [
        root / raw
        for raw in _SYS_PATH_INSERT.findall(script)
        if not raw.startswith("/")
    ]
    if not search_paths:
        return []

    violations: list[Violation] = []
    for module_name, clause in _FROM_IMPORT.findall(script):
        for directory in search_paths:
            module_path = directory / f"{module_name}.py"
            if not module_path.is_file():
                continue
            try:
                defined = module_level_names(module_path.read_text(encoding="utf-8"))
            except (
                SyntaxError
            ) as exc:  # pragma: no cover - a broken module is its own gate
                raise SystemExit(f"{module_path}: could not be parsed: {exc}") from exc
            violations.extend(
                Violation(
                    workflow=workflow.relative_to(root),
                    job=job,
                    step=step,
                    module=module_path.relative_to(root),
                    name=name,
                )
                for name in imported_names(clause)
                if name not in defined
            )
            break
    return violations


def _steps(document: Any) -> list[tuple[str, dict[str, Any]]]:
    """Every ``(job name, step)`` pair in a workflow or composite action."""
    pairs: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(document, dict):
        return pairs

    jobs = document.get("jobs")
    if isinstance(jobs, dict):
        for job_name, job in jobs.items():
            if isinstance(job, dict) and isinstance(job.get("steps"), list):
                pairs.extend(
                    (str(job_name), step)
                    for step in job["steps"]
                    if isinstance(step, dict)
                )

    runs = document.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("steps"), list):
        pairs.extend(("runs", step) for step in runs["steps"] if isinstance(step, dict))
    return pairs


def check_file(path: Path, *, root: Path) -> list[Violation]:
    """Every violation in one workflow or composite-action file."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SystemExit(f"{path}: could not be parsed as YAML: {exc}") from exc

    violations: list[Violation] = []
    for job_name, step in _steps(document):
        script = step.get("run")
        if not isinstance(script, str):
            continue
        violations.extend(
            check_run_block(
                script,
                root=root,
                workflow=path,
                job=job_name,
                step=str(step.get("name") or step.get("id") or "<unnamed>"),
            )
        )
    return violations


def discover(root: Path) -> list[Path]:
    """Workflow and composite-action files, in a stable order."""
    found: list[Path] = []
    for pattern in ("*.yml", "*.yaml"):
        found.extend((root / ".github/workflows").glob(pattern))
        found.extend((root / ".github/actions").glob(f"*/action{pattern[1:]}"))
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root to scan (default: this checkout).",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    violations: list[Violation] = []
    for path in discover(root):
        violations.extend(check_file(path, root=root))

    if violations:
        print(
            "Inline workflow Python imports a name its module does not define "
            "(OMN-18684):",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation.render()}", file=sys.stderr)
        return 1

    print("Inline workflow Python imports resolve against their modules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
