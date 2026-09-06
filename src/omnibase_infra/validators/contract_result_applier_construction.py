# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Forbid direct ``DispatchResultApplier(...)`` construction outside its factory.

OMN-15468 — enforcement, not detection (CLAUDE.md rule 5).

``DispatchResultApplier`` carries TWO contract-derived routing inputs, and both
default to empty:

* ``output_topic_map`` — the contract's ``published_events``, which is what
  routes a returned ``…Failed`` CLASS to the contract's failure topic. Empty
  ⇒ every returned class resolves to the single ``output_topic`` fallback, so
  class-based routing is not merely unused, it is dead.
* ``failure_terminal_topics`` — the contract's declared failure terminal, which
  is where ``apply_failure_terminal_guard`` re-routes a payload that STATES a
  failure verdict. Empty ⇒ the guard takes its ``len(...) != 1`` branch,
  returns the success topic, and logs NOTHING.

An applier built by hand therefore looks correct, boots green, publishes
happily, and silently republishes failure verdicts onto the contract's SUCCESS
terminal. Measured live on the ``.201`` dev lane 2026-09-05: 18 of the trailing
25 records on ``onex.evt.omnimarket.delegate-skill-completed.v1`` carried
``status="failed"`` with a typed ``terminal_failure_cause``, published by
``service_kernel``'s hand-rolled by-NAME registration — which takes precedence
over the contract-derived wiring in ``_subscribe_contract_topics``, so the
correct applier was constructed nowhere.

The rule this gate enforces: every applier in ``src/`` is built through
``build_contract_result_applier`` (derives both inputs from the contract path)
or ``build_static_result_applier`` (no contract path available; requires the
caller to STATE ``failure_terminal_topics``, including a deliberate ``()``).
Both live in ``omnibase_infra.runtime.service_dispatch_result_applier``, which
is the one module allowed to call the constructor.

Tests construct the class directly on purpose — a unit test pinning the
constructor's own behaviour must be able to reach it — so only ``src/`` is
scanned.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

CLASS_NAME = "DispatchResultApplier"

# The module that DEFINES the class and both factories. Only construction site.
FACTORY_MODULE_RELPATH = "omnibase_infra/runtime/service_dispatch_result_applier.py"

# Vacuity guard: a scan that finds no Python files at all proves nothing.
MIN_EXPECTED_FILES = 100


@dataclass(frozen=True)
class Violation:
    """One direct constructor call outside the factory module."""

    path: Path
    line: int


def _resolve_call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def scan_source(source: str, path: Path) -> list[Violation]:
    """Return every direct constructor call in one parsed source file."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        Violation(path=path, line=node.lineno)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _resolve_call_name(node) == CLASS_NAME
    ]


def scan(root: Path) -> tuple[list[Violation], int]:
    """Scan a source tree; returns (violations, files scanned)."""
    violations: list[Violation] = []
    scanned = 0
    for path in sorted(root.rglob("*.py")):
        if path.as_posix().endswith(FACTORY_MODULE_RELPATH):
            continue
        scanned += 1
        violations.extend(scan_source(path.read_text(encoding="utf-8"), path))
    return violations, scanned


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    scan_root = Path(args[0]) if args else Path("src/omnibase_infra")
    if not scan_root.exists():
        sys.stderr.write(
            f"[contract-result-applier-construction] FAIL: scan root {scan_root} "
            f"does not exist.\n"
        )
        return 1

    violations, scanned = scan(scan_root)
    if scanned < MIN_EXPECTED_FILES:
        sys.stderr.write(
            f"[contract-result-applier-construction] FAIL (vacuity guard): only "
            f"{scanned} Python files scanned under {scan_root} (expected >= "
            f"{MIN_EXPECTED_FILES}). A gate over a collapsed set proves nothing.\n"
        )
        return 1

    if violations:
        sys.stderr.write(
            f"[contract-result-applier-construction] FAIL: {len(violations)} direct "
            f"{CLASS_NAME}(...) construction(s) outside "
            f"{FACTORY_MODULE_RELPATH}:\n"
        )
        for violation in violations:
            sys.stderr.write(f"  - {violation.path}:{violation.line}\n")
        sys.stderr.write(
            "\n  Both contract-derived routing inputs default to empty, and an "
            "empty failure-terminal list makes the OMN-15468 failure-verdict "
            "guard inert WITHOUT logging anything. Build the applier through "
            "`build_contract_result_applier(contract_path=..., publish_topics=...)`, "
            "or `build_static_result_applier(..., failure_terminal_topics=...)` "
            "when no contract path is available — that argument is required "
            "precisely so the question is answered in the source.\n"
        )
        return 1

    sys.stderr.write(
        f"[contract-result-applier-construction] OK: {scanned} files scanned, "
        f"0 direct {CLASS_NAME}(...) constructions outside the factory module.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
