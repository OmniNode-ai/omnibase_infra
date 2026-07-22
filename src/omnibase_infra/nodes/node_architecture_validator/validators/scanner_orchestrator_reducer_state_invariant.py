# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""ARCH-005: ORCHESTRATOR/REDUCER handler state-persistence invariant (OMN-14222).

Surfaced by the OMN-14208 declarative-orchestrator census (2026-07-09),
operator-confirmed 2026-07-09.

Invariant: no ORCHESTRATOR or REDUCER handler may hold state (a ``ClassVar``
container or a module-level mutable container) that survives between two
separate ``handle()`` invocations for the same logical unit of work. All such
state must be a DB row scoped by ``tenant_id`` (+ secondary key), fetched fresh
on each ``handle()`` and persisted via the node's declared output channel. This
is what makes the single-runtime "stateless" premise actually true.

Under the corrected invariant the *key shape* is irrelevant -- the presence of
a mutable container at all inside an orchestrator/reducer handler file is the
defect, not what it is keyed by. This is a ZERO-TOLERANCE gate (no ratchet
baseline, unlike the ARCH-004 Signal A/B scanners): the live repo carries no
unexempted violations as of OMN-14222, so any new one fails the gate outright.
A finding is cleared with an explicit, reasoned exemption comment -- never a
baseline entry -- because "this container is fine" is a per-line engineering
judgment call, not accepted debt to be tracked down over time.

Scope (this scanner intentionally does NOT attempt):
    - Instance attributes assigned in ``__init__`` (e.g. ``self._cache = {}``).
      Whether such state actually "survives between handle() invocations"
      depends on singleton-vs-per-call instantiation semantics that are a
      runtime/wiring fact, not a static AST fact. Flagging every instance
      attribute would produce mass false positives on ordinary per-request
      dataclass-style handlers. Tracked as a known gap, not silently dropped.
    - Bare ``Service*``/``*Manager`` classes outside the node/handler tree
      (e.g. ``ServiceSavingsEstimator``) -- those are a DISTINCT non-canonical
      primitive concern (CLAUDE.md Rule 7a), not this handler-state invariant.

Detection (two static shapes, matching the ticket's concrete scope):
    1. A module-level ``Assign``/``AnnAssign`` in a ``handlers/*.py`` file
       (under a node directory whose ``contract.yaml`` declares
       ``node_type: ORCHESTRATOR*`` or ``*REDUCER*``) whose value is a mutable
       container literal or constructor call (``dict``/``list``/``set``/
       ``OrderedDict``/``defaultdict``/``Counter``).
    2. A class body ``AnnAssign`` annotated ``ClassVar[<mutable container>]``
       inside the same file set.

``__all__`` is always exempt (universal Python export idiom, never handler
state). Any other finding is cleared by adding an ``EXEMPTION_MARKER`` comment
(``# orchestrator-reducer-state-ok: <reason>``) on the same line or the line
immediately preceding the assignment.

Modes (mirrors the ARCH-004 scanner CLI shape, minus the baseline):
    * ``--check-all --report`` -- scan every node directory, report every
      violation. Exit 1 if any violation is found (this gate has zero
      tolerance; there is no accepted-debt baseline to compare against).
    * ``--check-changed`` -- scan only node directories touched by a
      changeset (pre-commit / PR-scoped invocation).

Related:
    - OMN-14222 (this ticket)
    - OMN-14208 (declarative-orchestrator census, origin of the invariant)
    - OMN-13472 / OMN-13486 (ARCH-004 Signal A / Signal B -- sibling ratchets,
      precedent for the discover/scan/CLI shape, extended here)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from omnibase_infra.nodes.node_architecture_validator.validators.scanner_imperative_orchestrator_ratchet import (
    discover_node_dirs,
    node_dirs_for_changed_files,
)

#: Rule identifier for this invariant (ARCH-005).
STATE_INVARIANT_RULE_ID = "ARCH-005"

#: Names that must NEVER be flagged regardless of container shape.
EXEMPT_NAMES = frozenset({"__all__"})

#: Container constructor names treated as "mutable" for this gate. Dict/List/
#: Set literals are matched by AST node type directly (see
#: ``_is_mutable_value``); this set additionally catches constructor calls.
MUTABLE_CONTAINER_CALL_NAMES = frozenset(
    {"dict", "list", "set", "OrderedDict", "defaultdict", "Counter"}
)

#: Inline (or line-above) comment that clears a finding. The reason text after
#: the colon is not machine-checked; it exists so review sees *why* a human
#: judged the container safe (e.g. a static read-only FSM transition table).
EXEMPTION_MARKER = "orchestrator-reducer-state-ok:"


@dataclass(frozen=True)  # internal-dataclass-ok: scanner-internal finding row
class ModelStateInvariantFinding:
    """One ORCHESTRATOR/REDUCER handler-state invariant violation."""

    repo: str
    node: str
    file_path: str
    line: int
    name: str
    shape: str  # "module-level" or "classvar"

    def format(self) -> str:
        return (
            f"{self.file_path}:{self.line}: [{self.shape}] '{self.name}' is a "
            f"mutable container held by an ORCHESTRATOR/REDUCER handler file "
            f"(node={self.node}). Move this state to a tenant_id-scoped DB row "
            f"fetched/persisted through the node's declared channel, or add "
            f"'# {EXEMPTION_MARKER} <reason>' if this is a genuinely static, "
            f"never-mutated table."
        )


@dataclass  # internal-dataclass-ok: scanner-internal aggregate scan result
class StateInvariantScanResult:
    """Aggregated ARCH-005 scan result across a set of node directories."""

    repo: str
    findings: list[ModelStateInvariantFinding] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Node-type classification
# --------------------------------------------------------------------------- #
def _load_node_type(contract_path: Path) -> str:
    """Return the declared ``node_type`` string, or '' if unreadable."""
    try:
        raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return ""
    if not isinstance(raw, dict):
        return ""
    return str(raw.get("node_type", ""))


def is_target_node_type(node_type: str) -> bool:
    """Return True if ``node_type`` is an ORCHESTRATOR* or *REDUCER* contract.

    Unlike the ARCH-004 scanners (which exempt reducers), THIS invariant
    targets both -- a reducer accumulating in-memory session state is exactly
    as much a violation as an orchestrator doing so (ticket Invariant text:
    "No ORCHESTRATOR or REDUCER handler may hold state...").
    """
    upper = node_type.upper()
    return upper.startswith("ORCHESTRATOR") or "REDUCER" in upper


def target_node_dirs(node_dirs: list[Path]) -> list[Path]:
    """Filter ``node_dirs`` down to those with an ORCHESTRATOR*/*REDUCER* contract."""
    targets: list[Path] = []
    for node_dir in node_dirs:
        contract_path = node_dir / "contract.yaml"
        if not contract_path.is_file():
            continue
        if is_target_node_type(_load_node_type(contract_path)):
            targets.append(node_dir)
    return targets


# --------------------------------------------------------------------------- #
# AST scanning
# --------------------------------------------------------------------------- #
def _is_mutable_value(value: ast.expr) -> bool:
    """Return True if ``value`` is a mutable-container literal or constructor call."""
    if isinstance(value, (ast.Dict, ast.List, ast.Set)):
        return True
    if isinstance(value, ast.Call):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        return name in MUTABLE_CONTAINER_CALL_NAMES
    return False


def _classvar_inner_name(annotation: ast.expr) -> str | None:
    """If ``annotation`` is ``ClassVar[X]`` (bare or ``typing.ClassVar[X]``), return X's name."""
    if not isinstance(annotation, ast.Subscript):
        return None
    base = annotation.value
    base_name = base.id if isinstance(base, ast.Name) else getattr(base, "attr", None)
    if base_name != "ClassVar":
        return None
    inner = annotation.slice
    if isinstance(inner, ast.Subscript):
        inner = inner.value
    if isinstance(inner, ast.Name):
        return inner.id
    if isinstance(inner, ast.Attribute):
        return inner.attr
    return None


def _has_exemption_comment(lines: list[str], lineno: int) -> bool:
    """Check for the exemption marker on ``lineno`` or a contiguous comment
    block immediately above it.

    ``lineno`` is 1-indexed (as produced by the ``ast`` module). Walking
    upward through contiguous ``#``-prefixed lines lets a multi-line
    explanatory comment carry the marker on any line, not only the one
    directly touching the assignment.
    """
    if 1 <= lineno <= len(lines) and EXEMPTION_MARKER in lines[lineno - 1]:
        return True
    candidate = lineno - 1
    while 1 <= candidate <= len(lines) and lines[candidate - 1].strip().startswith("#"):
        if EXEMPTION_MARKER in lines[candidate - 1]:
            return True
        candidate -= 1
    return False


def _assign_names(node: ast.stmt) -> list[str]:
    """Return the assigned name(s) for a module-level ``Assign``/``AnnAssign``.

    Typed as ``ast.stmt`` (the common base) rather than
    ``ast.Assign | ast.AnnAssign`` to stay within the repo's non-optional
    union ratchet (OMN-13412-adjacent gate) -- both branches are reached only
    via ``isinstance`` narrowing below, never blindly.
    """
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def scan_handler_file(
    path: Path, *, repo: str, node: str, repo_root: Path | None = None
) -> list[ModelStateInvariantFinding]:
    """Scan one handler file for module-level / ClassVar mutable-container state."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    lines = source.splitlines()
    display_path = _relative_path(path, repo_root)
    findings: list[ModelStateInvariantFinding] = []

    # Shape 1: module-level mutable container assignment.
    for stmt in tree.body:
        if not isinstance(stmt, (ast.Assign, ast.AnnAssign)) or stmt.value is None:
            continue
        names = _assign_names(stmt)
        if not names or any(n in EXEMPT_NAMES for n in names):
            continue
        if not _is_mutable_value(stmt.value):
            continue
        if _has_exemption_comment(lines, stmt.lineno):
            continue
        for name in names:
            findings.append(
                ModelStateInvariantFinding(
                    repo=repo,
                    node=node,
                    file_path=display_path,
                    line=stmt.lineno,
                    name=name,
                    shape="module-level",
                )
            )

    # Shape 2: ClassVar[<mutable>] class attribute.
    for cls_node in ast.walk(tree):
        if not isinstance(cls_node, ast.ClassDef):
            continue
        for stmt in cls_node.body:
            if not isinstance(stmt, ast.AnnAssign) or not isinstance(
                stmt.target, ast.Name
            ):
                continue
            if stmt.target.id in EXEMPT_NAMES:
                continue
            inner = _classvar_inner_name(stmt.annotation)
            if inner is None or inner not in MUTABLE_CONTAINER_CALL_NAMES:
                continue
            if _has_exemption_comment(lines, stmt.lineno):
                continue
            findings.append(
                ModelStateInvariantFinding(
                    repo=repo,
                    node=node,
                    file_path=display_path,
                    line=stmt.lineno,
                    name=stmt.target.id,
                    shape="classvar",
                )
            )

    return findings


def _relative_path(path: Path, repo_root: Path | None) -> str:
    """Render ``path`` repo-relative (never machine-absolute; Rule #6)."""
    if repo_root is not None:
        try:
            return str(path.resolve().relative_to(repo_root.resolve()))
        except ValueError:
            pass
    parts = path.parts
    if "src" in parts:
        return str(Path(*parts[parts.index("src") :]))
    return path.name


def scan_node_dirs_for_state_violations(
    repo: str, node_dirs: list[Path], repo_root: Path | None = None
) -> StateInvariantScanResult:
    """Run ARCH-005 over ``node_dirs`` (already filtered to ORCHESTRATOR*/REDUCER*)."""
    result = StateInvariantScanResult(repo=repo)
    for node_dir in node_dirs:
        handlers_dir = node_dir / "handlers"
        if not handlers_dir.is_dir():
            continue
        for py_file in sorted(handlers_dir.glob("*.py")):
            result.findings.extend(
                scan_handler_file(
                    py_file, repo=repo, node=node_dir.name, repo_root=repo_root
                )
            )
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_repo_root(arg: str | None) -> Path:
    if arg:
        return Path(arg).resolve()
    return Path(__file__).resolve().parents[5]


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the ARCH-005 orchestrator/reducer state invariant gate."""
    parser = argparse.ArgumentParser(
        prog="orchestrator-reducer-state-invariant",
        description=(
            "ARCH-005 ORCHESTRATOR/REDUCER handler state-persistence invariant "
            "(OMN-14222). Zero-tolerance: fails on any unexempted ClassVar or "
            "module-level mutable container in an orchestrator/reducer "
            "handlers/ file."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-all", action="store_true")
    mode.add_argument("--check-changed", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--repo-name", default="omnibase_infra")
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv)

    repo_root = _resolve_repo_root(args.repo_root)

    if args.check_all:
        node_dirs = discover_node_dirs(repo_root)
    else:
        node_dirs = node_dirs_for_changed_files(repo_root, args.files)

    node_dirs = target_node_dirs(node_dirs)
    result = scan_node_dirs_for_state_violations(
        args.repo_name, node_dirs, repo_root=repo_root
    )

    if args.report or args.check_all:
        print(
            f"ARCH-005 orchestrator/reducer state-invariant scan "
            f"({args.repo_name}): {len(result.findings)} finding(s) across "
            f"{len(node_dirs)} target node dir(s)."
        )
        for finding in result.findings:
            print(f"  {finding.format()}")

    if result.findings:
        print("\nARCH-005 state-invariant gate FAILED:", file=sys.stderr)
        for finding in result.findings:
            print(f"  - {finding.format()}", file=sys.stderr)
        return 1

    return 0


__all__ = [
    "EXEMPTION_MARKER",
    "EXEMPT_NAMES",
    "MUTABLE_CONTAINER_CALL_NAMES",
    "STATE_INVARIANT_RULE_ID",
    "ModelStateInvariantFinding",
    "StateInvariantScanResult",
    "discover_node_dirs",
    "is_target_node_type",
    "main",
    "node_dirs_for_changed_files",
    "scan_handler_file",
    "scan_node_dirs_for_state_violations",
    "target_node_dirs",
]


if __name__ == "__main__":
    raise SystemExit(main())
