# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Block Any annotations in handler handle() and __init__ signatures (OMN-10820).

Gate 2 extension: covers Handler* class method signatures that the existing
loose_typing_in_internal gate skips (it only checks Pydantic/Protocol classes).

Banned forms in handle() and __init__() of any Handler* class:
  - param: Any
  - param: dict[str, Any] / Dict[str, Any] / Mapping[str, Any]
  - -> Any  (return type)

Suppression: ``# substrate-allow: <reason>`` on the offending line suppresses.
``# ONEX_EXCLUDE: any_type`` and ``# ai-slop-ok`` are also recognised.

Usage (pre-commit):
    uv run python -m omnibase_infra.validators.handler_any_signature src/omnibase_infra

Usage (budget/ratchet mode):
    uv run python -m omnibase_infra.validators.handler_any_signature \\
        --max-violations 0 src/omnibase_infra
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")

_DICT_LIKE_NAMES = frozenset({"dict", "Dict", "Mapping"})

_ALLOW_RE = re.compile(
    r"#\s*(substrate-allow:|ONEX_EXCLUDE:|ai-slop-ok)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class AnySignatureFinding:
    path: Path
    line: int
    class_name: str
    method_name: str
    detail: str

    def format(self) -> str:
        return (
            f"{self.path}:{self.line}: {self.class_name}.{self.method_name}(): "
            f"banned Any in handler signature — {self.detail}"
        )


def _is_any(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and node.id == "Any"


def _is_dict_any(node: ast.expr) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    if not (isinstance(node.value, ast.Name) and node.value.id in _DICT_LIKE_NAMES):
        return False
    slc = node.slice
    if not isinstance(slc, ast.Tuple) or len(slc.elts) < 2:
        return False
    return _is_any(slc.elts[1])


def _contains_any(node: ast.expr) -> bool:
    """Return True if Any appears anywhere in the annotation tree."""
    return any(
        isinstance(child, ast.expr) and _is_any(child) for child in ast.walk(node)
    )


def _suppressed(source_lines: list[str], lineno: int) -> bool:
    if lineno < 1 or lineno > len(source_lines):
        return False
    return bool(_ALLOW_RE.search(source_lines[lineno - 1]))


def validate_file(path: Path) -> list[AnySignatureFinding]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    if "Any" not in source:
        return []

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    lines = source.splitlines()
    findings: list[AnySignatureFinding] = []

    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue

        for node in ast.walk(cls):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name not in ("handle", "__init__"):
                continue

            all_args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            if node.args.vararg is not None:
                all_args.append(node.args.vararg)
            if node.args.kwarg is not None:
                all_args.append(node.args.kwarg)
            for arg in all_args:
                if arg.arg in ("self", "cls"):
                    continue
                ann = arg.annotation
                if ann is None or not _contains_any(ann):
                    continue
                lineno = getattr(arg, "lineno", node.lineno)
                if _suppressed(lines, lineno):
                    continue
                findings.append(
                    AnySignatureFinding(
                        path=path,
                        line=lineno,
                        class_name=cls.name,
                        method_name=node.name,
                        detail=f"param '{arg.arg}: {ast.unparse(ann)}'",
                    )
                )

            if node.name == "handle" and node.returns is not None:
                if _contains_any(node.returns):
                    lineno = node.lineno
                    if not _suppressed(lines, lineno):
                        findings.append(
                            AnySignatureFinding(
                                path=path,
                                line=lineno,
                                class_name=cls.name,
                                method_name=node.name,
                                detail=f"return type '-> {ast.unparse(node.returns)}'",
                            )
                        )

    return findings


def validate_paths(paths: Sequence[Path]) -> list[AnySignatureFinding]:
    findings: list[AnySignatureFinding] = []
    for path in _iter_python_files(paths):
        findings.extend(validate_file(path))
    return findings


def _iter_python_files(paths: Sequence[Path]) -> Iterator[Path]:
    scan_paths = paths or (DEFAULT_SCAN_ROOT,)
    for path in scan_paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(
                p for p in path.rglob("*.py") if "__pycache__" not in p.parts
            )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Block Any annotations in handler handle() and __init__() signatures. "
            "Use --max-violations N for ratchet mode (budget gate)."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_SCAN_ROOT],
    )
    parser.add_argument(
        "--max-violations",
        type=int,
        default=None,
        metavar="N",
        help="Allow up to N violations (ratchet mode). Omit to require zero.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else list(argv))
    findings = validate_paths(args.paths)

    if not findings and args.max_violations is None:
        return 0

    for finding in findings:
        sys.stderr.write(f"  {finding.format()}\n")

    if args.max_violations is not None:
        count = len(findings)
        if count > args.max_violations:
            sys.stderr.write(
                f"[handler-any-gate] {count} violations exceed budget {args.max_violations}\n"
                "  Add '# substrate-allow: <reason>' on the offending line to suppress.\n"
                "  To reduce budget: lower --max-violations in .pre-commit-config.yaml.\n"
            )
            return 1
        if findings:
            sys.stderr.write(
                f"[handler-any-gate] {count} violations within budget {args.max_violations} "
                f"(target: 0). Fix to reduce budget.\n"
            )
        return 0

    if findings:
        sys.stderr.write(
            "\nHandler handle()/\\__init__() signatures must not use Any.\n"
            "Replace with concrete types or Pydantic models.\n"
            "To suppress a justified exception: add '# substrate-allow: <reason>'.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
