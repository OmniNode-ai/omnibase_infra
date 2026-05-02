# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Block Plugin* classes from owning daemon, service, or worker lifecycles."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SCAN_ROOT = Path("src/omnibase_infra")
LIFECYCLE_TERMS = frozenset(
    (
        "Daemon",
        "Service",
        "Worker",
        "Runtime",
        "Consumer",
        "Publisher",
        "Controller",
        "Server",
        "Client",
    )
)
COMPUTE_PLUGIN_BASE = "PluginComputeBase"


@dataclass(frozen=True, slots=True)
class LifecycleClassFinding:
    """A banned Plugin* lifecycle class found in source."""

    path: Path
    line: int
    column: int
    class_name: str
    bases: tuple[str, ...]
    reason: str

    def format(self) -> str:
        bases = ", ".join(self.bases) if self.bases else "<no bases>"
        return (
            f"{self.path}:{self.line}:{self.column + 1}: {self.class_name} "
            f"uses Plugin* lifecycle ownership ({self.reason}); bases: {bases}"
        )


def validate_paths(paths: Sequence[Path]) -> list[LifecycleClassFinding]:
    """Validate Python files under the provided paths."""
    findings: list[LifecycleClassFinding] = []
    for path in _iter_python_files(paths):
        findings.extend(validate_file(path))
    return findings


def validate_file(path: Path) -> list[LifecycleClassFinding]:
    """Validate one Python file for banned Plugin* lifecycle classes."""
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [
            LifecycleClassFinding(
                path=path,
                line=1,
                column=0,
                class_name="<parse-error>",
                bases=(),
                reason=f"could not decode as UTF-8: {exc}",
            )
        ]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [
            LifecycleClassFinding(
                path=path,
                line=exc.lineno or 1,
                column=exc.offset or 0,
                class_name="<syntax-error>",
                bases=(),
                reason=exc.msg,
            )
        ]

    findings: list[LifecycleClassFinding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            finding = _validate_class(path, node)
            if finding is not None:
                findings.append(finding)
    return findings


def _validate_class(path: Path, node: ast.ClassDef) -> LifecycleClassFinding | None:
    if not node.name.startswith("Plugin"):
        return None

    base_names = tuple(
        name for base in node.bases if (name := _qualified_expr_name(base)) is not None
    )
    if _is_compute_base_definition(path, node):
        return None
    if _is_direct_compute_plugin(base_names):
        return None

    reason = _lifecycle_reason(node.name, base_names)
    if reason is None:
        return None

    return LifecycleClassFinding(
        path=path,
        line=node.lineno,
        column=node.col_offset,
        class_name=node.name,
        bases=base_names,
        reason=reason,
    )


def _is_compute_base_definition(path: Path, node: ast.ClassDef) -> bool:
    return node.name == COMPUTE_PLUGIN_BASE and path.name == "plugin_compute_base.py"


def _is_direct_compute_plugin(base_names: Iterable[str]) -> bool:
    names = tuple(base_names)
    return bool(names) and all(
        _simple_name(base) == COMPUTE_PLUGIN_BASE for base in names
    )


def _lifecycle_reason(class_name: str, base_names: Sequence[str]) -> str | None:
    class_term = _matching_lifecycle_term(class_name)
    if class_term is not None:
        return f"class name contains {class_term}"

    for base in base_names:
        base_term = _matching_lifecycle_term(base)
        if base_term is not None:
            return f"base class {base} contains {base_term}"
    return None


def _matching_lifecycle_term(value: str) -> str | None:
    value_lower = value.lower()
    for term in sorted(LIFECYCLE_TERMS):
        if term.lower() in value_lower:
            return term
    return None


def _qualified_expr_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_expr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return _qualified_expr_name(node.value)
    if isinstance(node, ast.Call):
        return _qualified_expr_name(node.func)
    return None


def _simple_name(value: str) -> str:
    return value.rsplit(".", maxsplit=1)[-1]


def _iter_python_files(paths: Sequence[Path]) -> Iterator[Path]:
    scan_paths = paths or (DEFAULT_SCAN_ROOT,)
    for path in scan_paths:
        if path.is_file() and path.suffix == ".py":
            yield path
        elif path.is_dir():
            yield from sorted(
                candidate
                for candidate in path.rglob("*.py")
                if "__pycache__" not in candidate.parts
            )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Block Plugin* classes from owning daemon, service, worker, runtime, "
            "consumer, publisher, server, controller, or client lifecycles."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[DEFAULT_SCAN_ROOT],
        help="Python file or directory paths to scan.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    findings = validate_paths(args.paths)
    if not findings:
        return 0

    sys.stderr.write("Plugin lifecycle class guard failed:\n")
    for finding in findings:
        sys.stderr.write(f"  {finding.format()}\n")
    sys.stderr.write(
        "\nPlugin* classes may only represent deterministic compute plugins. "
        "Put daemon, service, worker, runtime, publisher, consumer, client, "
        "controller, or server lifecycles behind ONEX nodes/handlers instead.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
