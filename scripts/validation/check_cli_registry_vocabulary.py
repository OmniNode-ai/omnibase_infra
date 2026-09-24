# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Refuse a hand-written vocabulary in an ``onex`` CLI module (OMN-19407).

Operator ruling, 2026-09-24: the CLI must fetch everything from the registry
and keep no bespoke list. Before this gate, ``onex delegate`` carried
``TASK_TYPE_CHOICES`` (11 classes against a 15-class contract),
``DELEGATE_SOURCE_CHOICES``, a literal ``--criteria-mode`` choice and a copied
slug pattern, each "pinned" by a test holding a third copy. Every one had
drifted or was one widening away from drifting.

WHAT IS REFUSED, in any ``.py`` file under the scanned roots:

1. ``click.Choice(...)`` given a literal list, tuple or set of strings. A
   closed flag vocabulary comes from the registry
   (``contract_registry.ContractChoice``) or from an enum the CLI's own
   package defines, never from a list typed at the call site.
2. A module- or class-level name ending in ``_CHOICES`` bound to a literal
   collection of two or more strings.
3. ``Literal[...]`` of two or more strings. A closed vocabulary a CLI module
   declares for data it sends or receives belongs to the contract that owns
   that data.

THE ONE EXEMPTION is a vocabulary the CLI itself owns (an output format of
this command, a refusal reason this module invents): mark it on the flagged
line, or the line above, with ``# cli-own-vocabulary: <why the CLI owns it>``.
The reason is mandatory. Saying "it is copied from X" is not a reason: that is
the finding.

Exit 0 when clean, 1 with one ``path:line: message`` per finding otherwise.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

EXEMPTION_MARKER = "# cli-own-vocabulary:"
DEFAULT_ROOTS = ("src/omnibase_infra/cli",)


def _string_elements(node: ast.AST) -> list[str] | None:
    """Return the strings of a literal collection, or ``None`` if it is not one."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "tuple", "set", "list"}
        and len(node.args) == 1
    ):
        node = node.args[0]
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    values = [
        element.value
        for element in node.elts
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    ]
    if len(values) != len(node.elts):
        return None
    return values


def _callee_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _subscript_name(node: ast.Subscript) -> str | None:
    value = node.value
    if isinstance(value, ast.Attribute):
        return value.attr
    if isinstance(value, ast.Name):
        return value.id
    return None


def _target_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [target.id for target in targets if isinstance(target, ast.Name)]


def _findings(tree: ast.Module) -> Iterator[tuple[int, str]]:
    scopes: list[ast.AST] = [tree]
    scopes.extend(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    for scope in scopes:
        for statement in getattr(scope, "body", ()):
            if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
                continue
            if statement.value is None:
                continue
            names = [n for n in _target_names(statement) if n.endswith("_CHOICES")]
            values = _string_elements(statement.value)
            if names and values is not None and len(values) >= 2:
                yield (
                    statement.lineno,
                    f"{names[0]} is a hand-written vocabulary {values[:4]}; read "
                    "it from the registry (contract_registry) instead",
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _callee_name(node) == "Choice" and node.args:
            values = _string_elements(node.args[0])
            if values is not None:
                yield (
                    node.lineno,
                    f"click.Choice over a literal list {values[:4]}; use "
                    "contract_registry.ContractChoice (or an enum this package owns)",
                )
        elif isinstance(node, ast.Subscript) and _subscript_name(node) == "Literal":
            members = (
                node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            )
            strings = [
                m.value
                for m in members
                if isinstance(m, ast.Constant) and isinstance(m.value, str)
            ]
            if len(strings) >= 2:
                yield (
                    node.lineno,
                    f"Literal{strings[:4]} declares a closed vocabulary in a CLI "
                    "module; read it off the contract's model through the registry",
                )


def _exempt(lines: Sequence[str], lineno: int) -> bool:
    for index in (lineno - 1, lineno - 2):
        if 0 <= index < len(lines):
            marker_at = lines[index].find(EXEMPTION_MARKER)
            if marker_at != -1:
                reason = lines[index][marker_at + len(EXEMPTION_MARKER) :].strip()
                return bool(reason)
    return False


def check_file(path: Path) -> list[str]:
    """Return the findings for one file, formatted ``path:line: message``."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))
    return [
        f"{path}:{lineno}: {message}"
        for lineno, message in sorted(set(_findings(tree)))
        if not _exempt(lines, lineno)
    ]


def _python_files(paths: Sequence[str]) -> Iterator[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        elif path.suffix == ".py" and path.is_file():
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "paths",
        nargs="*",
        help="Files or directories to scan (default: the onex CLI package)",
    )
    args = parser.parse_args(argv)
    targets = args.paths or list(DEFAULT_ROOTS)
    findings = [
        finding for path in _python_files(targets) for finding in check_file(path)
    ]
    for finding in findings:
        print(finding)
    if findings:
        print(
            f"\n{len(findings)} hand-written CLI vocabulary finding(s). The onex CLI "
            "reads every vocabulary from the registry (OMN-19407). If the CLI "
            f"itself owns the values, mark the line '{EXEMPTION_MARKER} <reason>'.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
