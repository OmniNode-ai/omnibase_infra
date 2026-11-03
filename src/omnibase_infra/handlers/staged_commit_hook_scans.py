# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Staged-diff entry point of the ``topic-naming-lint`` commit hook (OMN-22323).

``topic-naming-lint`` read every node contract and every Python file under
``src/omnibase_infra`` on every commit, and put each file's consumer declarations
through the contract-validation node over an in-memory bus. Every finding of that lint
is a function of one file's content: a topic string is judged on its own, and the
consumer-declaration check is a request per file whose response is keyed by its
correlation id. So the hook judges the files the commit stages and reaches the verdict
the whole-tree scan reaches for them. The whole-tree scan is unchanged: it is what a call
with no file arguments does (``scripts/validation/run_topic_lint.sh``, CI and
``pre-commit run --all-files``).

    uv run --frozen python -m omnibase_infra.handlers.staged_commit_hook_scans topic-naming-lint [path ...]

The scan falls back to the whole tree when the commit stages a file the per-file verdict
depends on besides the file itself: the lint script, its baseline and denylist, the
contract-validation node, the topic constants, or the runtime that wires and dispatches
that node.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Final

_LINT_SCRIPT: Final[str] = "scripts/validation/lint_topic_names.py"
_NODES_ROOT: Final[str] = "src/omnibase_infra/nodes"
_PYTHON_ROOT: Final[str] = "src/omnibase_infra"

#: A staged file under one of these prefixes changes what every file's verdict depends on.
VALIDATOR_INPUT_PREFIXES: Final[tuple[str, ...]] = (
    "scripts/validation/",
    "src/omnibase_infra/nodes/node_contract_validate_compute/",
    "src/omnibase_infra/event_bus/",
    "src/omnibase_infra/runtime/",
    "src/omnibase_infra/protocols/",
    "uv.lock",
    "pyproject.toml",
)


def _load_lint(root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "lint_topic_names", root / _LINT_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {root / _LINT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _relative(root: Path, name: str) -> str:
    return (root / name).resolve().relative_to(root).as_posix()


def _contracts_among(names: Sequence[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name.startswith(f"{_NODES_ROOT}/")
        and name.rsplit("/", 1)[-1] == "contract.yaml"
    )


def _python_among(names: Sequence[str]) -> list[str]:
    return sorted(
        name
        for name in names
        if name.startswith(f"{_PYTHON_ROOT}/") and name.endswith(".py")
    )


def scan_named_contracts(lint: ModuleType, files: Sequence[Path]) -> list[str]:
    """``lint.scan_contracts`` over the named contract files: the same loop, one wiring."""
    all_violations: list[str] = []
    requests: dict[str, dict[str, object]] = {}
    for contract_path in files:
        try:
            with contract_path.open(encoding="utf-8") as fh:
                raw_yaml = lint.yaml.safe_load(fh)
        except (OSError, ValueError, lint.yaml.YAMLError) as exc:
            all_violations.append(f"Could not parse {contract_path}: {exc}")
            continue
        if not isinstance(raw_yaml, dict):
            continue
        raw_topics = lint._extract_raw_topics_from_contract(raw_yaml)
        event_bus = raw_yaml.get("event_bus")
        consumer_declaration = {
            "consumed_events": raw_yaml.get("consumed_events"),
            "event_bus": {
                "subscribe_topics": event_bus.get("subscribe_topics")
                if isinstance(event_bus, dict)
                else None,
            },
        }
        requests[str(contract_path)] = {
            "consumer_topics": tuple(
                lint._extract_raw_topics_from_contract(consumer_declaration)
            ),
        }
        for raw in raw_topics:
            result = lint.lint_topic(raw)
            if not result.is_valid:
                for violation in result.violations:
                    all_violations.append(f"{contract_path}: {violation}")
    all_violations.extend(asyncio.run(lint._validate_consumer_declarations(requests)))
    return all_violations


def scan_named_python(lint: ModuleType, files: Sequence[Path]) -> list[str]:
    """``lint.scan_python`` over the named Python files: the same loop, one wiring."""
    all_violations: list[str] = []
    requests: dict[str, dict[str, object]] = {}
    for py_path in files:
        try:
            source = py_path.read_text(encoding="utf-8")
        except OSError as exc:
            all_violations.append(
                f"{py_path}: could not read consumer declarations: {exc}"
            )
            continue
        requests[str(py_path)] = {"consumer_source": source}
        for raw, lineno in lint._extract_topics_from_python_file(py_path):
            result = lint.lint_topic(raw)
            if not result.is_valid:
                for violation in result.violations:
                    all_violations.append(f"{py_path}:{lineno}: {violation}")
    all_violations.extend(asyncio.run(lint._validate_consumer_declarations(requests)))
    return all_violations


def topic_naming_lint(root: Path, paths: Sequence[str]) -> int:
    """Run both lint scans, over the staged files when they alone decide the verdict."""
    lint = _load_lint(root)
    contracts_root = root / _NODES_ROOT
    python_root = root / _PYTHON_ROOT
    names = [_relative(root, name) for name in paths]
    whole_tree = not names or any(
        name.startswith(VALIDATOR_INPUT_PREFIXES) for name in names
    )
    if not whole_tree:
        contracts = [root / name for name in _contracts_among(names)]
        python = [root / name for name in _python_among(names)]
        vars(lint)["scan_contracts"] = lambda _root: scan_named_contracts(
            lint, [p for p in contracts if p.is_file()]
        )
        vars(lint)["scan_python"] = lambda _root: scan_named_python(
            lint, [p for p in python if p.is_file()]
        )
    # The shell wrapper runs both scans and exits with the last non-zero code.
    contracts_code = int(lint.main(["--scan-contracts", str(contracts_root)]))
    python_code = int(lint.main(["--scan-python", str(python_root)]))
    return python_code or contracts_code


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] != "topic-naming-lint":
        sys.stderr.write(
            "usage: staged_commit_hook_scans topic-naming-lint [path ...]\n"
        )
        return 64
    return topic_naming_lint(Path.cwd().resolve(), args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
