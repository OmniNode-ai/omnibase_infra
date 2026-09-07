# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A test may not assert an event_type shape the bus never carries (OMN-18013).

OPERATOR RULING (2026-09-06), ITEM 4
------------------------------------
Golden chains build their input from the publisher contract, so a test cannot
pass on a shape the bus never carries.

THE DEFECT
----------
The runtime stamps ``event_type`` with the ALIAS
``derive_event_type_alias_for_topic(topic)`` -> ``<producer>.<event-name>``, the
single source for that alias on both sides of the wire (OMN-17296). It never
stamps the full topic ``onex.<kind>.<producer>.<event-name>.v<n>``. Yet 108 test
sites across 44 files passed the TOPIC as an envelope ``event_type``. They were
green only because ``derive_entry_message_types`` registers BOTH spellings as
dispatcher index keys, so the wrong one still resolved — inside the test.
Production used the other one. A green suite over the wrong spelling is worse
than no suite: it is evidence pointing the wrong way.

WHAT IS REFUSED
---------------
An ``event_type`` whose value is a literal ONEX TOPIC, in either of the two
places where it asserts a wire shape:

* a keyword argument to an envelope constructor (``ModelEventEnvelope(...)``),
* any ``event_type`` key or keyword anywhere inside a ``test_golden_chain_*.py``
  module, whose entire job is to reproduce a real chain.

Anything else — a ledger ROW whose ``event_type`` column genuinely stores a
topic, a projection fixture, a TUI feed row — is untouched, because those are
storage shapes, not wire shapes.

THE FIX
-------
``omnibase_infra.testing.publisher_contract_fixture.PublisherContractCorpus``.
It resolves the topic against the contract corpus, REFUSES a topic no contract
publishes, and stamps the derived alias — so the envelope is byte-for-byte what
the consume boundary produces and a chain over an orphan topic cannot be written
at all.

Usage (pre-commit / CI):
    uv run python -m omnibase_infra.validators.no_literal_event_type_in_tests tests
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_LITERAL_TOPIC = re.compile(r"^onex\.(?:evt|cmd|intent|dlq|snapshot)\..+$")
_ALLOW = re.compile(r"#\s*onex-topic-allow:\s*\S")
_ENVELOPE_CTORS = ("ModelEventEnvelope",)

# A tests tree that suddenly contains almost no test modules is a broken scan, not
# a clean tree. Fail closed rather than reporting a vacuous pass.
DEFAULT_MIN_TEST_FILES = 200


@dataclass(frozen=True, slots=True)  # internal-dataclass-ok: validator-internal finding
class LiteralEventTypeFinding:
    """One test site feeding a topic string where the bus carries an alias."""

    location: str
    detail: str


def _is_envelope_ctor(node: ast.Call) -> bool:
    """Whether this call constructs an event envelope (possibly subscripted)."""
    func = node.func
    if isinstance(func, ast.Subscript):
        func = func.value
    name = (
        func.attr
        if isinstance(func, ast.Attribute)
        else func.id
        if isinstance(func, ast.Name)
        else ""
    )
    return name in _ENVELOPE_CTORS


def _scope_topic_names(body: list[ast.stmt]) -> dict[str, str]:
    """Names bound to a ONEX topic literal within ONE scope.

    Assignments inside nested function scopes are skipped — they belong to that
    scope's own map. A name whose bindings in this scope are not ALL topic
    literals is dropped, so a name reused for something else cannot produce a
    false positive.
    """
    bound: dict[str, str] = {}
    rejected: set[str] = set()

    def visit(stmt: ast.stmt) -> None:
        for node in ast.walk(stmt):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue
            targets: list[ast.expr] = []
            value: ast.expr | None = None
            if isinstance(node, ast.Assign):
                targets, value = list(node.targets), node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and _LITERAL_TOPIC.match(value.value)
                ):
                    bound.setdefault(target.id, value.value)
                else:
                    rejected.add(target.id)

    for stmt in body:
        if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        visit(stmt)
    return {name: v for name, v in bound.items() if name not in rejected}


def _topic_name_scopes(tree: ast.Module) -> list[tuple[ast.AST, dict[str, str]]]:
    """(scope node, name -> topic) for the module and every function in it.

    An ``event_type=`` argument does not have to be written inline to be a topic:
    ``validated = "onex.evt...."`` followed by ``event_type=validated`` is the
    same defect one indirection away, and a lint reading only ``ast.Constant`` is
    evaded by a local variable — which is exactly how
    ``test_real_dispatch_multitopic_routing.py`` kept feeding a topic as an
    event_type while this gate reported the tree clean (OMN-18013). Resolution is
    per SCOPE so one function's ``topic = some_call()`` cannot mask another
    function's ``topic = "onex...."``.
    """
    scopes: list[tuple[ast.AST, dict[str, str]]] = [
        (tree, _scope_topic_names(list(tree.body)))
    ]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            scopes.append((node, _scope_topic_names(list(node.body))))
    return scopes


def findings_for_module(path: Path, text: str) -> list[LiteralEventTypeFinding]:
    """Every refused ``event_type`` literal in one test module."""
    golden_chain = path.name.startswith("test_golden_chain_")
    allowed = {
        n for n, line in enumerate(text.splitlines(), start=1) if _ALLOW.search(line)
    }
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return [
            LiteralEventTypeFinding(
                location=f"{path}:{exc.lineno or 0}",
                detail=f"module does not parse, so it cannot be checked: {exc.msg}",
            )
        ]

    out: list[LiteralEventTypeFinding] = []

    def record(lineno: int, value: str) -> None:
        if lineno in allowed or not _LITERAL_TOPIC.match(value):
            return
        out.append(
            LiteralEventTypeFinding(
                location=f"{path}:{lineno}",
                detail=(
                    f"event_type={value!r} is a TOPIC. The bus carries the alias "
                    f"for this topic, not the topic. Build the envelope with "
                    "PublisherContractCorpus.publisher_envelope(topic, payload=...)."
                ),
            )
        )

    module_names = _scope_topic_names(list(tree.body))
    # A site inside a function is reached twice — once walking the module scope,
    # once walking the function's own — so findings are deduped by location.
    seen: set[str] = set()

    def check_call(node: ast.Call, names: dict[str, str]) -> None:
        for kw in node.keywords:
            if kw.arg != "event_type":
                continue
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                record(kw.value.lineno, kw.value.value)
            elif isinstance(kw.value, ast.Name):
                resolved = names.get(kw.value.id)
                if resolved is not None:
                    record(kw.value.lineno, resolved)

    def check_dict(node: ast.Dict, names: dict[str, str]) -> None:
        for key, value in zip(node.keys, node.values, strict=True):
            if not (isinstance(key, ast.Constant) and key.value == "event_type"):
                continue
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                record(value.lineno, value.value)
            elif isinstance(value, ast.Name):
                resolved = names.get(value.id)
                if resolved is not None:
                    record(value.lineno, resolved)

    for scope, local_names in _topic_name_scopes(tree):
        names = {**module_names, **local_names}
        for node in ast.walk(scope):
            if isinstance(node, ast.Call) and (golden_chain or _is_envelope_ctor(node)):
                check_call(node, names)
            elif isinstance(node, ast.Dict) and golden_chain:
                check_dict(node, names)

    deduped: list[LiteralEventTypeFinding] = []
    for finding in out:
        if finding.location in seen:
            continue
        seen.add(finding.location)
        deduped.append(finding)
    return deduped


def scan(tests_root: Path) -> tuple[list[LiteralEventTypeFinding], int]:
    """Return (findings, test_module_count) under ``tests_root``."""
    findings: list[LiteralEventTypeFinding] = []
    modules = sorted(
        p
        for p in tests_root.rglob("test_*.py")
        if ".venv" not in p.parts and "site-packages" not in p.parts
    )
    for path in modules:
        findings.extend(findings_for_module(path, path.read_text(encoding="utf-8")))
    return findings, len(modules)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refuse a test that feeds a literal ONEX topic as an envelope event_type "
            "— a shape the bus never carries (OMN-18013 item 4)."
        )
    )
    parser.add_argument("tests_root", nargs="?", default="tests")
    parser.add_argument("--min-test-files", type=int, default=DEFAULT_MIN_TEST_FILES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    tests_root = Path(args.tests_root)
    findings, module_count = scan(tests_root)

    if module_count < args.min_test_files:
        sys.stderr.write(
            f"[no-literal-event-type-in-tests] FAIL (vacuity guard): only "
            f"{module_count} test modules discovered under {tests_root} (expected >= "
            f"{args.min_test_files}). A lint over a collapsed set proves nothing.\n"
        )
        return 1

    if findings:
        sys.stderr.write(
            "[no-literal-event-type-in-tests] FAIL: test site(s) feeding a TOPIC "
            "string as an envelope event_type. The bus carries the derived alias "
            "(OMN-17296), so these assert a shape production never produces:\n"
        )
        for f in sorted(findings, key=lambda x: x.location):
            sys.stderr.write(f"  - {f.location}\n      {f.detail}\n")
        sys.stderr.write(
            "\n  Fix: build the input from the publisher's contract —\n"
            "    from omnibase_infra.testing.publisher_contract_fixture import "
            "PublisherContractCorpus\n"
            "    corpus = PublisherContractCorpus.from_repo_root(Path('src'))\n"
            "    envelope = corpus.publisher_envelope(TOPIC, payload=...)\n"
            "  It refuses a topic no contract publishes, so a chain cannot be written "
            "over an orphan topic at all. There is no baseline for this lint.\n"
        )
        return 1

    sys.stderr.write(
        f"[no-literal-event-type-in-tests] OK: {module_count} test modules scanned "
        f"under {tests_root}, no topic-as-event_type sites.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
