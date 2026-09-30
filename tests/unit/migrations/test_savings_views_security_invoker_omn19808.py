# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Savings view replacements must leave invoker rights on after the corpus."""

import re
from collections.abc import Sequence
from pathlib import Path

import pytest
import sqlparse
from sqlparse import tokens

pytestmark = pytest.mark.unit

FORWARD = Path(__file__).resolve().parents[3] / "docker" / "migrations" / "forward"
VIEWS = ("projection_delegation_savings", "projection_delegation_savings_series")
_RELATION = (
    r"(?:public\s*\.\s*)?(?P<view>projection_delegation_savings(?:_series)?)(?!\w)"
)
_CREATE = re.compile(
    rf"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+{_RELATION}(?P<header>.*?)\bAS\b",
    re.IGNORECASE | re.DOTALL,
)
_ALTER = re.compile(
    rf"^\s*ALTER\s+VIEW\s+(?:IF\s+EXISTS\s+)?{_RELATION}\s+"
    r"(?P<action>SET|RESET)\s*\((?P<options>[^)]*)\)",
    re.IGNORECASE,
)
_WITH = re.compile(r"\bWITH\s*\(([^)]*)\)", re.IGNORECASE)
_INVOKER = re.compile(r"\bsecurity_invoker\s*=\s*(true|false)\b", re.IGNORECASE)


def _assert_invoker_rights(corpus: Sequence[tuple[str, str]]) -> None:
    """Track CREATE and ALTER statements in application order, per public view.

    This is a static contract for literal DDL, not an executor for dynamic SQL.
    Comments and string literals cannot serve as evidence of restored rights.
    """
    protected = dict.fromkeys(VIEWS, False)
    last_change: dict[str, str] = {}
    for filename, sql in corpus:
        if not any(view in sql.lower() for view in VIEWS):
            continue
        for statement in sqlparse.parse(sql):
            executable = "".join(
                " "
                if token.ttype in tokens.Comment
                or token.ttype in tokens.Literal.String.Single
                or token.ttype == tokens.Literal
                else token.value.replace('"', "")
                for token in statement.flatten()
            )
            if created := _CREATE.match(executable):
                view = created["view"].lower()
                options = _WITH.search(created["header"])
                invoker = _INVOKER.search(options[1]) if options else None
                protected[view] = invoker is not None and invoker[1].lower() == "true"
                last_change[view] = filename
            elif altered := _ALTER.match(executable):
                view = altered["view"].lower()
                options = altered["options"]
                if altered["action"].lower() == "reset":
                    if re.search(r"\bsecurity_invoker\b", options, re.IGNORECASE):
                        protected[view] = False
                        last_change[view] = filename
                elif invoker := _INVOKER.search(options):
                    protected[view] = invoker[1].lower() == "true"
                    last_change[view] = filename
    failures = [
        f"{view} ({last_change.get(view, 'no DDL found')})"
        for view, secure in protected.items()
        if not secure
    ]
    assert not failures, (
        "security_invoker must be restored after the last CREATE: "
        + ", ".join(failures)
    )


def _initial_corpus() -> list[tuple[str, str]]:
    return [
        (
            "001_create.sql",
            "\n".join(
                f"CREATE VIEW public.{view} WITH (security_invoker = true) AS SELECT 1;"
                for view in VIEWS
            ),
        )
    ]


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "-- ALTER VIEW public.{view} SET (security_invoker = true);",
        "/* ALTER VIEW public.{view} SET (security_invoker = true); */",
        "SELECT 'ALTER VIEW public.{view} SET (security_invoker = true);';",
        "ALTER VIEW other_schema.{view} SET (security_invoker = true);",
        "ALTER VIEW public.{view}_decoy SET (security_invoker = true);",
        "ALTER VIEW public.{view} SET (security_invoker = false);",
    ],
)
def test_bare_replacement_fails(view: str, suffix: str) -> None:
    corpus = _initial_corpus() + [
        (
            "002_replace.sql",
            f"CREATE OR REPLACE VIEW public.{view} AS SELECT 1;\n"
            + suffix.format(view=view),
        )
    ]
    with pytest.raises(AssertionError, match=view):
        _assert_invoker_rights(corpus)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("same_file", [True, False])
def test_replacement_then_alter_passes(view: str, same_file: bool) -> None:
    replacement = f"CREATE OR REPLACE VIEW public.{view} AS SELECT 1;"
    repair = f"ALTER VIEW IF EXISTS public.{view} SET (security_invoker = true);"
    corpus = _initial_corpus()
    if same_file:
        corpus.append(("002_replace.sql", replacement + repair))
    else:
        corpus.extend([("002_replace.sql", replacement), ("003_repair.sql", repair)])
    _assert_invoker_rights(corpus)


@pytest.mark.parametrize("view", VIEWS)
def test_replacement_with_invoker_passes(view: str) -> None:
    _assert_invoker_rights(
        _initial_corpus()
        + [
            (
                "002_replace.sql",
                f'create or replace view "public"."{view}" '
                "with (security_barrier = true, security_invoker = true) as select 1;",
            )
        ]
    )


@pytest.mark.parametrize("view", VIEWS)
def test_alter_before_replacement_fails(view: str) -> None:
    _assert_corpus = _initial_corpus() + [
        (
            "002_replace.sql",
            f"ALTER VIEW public.{view} SET (security_invoker = true);"
            f"CREATE OR REPLACE VIEW public.{view} AS SELECT 1;",
        )
    ]
    with pytest.raises(AssertionError, match=view):
        _assert_invoker_rights(_assert_corpus)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize(
    "action", ["RESET (security_invoker)", "SET (security_invoker = false)"]
)
def test_later_revocation_fails(view: str, action: str) -> None:
    with pytest.raises(AssertionError, match=view):
        _assert_invoker_rights(
            _initial_corpus()
            + [("002_revoke.sql", f"ALTER VIEW public.{view} {action};")]
        )


def test_real_forward_corpus_preserves_invoker_rights() -> None:
    # The runner applies the flat stream, then nodes/ in lexical node/file order.
    paths = sorted(FORWARD.glob("*.sql")) + sorted(FORWARD.glob("nodes/*/*.sql"))
    assert paths, "migration discovery must not pass on an empty corpus"
    _assert_invoker_rights(
        [
            (str(path.relative_to(FORWARD)), path.read_text(encoding="utf-8"))
            for path in paths
        ]
    )


def test_real_corpus_without_forward_repair_fails() -> None:
    paths = sorted((FORWARD / "nodes" / "node_projection_savings").glob("*.sql"))
    with pytest.raises(AssertionError, match="projection_delegation_savings"):
        _assert_invoker_rights(
            [
                (path.name, path.read_text(encoding="utf-8"))
                for path in paths
                if path.name < "092_"
            ]
        )
