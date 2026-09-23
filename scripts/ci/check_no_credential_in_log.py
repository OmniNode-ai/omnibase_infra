#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""CI gate: reject log statements that format a credential value (OMN-17423).

WHY A GATE AND NOT ONLY A FILTER. ``CredentialRedactionFilter`` scrubs records
on their way to a handler, but it is defense in depth: it matches field names
and value shapes, so a credential concatenated into prose or carried under an
innocuous key walks straight through it. This gate is the other layer -- it
refuses the source pattern before a log handler ever sees it. Rule 5 of the
ticket: detection without enforcement gets ignored.

WHY THE VOCABULARY IS NARROWER HERE THAN IN THE FILTER. Field names are
structured; prose is not. ``{"token": v}`` is a credential-bearing field, but
"refresh token expired" is a sentence, and a gate that flags the latter gets
switched off within a week. So:

  * a COMPOUND fragment (``access_token``, ``api_key``, ``client_secret``...)
    matches as a substring anywhere -- ``linear_api_key`` is a credential;
  * a prose-ambiguous BARE word (``token``, ``secret``, ``password``...) must
    END the name. ``gateway_token`` is a credential; ``token_savings_pct``,
    ``total_direct_tokens`` and ``secrets_seeded`` are metrics and stay
    loggable. In a format string a bare word must ADDITIONALLY be assigned an
    interpolated value, because "refresh token expired" is a sentence.

Both lists are derived from the shared vocabulary rather than retyped, so a
fragment added for the runtime filter is picked up here automatically.

An ALL-CAPS token in a format string is an environment variable NAME, not a
value -- "LINEAR_API_KEY is not set" leaks nothing. Those are suppressed
UNLESS the name is immediately followed by an assignment to an interpolated
value (``API_KEY=%s``), which is a leak regardless of case.

ESCAPE HATCH. ``# credential-log-allow: <reason>`` anywhere inside a logger
call suppresses that call. The reason is required and shows up in the diff, so
a suppression is reviewable rather than silent.

Usage:
  python scripts/ci/check_no_credential_in_log.py --root src/omnibase_infra

  python scripts/ci/check_no_credential_in_log.py --mode=self-test \\
    --fixture scripts/ci/tests/fixtures/credential_in_log_fixture.py \\
    --expected 9
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import re
import sys
from pathlib import Path
from typing import Final, NamedTuple

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_VOCABULARY_PATH: Final[Path] = (
    _REPO_ROOT / "src" / "omnibase_infra" / "utils" / "util_dlq_credential_redaction.py"
)


def _load_shared_vocabulary() -> object:
    """Load the credential-name vocabulary by path, not by package import.

    A CI gate must run before -- and independently of -- a working install of
    the package it scans. ``import omnibase_infra`` pulls the whole runtime
    package graph; a syntax error anywhere in it would turn this gate into a
    crash instead of a verdict. Loading the one module by path keeps the gate's
    only dependency the file it actually reads, while still leaving exactly one
    definition of "what counts as a credential field name" in the repo.
    """
    spec = importlib.util.spec_from_file_location(
        "_omn17423_credential_vocabulary", _VOCABULARY_PATH
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot load credential vocabulary from {_VOCABULARY_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VOCAB = _load_shared_vocabulary()

_FRAGMENTS: Final[tuple[str, ...]] = _VOCAB.CREDENTIAL_FIELD_NAME_FRAGMENTS  # type: ignore[attr-defined]
_NON_CREDENTIAL_NAMES: Final[frozenset[str]] = _VOCAB.NON_CREDENTIAL_FIELD_NAMES  # type: ignore[attr-defined]
_NON_CREDENTIAL_SUFFIXES: Final[tuple[str, ...]] = _VOCAB.NON_CREDENTIAL_FIELD_SUFFIXES  # type: ignore[attr-defined]

# Bare words that are credential-bearing as a FIELD NAME but ordinary English
# in a sentence. Excluded from format-string matching only.
_PROSE_AMBIGUOUS: Final[frozenset[str]] = frozenset(
    {
        "secret",
        "token",
        "password",
        "passwd",
        "passphrase",
        "authorization",
        "credential",
        "credentials",
    }
)

# Derived, never hand-maintained: a fragment added to the shared tuple is
# picked up here automatically unless it is explicitly prose-ambiguous.
_FORMAT_STRING_FRAGMENTS: Final[tuple[str, ...]] = tuple(
    fragment for fragment in _FRAGMENTS if fragment not in _PROSE_AMBIGUOUS
)

_LOG_METHODS: Final[frozenset[str]] = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)

_NORMALISE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")
_ENV_VAR_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
# ``name=%s`` / ``name: {value}`` -- an assignment to something interpolated.
_ASSIGNED_VALUE_RE: Final[re.Pattern[str]] = re.compile(r"""^\s*[=:]\s*['"]?[%{]""")

_ALLOW_RE: Final[re.Pattern[str]] = re.compile(
    r"#\s*credential-log-allow:\s*(?P<reason>\S.*)$"
)

_FIXTURE_DIR_NAMES: Final[frozenset[str]] = frozenset({"fixtures", "lint-fixtures"})


class Finding(NamedTuple):
    lineno: int
    entry: str
    detail: str


def _normalise(name: str) -> str:
    return _NORMALISE_RE.sub("", name.lower())


def _is_exempt(normalised: str) -> bool:
    """Shared reference exemptions: ``api_key_id``, ``token_count``, ``secret_ref``."""
    if normalised in _NON_CREDENTIAL_NAMES:
        return True
    return any(
        normalised.endswith(suffix) and normalised != suffix
        for suffix in _NON_CREDENTIAL_SUFFIXES
    )


def _bare_word_suffix(normalised: str) -> str | None:
    """Return the bare credential word a name ENDS with, or None.

    Position carries the meaning. ``gateway_token`` and ``broker_secret`` name
    the thing itself; ``token_savings_pct`` and ``secrets_seeded`` name a
    measurement about it. Matching anywhere in the name conflates the two --
    that is how a gate ends up flagging seven metrics and getting disabled.
    """
    for word in _PROSE_AMBIGUOUS:
        if normalised == word or normalised.endswith(word):
            return word
    return None


def _credential_hit(name: str) -> str | None:
    """Return the fragment that makes ``name`` credential-bearing, or None.

    A compound fragment may match as a SUBSTRING -- ``linear_api_key`` and
    ``tenant_access_token`` are both credentials. A prose-ambiguous bare word
    must match the WHOLE normalised name: ``token`` is a credential,
    ``token_savings_pct`` and ``secrets_seeded`` are metrics. The shared
    reference exemptions are consulted first, so ``api_key_id`` and
    ``token_count`` never reach either rule.
    """
    normalised = _normalise(name)
    if not normalised or _is_exempt(normalised):
        return None
    for fragment in _FORMAT_STRING_FRAGMENTS:
        if fragment in normalised:
            return fragment
    return _bare_word_suffix(normalised)


def _format_string_hit(text: str) -> str | None:
    """Return the fragment a format string leaks, or None.

    Scans identifier-shaped tokens rather than doing a raw substring search, so
    the environment-variable and assignment rules have a token to reason about.
    """
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        normalised = _normalise(token)
        if not normalised or _is_exempt(normalised):
            continue

        trailing = text[match.end() : match.end() + 8]
        assigned = bool(_ASSIGNED_VALUE_RE.match(trailing))

        compound = next((f for f in _FORMAT_STRING_FRAGMENTS if f in normalised), None)
        if compound is not None:
            # An ALL-CAPS name in a status message is an environment variable
            # NAME ("LINEAR_API_KEY is not set"), which leaks nothing --
            # unless it is being assigned an interpolated value.
            if _ENV_VAR_RE.fullmatch(token) and not assigned:
                continue
            return compound

        # A bare word in prose is prose. Only an assignment makes it a leak.
        bare = _bare_word_suffix(normalised)
        if bare is not None and assigned:
            return bare
    return None


def _expr_credential_name(node: ast.expr) -> str | None:
    """Return the credential name a variable or attribute reference carries."""
    if isinstance(node, ast.Name):
        return node.id if _credential_hit(node.id) else None
    if isinstance(node, ast.Attribute):
        return node.attr if _credential_hit(node.attr) else None
    return None


def _dict_key_findings(node: ast.Dict, label: str) -> list[Finding]:
    findings: list[Finding] = []
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            if _credential_hit(key.value):
                findings.append(
                    Finding(node.lineno, key.value, f"{label} {key.value!r}")
                )
    return findings


def _check_call(node: ast.Call) -> list[Finding]:
    findings: list[Finding] = []

    if node.args:
        first = node.args[0]

        # (A) literal format string: logger.info("api_key=%s", key)
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            hit = _format_string_hit(first.value)
            if hit:
                snippet = first.value[:80].replace("\n", " ")
                findings.append(
                    Finding(first.lineno, hit, f'format string "{snippet}"')
                )

        # (B) f-string: logger.info(f"key={access_token}")
        elif isinstance(first, ast.JoinedStr):
            for child in ast.walk(first):
                if isinstance(child, ast.FormattedValue):
                    name = _expr_credential_name(child.value)
                    if name:
                        findings.append(
                            Finding(
                                first.lineno, name, f"f-string interpolates {name!r}"
                            )
                        )
                elif isinstance(child, ast.Constant) and isinstance(child.value, str):
                    hit = _format_string_hit(child.value)
                    if hit:
                        findings.append(
                            Finding(
                                first.lineno, hit, f"f-string literal names {hit!r}"
                            )
                        )

        # (C)/(D) positional args after the format string
        for arg in node.args[1:]:
            name = _expr_credential_name(arg)
            if name:
                findings.append(Finding(arg.lineno, name, f"positional arg {name!r}"))
            elif isinstance(arg, ast.Dict):
                findings.extend(_dict_key_findings(arg, "dict literal key"))

    # (E) extra={"api_key": ...}
    for keyword in node.keywords:
        if keyword.arg == "extra" and isinstance(keyword.value, ast.Dict):
            findings.extend(_dict_key_findings(keyword.value, "extra= key"))

    return findings


def _check_file(path: Path) -> list[Finding]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []

    lines = source.splitlines()
    allowed: set[int] = {
        index + 1 for index, line in enumerate(lines) if _ALLOW_RE.search(line)
    }

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in _LOG_METHODS:
            continue
        # A suppression is honoured anywhere inside the logger call it
        # annotates. Anchoring it to the finding's exact line looks tighter but
        # is not stable: `ruff format` reflows a call and moves a trailing
        # comment onto the closing paren, which would silently un-suppress a
        # reviewed line at the next format pass.
        span = range(node.lineno, (node.end_lineno or node.lineno) + 1)
        if any(line in allowed for line in span):
            continue
        findings.extend(_check_call(node))
    return findings


def _collect_py_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if not any(part in _FIXTURE_DIR_NAMES for part in path.parts)
    )


def run_normal(root: Path) -> int:
    if not root.is_dir():
        print(f"ERROR: scan root not found: {root}")
        return 1
    files = _collect_py_files(root)
    total = 0
    for path in files:
        for finding in _check_file(path):
            print(
                f"VIOLATION {path}:{finding.lineno}: "
                f"credential field {finding.entry!r} in a logger call -- {finding.detail}"
            )
            total += 1
    if total:
        print(
            f"\n{total} credential-in-log violation(s) in {root}. Log an "
            "identifier (``api_key_id``), a fingerprint, or a length -- never "
            "the value. If this is a false positive, annotate the line with "
            "`# credential-log-allow: <reason>`."
        )
        return 1
    print(f"OK: scanned {len(files)} file(s) under {root}, zero violations.")
    return 0


def run_self_test(fixture: Path, expected: int) -> int:
    """Prove the gate is non-vacuous against a file of injected violations."""
    if not fixture.exists():
        print(f"ERROR: fixture file not found: {fixture}")
        return 1
    findings = _check_file(fixture)
    if len(findings) != expected:
        print(
            f"SELF-TEST FAIL: expected {expected} violation(s) in the fixture, "
            f"found {len(findings)}. Fixture and expected count must stay in sync."
        )
        for finding in findings:
            print(f"  line {finding.lineno}: {finding.entry!r} -- {finding.detail}")
        return 1
    print(f"SELF-TEST OK: fixture yields exactly {expected} violation(s).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("src/omnibase_infra"))
    parser.add_argument("--mode", choices=["normal", "self-test"], default="normal")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("scripts/ci/tests/fixtures/credential_in_log_fixture.py"),
    )
    parser.add_argument("--expected", type=int, default=9)
    args = parser.parse_args()

    if args.mode == "self-test":
        sys.exit(run_self_test(args.fixture, args.expected))
    sys.exit(run_normal(args.root))


if __name__ == "__main__":
    main()
