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
  * an ambiguous BARE word (``token``, ``secret``, ``credential``) must be the
    WHOLE name. ``token`` is a credential; ``masked_token``,
    ``token_savings_pct`` and ``secrets_seeded`` are markers and metrics, and
    stay loggable. Bare words apply to field NAMES only, never to format
    strings.

Both lists are derived from the shared vocabulary rather than retyped, so a
fragment added for the runtime filter is picked up here automatically.

A format string only leaks when a credential name is ASSIGNED an interpolated
value (``api_key=%s``). Merely naming one is a status message -- "LINEAR_API_KEY
is not set", "OIDC for api-keys router disabled: %s" -- where the interpolated
value is a reason, not the credential. Bare words are not consulted in format
strings at all; see ``_format_string_hit``.

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

# Words this codebase uses for BOTH a credential and an opaque marker. Measured,
# not guessed: matching them as a substring flags `masked_token` (a token that
# is already masked), `matched_token` (a merge-hold marker), `cursor.token` (a
# pagination cursor), `INFERENCE_TIMEOUT_LOG_TOKEN` (a log marker constant),
# `secrets_seeded` and `named_credential` -- six false positives across
# omnibase_infra, omnimarket and onex-api, and zero true positives. They are
# matched as a WHOLE name only, and never inside a format string.
#
# `password`, `passphrase` and `authorization` are deliberately NOT here: this
# platform never uses them for anything else, so they stay substring-matchable.
_AMBIGUOUS_BARE: Final[frozenset[str]] = frozenset(
    {"secret", "token", "credential", "credentials"}
)

# Derived, never hand-maintained: a fragment added to the shared vocabulary is
# picked up here automatically unless it is explicitly ambiguous.
_COMPOUND_FRAGMENTS: Final[tuple[str, ...]] = tuple(
    fragment for fragment in _FRAGMENTS if fragment not in _AMBIGUOUS_BARE
)

_LOG_METHODS: Final[frozenset[str]] = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}
)

_NORMALISE_RE: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")
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


def _bare_word_whole_name(normalised: str) -> str | None:
    """Return the ambiguous word a name IS, or None.

    Whole-name only. A first draft matched these as a suffix, which reads
    plausibly -- ``gateway_token`` names the thing itself -- but measured
    against real code it flagged six markers and no credentials. Genuine
    compound credentials are covered by the compound fragments instead, which
    is why ``gatewaytoken`` was added to the shared vocabulary rather than
    left to this rule.
    """
    return normalised if normalised in _AMBIGUOUS_BARE else None


def _credential_hit(name: str) -> str | None:
    """Return the fragment that makes ``name`` credential-bearing, or None.

    A compound fragment may match as a SUBSTRING -- ``linear_api_key``,
    ``tenant_access_token`` and ``admin_password`` are all credentials. An
    ambiguous bare word must be the WHOLE normalised name: ``token`` is a
    credential, ``masked_token`` and ``token_savings_pct`` are not. The shared
    reference exemptions are consulted first, so ``api_key_id`` and
    ``token_count`` never reach either rule.
    """
    normalised = _normalise(name)
    if not normalised or _is_exempt(normalised):
        return None
    for fragment in _COMPOUND_FRAGMENTS:
        if fragment in normalised:
            return fragment
    return _bare_word_whole_name(normalised)


def _format_string_hit(text: str) -> str | None:
    """Return the fragment a format string leaks, or None.

    A format string is PROSE with interpolation holes in it, and the only shape
    that actually leaks is a credential name immediately ASSIGNED to a hole:
    ``"api_key=%s"``. Everything else that merely mentions the word is a status
    message -- ``"LINEAR_API_KEY is not set"``, ``"OIDC for api-keys router
    disabled: %s"``, ``"failed to mint access_token for %s"`` -- where the
    interpolated value is a reason or a tenant id, not the credential. The last
    two are live lines in ``onex-api`` on ``dev``; requiring the assignment is
    what tells them apart from a leak.

    Prose-ambiguous bare words (``token``, ``secret``) are not consulted here at
    all. ``"Invalid tenant_id format in token: %s"`` -- also live in
    ``onex-api`` -- is indistinguishable from ``"token: %s"`` by trailing
    context, and the value interpolated there is the tenant id. The cost is
    that ``logger.info("token=%s", v)`` with a non-credential-named ``v`` is not
    caught by THIS rule; a credential-named argument still is (rule C), an
    f-string still is (rule B), and the runtime filter's shape pass still
    redacts the rendered value. Bare words stay in force for FIELD names, where
    they are unambiguous.
    """
    for match in _TOKEN_RE.finditer(text):
        normalised = _normalise(match.group(0))
        if not normalised or _is_exempt(normalised):
            continue
        if not _ASSIGNED_VALUE_RE.match(text[match.end() : match.end() + 8]):
            continue
        compound = next((f for f in _COMPOUND_FRAGMENTS if f in normalised), None)
        if compound is not None:
            return compound
    return None


def _joined_str_template(node: ast.JoinedStr) -> str:
    """Render an f-string with its interpolations as ``{}`` placeholders."""
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            parts.append("{}")
    return "".join(parts)


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
            # Rebuild the f-string as a template so the assignment rule can see
            # the hole. Scanning each literal chunk separately cannot: the chunk
            # before an interpolation ends at "client_secret=", with the thing
            # it is assigned sitting in the NEXT node.
            hit = _format_string_hit(_joined_str_template(first))
            if hit:
                findings.append(
                    Finding(first.lineno, hit, f"f-string literal names {hit!r}")
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
