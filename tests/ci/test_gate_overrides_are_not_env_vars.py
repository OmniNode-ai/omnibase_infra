# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Ratchet: no local gate may honor an inheritable env-var override (OMN-16480).

This is the class-closure half of the fix, and it is the half that matters in
six months. Rewriting one override to use a scoped token fixes one instance;
nothing stops the next guard author from adding
``PREPUSH_ALLOW_SOMETHING_ELSE`` and reproducing the incident exactly.

The incident, stated once so the rule is not mistaken for hygiene: on
2026-08-23 an operator used the gate's own documented escape hatch correctly.
Because that hatch was a plain environment variable, it was inherited by a
guard test's ``env=dict(os.environ)`` subprocess copy; the hook took its
override branch and launched another full 44,064-test suite, which reached the
same test and recursed. ~9h03m -- about 72% of all serialized suite wall-clock
in that window -- with zero ``[skip-*`` tokens and zero ``--no-verify`` in the
ledger. Compliance was perfect. The mechanism was wrong.

An override that widens what the gate accepts must therefore be:
  * consumed (single-use), so a descendant cannot replay it,
  * scoped to a repo and a commit, so it does not authorize other work,
  * time-bounded, so it stops being permission, and
  * receipted, so it is never invisible.

An environment variable is none of those things, so this gate refuses the shape
outright rather than trying to police its use. Compare Rule 10's ``[skip-*``
hardening (OMN-9731 / OMN-13388): the same conclusion, one layer up.

Static half + behavioral half, because a scan alone would pass on a hook whose
rejection branch had been quietly made unreachable.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOKS_DIR = REPO_ROOT / "scripts" / "hooks"
HOOK_SCRIPT = HOOKS_DIR / "prepush_smart_tests.sh"
PYTEST_GUARD = HOOKS_DIR / "pytest_full_suite_host_guard.py"
GRANT_MODULE = HOOKS_DIR / "prepush_override_grant.py"

#: Variable names in the permission-widening class. Deliberately broader than
#: the one name that caused the incident -- the failure was the SHAPE.
_OVERRIDE_NAME = r"[A-Z][A-Z0-9_]*_(?:ALLOW|SKIP|BYPASS|DISABLE|FORCE)_[A-Z0-9_]+"

#: A *read* of such a variable, in either language. Bare mentions (a constant
#: holding the prefix, a refusal message naming the variable, this docstring)
#: are not reads and are not flagged -- the rule is "must not be honored", not
#: "must not be spelled".
_READ_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"\$\{{{_OVERRIDE_NAME}[:\-}}]"),  # bash ${VAR} / ${VAR:-}
    re.compile(rf"\${_OVERRIDE_NAME}\b"),  # bash $VAR
    re.compile(rf"""environ\.get\(\s*["']{_OVERRIDE_NAME}["']"""),
    re.compile(rf"""environ\[\s*["']{_OVERRIDE_NAME}["']"""),
    re.compile(rf"""getenv\(\s*["']{_OVERRIDE_NAME}["']"""),
)

#: Narrow, auditable escape hatch for a line that reads such a variable in
#: order to REFUSE it (or in a fixture that must set one up). Free-text
#: justification elsewhere does not count -- same posture as Rule 10's
#: ``# skip-token-allowed:`` form.
_ANNOTATION = "gate-override-read-ok:"


def _scanned_files() -> list[Path]:
    return sorted(
        path
        for path in HOOKS_DIR.rglob("*")
        if path.is_file() and path.suffix in {".sh", ".py"}
    )


def _override_reads(path: Path) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    for lineno, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw.strip()
        if stripped.startswith("#"):
            continue
        if _ANNOTATION in raw:
            continue
        if any(pattern.search(raw) for pattern in _READ_PATTERNS):
            findings.append((lineno, stripped))
    return findings


def test_no_hook_honors_an_inheritable_env_override() -> None:
    """No script under scripts/hooks/ may READ a ``*_ALLOW_*`` / ``*_SKIP_*`` /
    ``*_BYPASS_*`` / ``*_DISABLE_*`` / ``*_FORCE_*`` environment variable.

    If this fails on a new gate you are writing: the override belongs in
    ``scripts/hooks/prepush_override_grant.py`` as a minted grant, not in the
    environment. An env var cannot be single-use, cannot be scoped to a commit,
    cannot expire, and leaves no receipt -- so it hands every descendant
    process a permanent, silent bypass of your gate.
    """
    assert HOOKS_DIR.is_dir(), f"expected hooks directory at {HOOKS_DIR}"
    violations = {
        str(path.relative_to(REPO_ROOT)): _override_reads(path)
        for path in _scanned_files()
    }
    violations = {path: found for path, found in violations.items() if found}
    assert not violations, (
        "inheritable env-var gate override(s) read in the hook surface "
        f"(OMN-16480): {violations!r}. Use a scoped single-use grant "
        "(scripts/hooks/prepush_override_grant.py) instead, or annotate a "
        f"rejection line with `{_ANNOTATION} <reason>`."
    )


def test_the_grant_mechanism_is_present() -> None:
    """Anti-hollowing pin: the ratchet above passes trivially if the override
    mechanism is deleted outright, leaving a gate with no escape path at all --
    which gets the gate itself disabled within a week (the verbatim rationale
    the OMN-15059 guard was written under)."""
    assert GRANT_MODULE.is_file(), f"expected the grant module at {GRANT_MODULE}"
    text = GRANT_MODULE.read_text(encoding="utf-8")
    for required in ("def consume(", "def build_grant(", "MAX_TTL_MINUTES"):
        assert required in text, f"expected {required!r} in {GRANT_MODULE}"


def test_both_entry_points_reject_rather_than_honor() -> None:
    """The bash hook and the pytest-side guard must BOTH refuse a leaked
    variable. One entry point honoring it is enough to reproduce the incident,
    and the two guards have drifted before (OMN-15977 Hole 1 existed precisely
    because only the push path was covered)."""
    hook_text = HOOK_SCRIPT.read_text(encoding="utf-8")
    assert "reject_inherited_env_overrides" in hook_text
    assert "reject_inherited_env_overrides\n" in hook_text, (
        "the rejection must be CALLED at hook entry, not merely defined"
    )
    guard_text = PYTEST_GUARD.read_text(encoding="utf-8")
    assert "inherited_override_env_vars" in guard_text
    assert "env_rejection_message" in guard_text


def test_hook_refuses_at_entry_when_an_override_var_is_inherited() -> None:
    """Behavioral proof, not a grep: run the real hook with the leaked variable
    in its environment. It must exit non-zero and name the variable, and it
    must do so BEFORE reaching any pytest invocation -- that ordering is what
    terminates the OMN-16425 recursion instead of feeding it.
    """
    env = dict(os.environ)
    for leaky in (
        "PREPUSH_FULL_SUITE",
        "ENABLE_SMART_TESTS",
        "PREPUSH_ADJACENCY",
        "PREPUSH_PYTEST_ARGS",
    ):
        env.pop(leaky, None)
    # The variable under test, set deliberately.
    env["PREPUSH_ALLOW_LOCAL_FULL_SUITE"] = (
        "1"  # gate-override-read-ok: fixture arms the leak this test refuses
    )

    result = subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0, (
        "expected the hook to REFUSE when an inheritable override variable is "
        f"present; got exit {result.returncode}. stderr={result.stderr!r}"
    )
    assert "PREPUSH_ALLOW_LOCAL_FULL_SUITE" in result.stderr
    assert "REJECTED" in result.stderr
    assert "prepush_override_grant.py mint" in result.stderr, (
        "a refusal without the supported alternative is how gates get disabled"
    )
    assert "running FULL unit suite" not in result.stderr, (
        "the refusal must land before any pytest invocation -- reaching the "
        "suite is the recursion this fix exists to terminate"
    )


def test_hook_rejection_matches_the_whole_prefix_class() -> None:
    """A future ``PREPUSH_ALLOW_SOMETHING_ELSE`` must be refused too, without
    anyone remembering to add it to a list."""
    env = dict(os.environ)
    for leaky in ("PREPUSH_FULL_SUITE", "ENABLE_SMART_TESTS"):
        env.pop(leaky, None)
    env["PREPUSH_ALLOW_NOT_YET_INVENTED"] = (
        "1"  # gate-override-read-ok: fixture arms the leak this test refuses
    )

    result = subprocess.run(
        ["bash", str(HOOK_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "PREPUSH_ALLOW_NOT_YET_INVENTED" in result.stderr
