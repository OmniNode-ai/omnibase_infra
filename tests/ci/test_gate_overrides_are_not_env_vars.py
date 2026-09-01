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
import signal
import subprocess
from pathlib import Path

import pytest

from tests.ci._prepush_lab_isolation import network_free_lab_env

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
        # OMN-16489: this test exercises FIRST-entry behavior, so the recursion
        # sentinel an outer hook run exports must not leak in.
        "ONEX_PREPUSH_HOOK_ACTIVE",
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
    for leaky in (
        "PREPUSH_FULL_SUITE",
        "ENABLE_SMART_TESTS",
        # OMN-16489: first-entry behavior — strip the recursion sentinel.
        "ONEX_PREPUSH_HOOK_ACTIVE",
    ):
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


# --------------------------------------------------------------------------
# The PLACEMENT-map class (OMN-17441)
# --------------------------------------------------------------------------
# A second inheritable class, and a DIFFERENT one -- conflating them would
# overstate the risk and get the distinction dropped in the next refactor.
#
# `PREPUSH_LOAD_OVERRIDE_MAP` / `PREPUSH_SLOT_OVERRIDE_MAP` (and, found live in
# scripts/hooks/prepush_dispatch.sh alongside them, `PREPUSH_MEM_OVERRIDE_MAP`
# and `PREPUSH_UV_OVERRIDE_MAP`) are read by `prepush_map_lookup` to fake the
# per-host load ratio, slot state, free memory and uv version that the picker
# would otherwise measure over ssh. They exist ONLY as a test-injection seam.
#
# They cannot manufacture a false PASS: the verdict is still a real pytest exit
# bound to the tree by a completion marker. What an inherited value CAN do is
# steer PLACEMENT -- send a real push to a host the hook believes is idle, or
# to a slot it believes is free, on evidence that came from a stale fixture in
# an operator's shell rather than from the host. omnibase_infra#3091 named this
# residual in its own PR body and deferred it; this is that ticket.
#
# Matched by the `PREPUSH_*_OVERRIDE_MAP` SHAPE for the OMN-16480 reason: the
# two names in the ticket are not the class, and a future
# `PREPUSH_DISK_OVERRIDE_MAP` must not need anyone to remember a list.
_PLACEMENT_MAP_NAMES = (
    "PREPUSH_LOAD_OVERRIDE_MAP",
    "PREPUSH_SLOT_OVERRIDE_MAP",
    "PREPUSH_MEM_OVERRIDE_MAP",
    "PREPUSH_UV_OVERRIDE_MAP",
)


#: How long a REFUSAL is allowed to take. The rejection runs at hook entry,
#: before the selector, before any host probe -- it is sub-second in practice.
#: Anything past this window means the hook did NOT stop at entry.
_ENTRY_REFUSAL_BUDGET_SECONDS = 45


def _hook_env_without_the_harness_marker() -> dict[str, str]:
    """A real (non-harness) invocation's environment.

    Lab isolation is applied unconditionally. Without the harness marker these
    cases exercise the pre-fix path too, and the pre-fix path proceeds into
    OFF-BOX ROUTING and ssh-probes every row of the real host table -- a unit
    test taking a real lab host's exclusive slot, which is the very hazard
    `_prepush_lab_isolation` was written for. Its map names no real label, so
    every row resolves unfit with zero ssh.

    That the isolation seam is itself a member of the class under test is not a
    conflict, it is the point: the SLOT case below arms its leak by setting that
    same variable, so one value serves as both the injection and the fixture.
    """
    env = dict(os.environ)
    for leaky in (
        "PREPUSH_FULL_SUITE",
        "ENABLE_SMART_TESTS",
        "PREPUSH_ADJACENCY",
        "PREPUSH_PYTEST_ARGS",
        "ONEX_PREPUSH_HOOK_ACTIVE",
        # The harness marker the carve-out keys on. Removed here so these cases
        # exercise the REAL-invocation path -- otherwise every one of them would
        # pass by taking the exemption.
        "PYTEST_CURRENT_TEST",
        *_PLACEMENT_MAP_NAMES,
    ):
        env.pop(leaky, None)
    env.update(network_free_lab_env())
    return env


def _run_hook_bounded(env: dict[str, str]) -> tuple[int | None, str]:
    """Run the hook, killing the whole process group if it does not stop.

    ``(returncode, stderr)``, with ``None`` for "still running when the budget
    expired". A plain ``subprocess.run(timeout=...)`` is not enough here: RED
    for these cases means the hook sails past entry into the off-box
    queue-and-wait, which sleeps against a 900s budget, and leaving that running
    (or leaving its probe children orphaned) from inside a test is the
    distributed form of the recursion this module exists to stop.

    A timeout is reported as a FAILURE to refuse rather than as an error,
    because that is exactly what it is.
    """
    process = subprocess.Popen(
        ["bash", str(HOOK_SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        _, stderr = process.communicate(timeout=_ENTRY_REFUSAL_BUDGET_SECONDS)
        return process.returncode, stderr
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        _, stderr = process.communicate()
        return None, stderr


@pytest.mark.parametrize("name", _PLACEMENT_MAP_NAMES)
def test_hook_refuses_an_inherited_placement_override_map(name: str) -> None:
    """RED against dev: today the hook silently HONORS each of these.

    Behavioral, not a grep, and for the same reason the OMN-16480 pair above is
    behavioral: a static scan passes on a hook whose rejection branch has been
    made unreachable.
    """
    env = _hook_env_without_the_harness_marker()
    env[name] = "h999=0.01"

    returncode, stderr = _run_hook_bounded(env)

    assert returncode is not None and returncode != 0, (
        f"the hook honored an inherited {name} instead of refusing it at entry "
        f"(exit {returncode}); a stale fixture value from an operator shell can "
        "then route a real push to a host that was never probed. "
        f"stderr tail={stderr[-600:]!r}"
    )
    assert name in stderr, stderr
    assert "REJECTED" in stderr, stderr


def test_hook_rejection_matches_the_whole_placement_map_class() -> None:
    """A future `PREPUSH_DISK_OVERRIDE_MAP` must be refused with nobody
    remembering to add it -- the OMN-16480 prefix lesson applied to the suffix
    that actually delimits this class."""
    env = _hook_env_without_the_harness_marker()
    env["PREPUSH_NOT_YET_INVENTED_OVERRIDE_MAP"] = "h999=free"

    returncode, stderr = _run_hook_bounded(env)

    assert returncode is not None and returncode != 0, stderr[-600:]
    assert "PREPUSH_NOT_YET_INVENTED_OVERRIDE_MAP" in stderr, stderr


def test_the_pytest_harness_may_still_inject_placement_maps() -> None:
    """AC2, and the reason the guard is conditional rather than absolute.

    `tests/ci/_prepush_lab_isolation.py` sets `PREPUSH_SLOT_OVERRIDE_MAP` to a
    map naming no real row, which makes every host unfit with zero ssh. That is
    what stops the hook-subprocess tests in this directory from shipping a real
    git bundle to a real lab host and burning its cores for an hour (observed
    live 2026-08-30, minutes after OMN-16991 removed the accidental containment
    that had been hiding it). A rejection with no carve-out would delete that
    isolation and re-arm exactly the distributed recursion it prevents.

    The carve-out keys on `PYTEST_CURRENT_TEST`, which pytest itself sets per
    test and clears afterwards. It is not a knob this repo invented, cannot be
    widened by anyone here, and -- because this class cannot manufacture a PASS
    -- an implausible leak of it alongside a map costs routing accuracy, never
    a verdict.
    """
    env = _hook_env_without_the_harness_marker()
    env["PYTEST_CURRENT_TEST"] = "tests/ci/test_gate_overrides_are_not_env_vars.py::x"

    _, stderr = _run_hook_bounded(env)

    assert "REJECTED" not in stderr, (
        "the harness carve-out is gone, so the lab-isolation seam every "
        f"hook-subprocess test depends on is broken: {stderr!r}"
    )
    assert "selection:" in stderr, (
        "the hook never reached the selector, so this proves nothing about the "
        f"carve-out: {stderr!r}"
    )


def test_the_lab_isolation_seam_is_in_the_class_the_carve_out_admits() -> None:
    """Pins the two halves together.

    If the isolation helper ever switched to a variable outside this class, the
    carve-out above would be dead code protecting nothing, and nobody would find
    out until a test shipped a bundle to a lab host again.
    """
    from scripts.hooks.prepush_override_grant import (
        inherited_placement_map_env_vars,
    )

    assert set(network_free_lab_env()) <= set(_PLACEMENT_MAP_NAMES)
    assert inherited_placement_map_env_vars(network_free_lab_env()) == sorted(
        network_free_lab_env()
    )


def test_the_placement_map_helper_honors_the_harness_marker_and_emptiness() -> None:
    """The pure decision function, exercised directly.

    An empty value is not "set" -- that matches the `[ -n "$VAR" ]` semantics
    the hook has always used, so exporting an empty string is neither an arming
    signal nor a spurious refusal. `tests/unit/scripts/test_prepush_host_table.py`
    relies on it: it clears these maps by exporting them empty.
    """
    from scripts.hooks.prepush_override_grant import (
        inherited_placement_map_env_vars,
    )

    assert inherited_placement_map_env_vars({"PREPUSH_LOAD_OVERRIDE_MAP": "h=0.1"}) == [
        "PREPUSH_LOAD_OVERRIDE_MAP"
    ]
    assert inherited_placement_map_env_vars({"PREPUSH_LOAD_OVERRIDE_MAP": ""}) == []
    assert inherited_placement_map_env_vars({"PREPUSH_LOAD_OVERRIDE_MAP": "   "}) == []
    assert (
        inherited_placement_map_env_vars({"PREPUSH_HOST_OVERRIDE_H200": "x"}) == []
    ), (
        "the per-row host override is a different, already-governed knob; "
        "widening this class to swallow it would break the host table"
    )
    assert (
        inherited_placement_map_env_vars(
            {"PREPUSH_LOAD_OVERRIDE_MAP": "h=0.1", "PYTEST_CURRENT_TEST": "t::x (call)"}
        )
        == []
    )


def test_both_entry_points_reject_the_placement_map_class() -> None:
    """The bash hook and the pytest-side guard, same as OMN-16480.

    The pytest guard's check is NOT vacuous despite living inside pytest: it
    runs at `pytest_configure`, before collection, and `PYTEST_CURRENT_TEST` is
    set per TEST -- so a direct `uv run pytest tests/unit/` launched from a shell
    carrying a leaked map is refused there exactly as a push would be, while a
    subprocess spawned from inside a running test keeps its exemption.
    """
    hook_text = HOOK_SCRIPT.read_text(encoding="utf-8")
    assert "_OVERRIDE_MAP" in hook_text, (
        "the hook does not mention the placement-map class at all"
    )
    guard_text = PYTEST_GUARD.read_text(encoding="utf-8")
    assert "inherited_placement_map_env_vars" in guard_text
    assert "placement_map_rejection_message" in guard_text
