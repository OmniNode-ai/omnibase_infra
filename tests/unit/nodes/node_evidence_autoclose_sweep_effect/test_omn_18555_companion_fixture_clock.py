# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18555 — a companion fixture may not pin an absolute instant.

THE DEFECT, MEASURED
--------------------
On 2026-09-17 ``omnibase_infra`` ``dev`` was red for every open pull
request. Nine of the eleven tests in
``test_omn_18125_coverage_corpus.py`` failed, each with
``IndexError: tuple index out of range`` at ``result.outcomes[0]``,
because the handler returned no outcomes at all.

The cause was one line — ``_merged_pr`` binding its companion timestamp
to a constant::

    recent = "2026-09-10T02:00:00Z"

That value is the ``merged_at`` and ``updated_at`` of the only companion
the fake ``gh`` layer yields, and the handler enumerates companions in
two windows it computes from the WALL CLOCK: the forward window from
``lookback_hours`` and the wider backfill window from
``backfill_lookback_hours``, filtered on those two strings. A constant
therefore does not describe a fixture; it describes a fixture with an
expiry date.

The expiry was the 168-hour backfill default. Holding everything else
fixed and moving only that literal, measured in a clean worktree:

A fixture ``merged_at`` of now minus 167 hours yields one outcome, one
``companions_scanned`` and a ``backfill_pool_size`` of one. The same
fixture at now minus 169 hours yields zero of all three.

2026-09-10T02:00:00Z plus 168 hours is 2026-09-17T02:00:00Z. After that
instant the companion was in neither window, so the sweep had nothing to
report on and every test that indexed its first outcome raised.

WHAT THIS MODULE PINS, AND WHY IT IS SHAPED THIS WAY
----------------------------------------------------
The convention this defect broke was already the overwhelming local
standard, which is what makes a check worth having rather than a new
rule imposed on existing code: at the commit this was written against,
thirteen of the fifteen modules in this directory already derived their
companion timestamp from ``datetime.now(tz=UTC)``. Exactly two pinned a
constant, and both were rotting — the second,
``test_omn_18135_readback_join.py``, was twelve hours from red when this
was written. So the check does not introduce a convention; it refuses
the two deviations from one.

**The rule is bound to** ``updated_at`` **deliberately.** It is the key
the window filter reads, and in this directory it is carried by exactly
one kind of record: a companion in the enumerated list. Other fixtures
here legitimately carry absolute ``merged_at`` values — the cited
product pull requests in ``test_omn_18233_verified_supersession.py``,
the closed-bump payloads in the two OMN-16106 modules — and none of
those records carries ``updated_at``, because nothing ever filters them
by window. Keying on ``updated_at`` is what lets this refuse the real
defect without an allowlist, and an allowlist is what a check of this
shape must not need.

**There is no suppression marker and no exempt list.** A fixture that
genuinely needs a fixed instant needs it relative to the clock the code
under test reads, which is what the offset idiom expresses.

WHAT IT CANNOT DO, STATED RATHER THAN IMPLIED
----------------------------------------------
It reads syntax. A fixture that computed a stale instant at runtime —
say, from a constant epoch — would pass this and still rot. What it
removes is the *silent* recurrence of the exact shape that took ``dev``
down: a literal sitting in a fixture, correct on the day it was written,
with nothing in the tree recording that it has a shelf life.

It also cannot substitute for running the tests, and deliberately does
not try. The enforcement point is this suite itself: it is selected by
the change-aware selector whenever anything in this directory changes,
which is the only moment a new constant can be introduced.

Note on what a test-selection change would NOT have fixed. The rotted
test failed with no code change of any kind, so no path-to-test mapping
could have selected it — there was no changed path to map from. That is
why the mechanism here is an invariant on the fixture and not a wider
selector.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

pytestmark = pytest.mark.unit

#: The directory whose companion fixtures this module governs.
_FIXTURE_DIR = Path(__file__).parent

#: The dict key the handler's window filter reads. See the module docstring
#: for why the rule is bound to this key and not to ``merged_at``.
_WINDOW_FILTERED_KEY = "updated_at"

#: An ISO-8601 instant as these fixtures spell one. Matched against string
#: constants only; a name bound to a computed expression never reaches it.
_ABSOLUTE_INSTANT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")


def _string_constant_bindings(tree: ast.Module) -> dict[str, list[tuple[int, str]]]:
    """Every name in the module bound anywhere to a string constant.

    Scope is deliberately ignored. A module that binds one name to a
    constant in one function and to a clock offset in another is already
    ambiguous to a reader, and resolving that ambiguity in favour of the
    constant is the safe direction for a check whose whole purpose is to
    refuse a fixture that silently expires.
    """
    bindings: dict[str, list[tuple[int, str]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                bindings.setdefault(target.id, []).append(
                    (node.value.lineno, node.value.value)
                )

    return bindings


def _window_filtered_values(tree: ast.Module) -> Iterator[ast.expr]:
    """The value expression of every ``updated_at`` entry in a dict literal."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == _WINDOW_FILTERED_KEY:
                yield value


def _pinned_instants(source: str, label: str) -> list[str]:
    """Companion timestamps in ``source`` that are pinned to a constant.

    Returns one human-readable finding per violation, naming the file, the
    line and the literal, so a failure is actionable without a diagnosis
    pass. An empty list means every companion record in the module derives
    its timestamp from something other than a string literal.
    """
    tree = ast.parse(source)
    bindings = _string_constant_bindings(tree)

    findings: list[str] = []
    for value in _window_filtered_values(tree):
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            if _ABSOLUTE_INSTANT_RE.match(value.value):
                findings.append(
                    f"{label}:{value.lineno}: companion "
                    f"{_WINDOW_FILTERED_KEY} is the literal {value.value!r}"
                )
            continue
        if not isinstance(value, ast.Name):
            continue
        for lineno, literal in bindings.get(value.id, ()):
            if _ABSOLUTE_INSTANT_RE.match(literal):
                findings.append(
                    f"{label}:{lineno}: companion "
                    f"{_WINDOW_FILTERED_KEY} reads `{value.id}`, which is "
                    f"bound to the literal {literal!r}"
                )
    return findings


# -- the fixture modules this governs ---------------------------------------


def _default_request() -> ModelEvidenceAutocloseSweepRequest:
    """The request as a dispatcher builds it, carrying only required fields.

    Every window bound this module asserts against is read from the CONTRACT's
    own default rather than restated here, so a change to either window makes
    this move with it instead of silently disagreeing.
    """
    return ModelEvidenceAutocloseSweepRequest(
        correlation_id=uuid4(),
        occ_repo="OmniNode-ai/onex_change_control",
    )


def _fixture_modules() -> list[Path]:
    return sorted(p for p in _FIXTURE_DIR.glob("test_*.py") if p != Path(__file__))


def test_no_companion_fixture_pins_an_absolute_instant() -> None:
    """RED at 518a3e5de on two modules; green once both use the clock.

    This is the check. Everything below it proves the check discriminates.
    """
    findings: list[str] = []
    for path in _fixture_modules():
        findings.extend(_pinned_instants(path.read_text(), path.name))

    assert not findings, (
        "A companion fixture pins an absolute instant, so it will stop being "
        "enumerated once the handler's backfill window "
        f"({_default_request().backfill_lookback_hours}h by "
        "default) passes it, and every test reading its outcome will raise. "
        "Derive it from the clock, as the rest of this directory does:\n"
        '    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")\n'
        + "\n".join(findings)
    )


# -- controls: a check that flags everything is not a check -----------------


_PINNED_SOURCE = """
def _merged_pr(number):
    recent = "2026-09-10T02:00:00Z"
    return {"number": number, "updated_at": recent, "merged_at": recent}
"""

_CLOCK_SOURCE = """
from datetime import UTC, datetime, timedelta


def _merged_pr(number):
    recent = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"number": number, "updated_at": recent, "merged_at": recent}
"""

_INLINE_PINNED_SOURCE = """
def _merged_pr(number):
    return {"number": number, "updated_at": "2026-09-10T02:00:00Z"}
"""

_UNFILTERED_ABSOLUTE_SOURCE = """
def _cited_product_pr():
    return {"number": 3448, "state": "closed", "merged_at": "2026-09-12T03:14:52Z"}
"""


def test_the_check_catches_the_shape_that_took_dev_down() -> None:
    """Positive control. The exact pre-fix idiom must be reported."""
    findings = _pinned_instants(_PINNED_SOURCE, "planted.py")

    assert len(findings) == 1, findings
    assert "2026-09-10T02:00:00Z" in findings[0]
    assert "recent" in findings[0]


def test_the_check_catches_the_literal_written_in_place() -> None:
    """Positive control. Skipping the intermediate name is the same defect."""
    findings = _pinned_instants(_INLINE_PINNED_SOURCE, "planted.py")

    assert len(findings) == 1, findings
    assert "2026-09-10T02:00:00Z" in findings[0]


def test_the_check_passes_the_clock_idiom_the_directory_already_uses() -> None:
    """Negative control. The convention must not be reported as a violation."""
    assert _pinned_instants(_CLOCK_SOURCE, "conventional.py") == []


def test_the_check_ignores_an_absolute_instant_nothing_filters_by_window() -> None:
    """Negative control, and the reason the rule is keyed on one dict key.

    A cited product pull request carries an absolute ``merged_at`` and no
    ``updated_at``, because no window filter ever reads it. Reporting it
    would make this check indistinguishable from one that flags every date
    in the tree.
    """
    assert _pinned_instants(_UNFILTERED_ABSOLUTE_SOURCE, "cited.py") == []


def test_the_conventional_idiom_lands_inside_both_windows() -> None:
    """The convention is correct, not merely consistent.

    The one-hour offset the directory uses must sit inside the handler's
    NARROWER window, so a fixture written this way is enumerated by the
    forward arm and does not depend on the backfill arm being armed.
    """
    request = _default_request()
    now = datetime.now(tz=UTC)
    stamp = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    forward_since = (now - timedelta(hours=request.lookback_hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    backfill_since = (now - timedelta(hours=request.backfill_lookback_hours)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    assert stamp >= forward_since
    assert stamp >= backfill_since
