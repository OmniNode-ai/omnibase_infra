# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The readback asserts the floors the INSTALLED packages declare (OMN-18752).

The defect, measured on the operator Mac 2026-09-18 by lane
``omn17255-drift-guard-off-registry-build-1835``: the shared plugin CLI venv
carried omnimarket 0.4.121, whose own packaged requirements declare
``omnibase-infra>=0.38.31``, against an installed omnibase-infra 0.38.30. The
layer disagreed with itself and nothing caught it.

Nothing caught it because of two deliberate choices that are individually
right and jointly blind:

* ``install-node-skill-package.sh`` installs with ``--no-deps`` on purpose, so
  the resolver never gets a chance to notice the unsatisfied floor;
* the drift guard only compares omnimarket's own COMMIT against the canonical
  clone, which says nothing about what omnimarket needs underneath it.

``--no-deps`` stays. What this adds is the assertion after the install: read
each installed distribution's own ``Requires-Dist`` -- the pins the installer
shipped inside the artifact, which need no clone, no network and no resolver --
and compare them against what the same venv actually carries. A violated floor
is a named row and the readback exits non-zero, after printing, so the operator
sees the whole picture rather than an early abort.

OMN-17255 shipped this comparison for the OFF-registry path only
(``omnibase_infra/src/omnibase_infra/cli/omnimarket_drift_guard.py``, squash
``b160b9be``), where it reports and never blocks because there is no clone to
reconcile against and no remedy the machine could apply. Here there IS a clone
and there IS a remedy, so the same fact is enforced rather than reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from venv_readback import (
    ReadbackOutcome,
    ReadbackVerdict,
    outcome_for,
    packaged_floor_rows,
)


def _facts(version: str | None, requires: tuple[str, ...] = ()) -> dict[str, object]:
    return {"version": version, "commit": None, "requires": list(requires)}


def test_the_live_defect_is_a_named_finding() -> None:
    """omnimarket 0.4.121 declares omnibase-infra>=0.38.31; 0.38.30 is installed."""
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.121", ("omnibase-infra>=0.38.31,<0.39.0",)),
            "omnibase-infra": _facts("0.38.30"),
        }
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.verdict is ReadbackVerdict.MISMATCH
    assert "omnibase-infra" in row.subject
    assert row.installed == "0.38.30"
    assert "0.38.31" in row.expected
    assert "omnimarket" in row.subject, (
        "the row must name the package that DECLARED the floor, or the reader "
        "cannot tell which artifact disagrees with the venv"
    )
    assert outcome_for(rows) is ReadbackOutcome.DRIFTED


def test_a_satisfied_floor_is_a_match() -> None:
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", ("omnibase-infra>=0.38.31,<0.39.0",)),
            "omnibase-infra": _facts("0.38.32"),
        }
    )
    assert [row.verdict for row in rows] == [ReadbackVerdict.MATCH]
    assert outcome_for(rows) is ReadbackOutcome.IN_SYNC


def test_third_party_requirements_are_not_floors_this_readback_owns() -> None:
    """Only the omni-internal layer is asserted.

    A venv is legitimately a superset with third-party versions resolved by
    other means; claiming authority over them would make this readback fire on
    conditions no reconciler here can fix.
    """
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts(
                "0.4.123",
                ("pydantic>=2.12.5,<3.0.0", "httpx>=0.27.0"),
            ),
        }
    )
    assert rows == []


def test_a_declared_dependency_absent_from_the_venv_is_absent_not_silence() -> None:
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", ("omninode-memory>=0.18.0",)),
            "omninode-memory": _facts(None),
        }
    )
    assert [row.verdict for row in rows] == [ReadbackVerdict.ABSENT]
    assert outcome_for(rows) is ReadbackOutcome.DRIFTED


def test_an_unparseable_requirement_fails_closed() -> None:
    """An unreadable floor is INDETERMINATE, never a pass."""
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", ("omnibase-infra !!! broken",)),
            "omnibase-infra": _facts("0.38.32"),
        }
    )
    assert outcome_for(rows) is not ReadbackOutcome.IN_SYNC


def test_environment_markers_that_do_not_apply_are_skipped() -> None:
    """A requirement gated behind an extra is not a floor this venv must meet."""
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts(
                "0.4.123", ('omnibase-infra>=0.99.0; extra == "dev"',)
            ),
            "omnibase-infra": _facts("0.38.32"),
        }
    )
    assert rows == [], (
        "an extras-gated requirement was treated as a floor the base install "
        "must satisfy, which would fire on every correct venv"
    )


def test_every_installed_package_is_a_source_of_floors_not_just_omnimarket() -> None:
    """The layer is compat -> core -> spi -> infra -> omnimarket.

    omnimarket is the top of it, so its requirements are the tightest, but a
    disagreement lower down is the same class of defect and is not less
    serious for being lower.
    """
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", ("omnibase-infra>=0.38.31",)),
            "omnibase-infra": _facts("0.38.32", ("omnibase-core>=0.47.17",)),
            "omnibase-core": _facts("0.47.16"),
        }
    )
    subjects = {row.subject for row in rows}
    assert any("omnibase-core" in s and "omnibase-infra" in s for s in subjects), (
        "a floor declared by omnibase-infra was not checked"
    )
    assert outcome_for(rows) is ReadbackOutcome.DRIFTED


def test_a_package_declaring_no_omni_internal_pins_yields_no_rows() -> None:
    assert packaged_floor_rows({"omnimarket": _facts("0.4.123", ())}) == []


# --------------------------------------------------------------------------
# floors only — a ceiling is what an override exists to raise
# --------------------------------------------------------------------------


def test_a_version_above_an_exact_pin_is_not_a_floor_violation() -> None:
    """A declared ceiling is not this assertion's business.

    Found live on the operator Mac minutes after the first version shipped:
    omnibase-infra 0.38.32 declares ``omnibase-spi==0.23.3`` while the venv
    carries 0.23.4 -- because omniclaude's ``[tool.uv] override-dependencies``
    declares ``omnibase-spi>=0.23.1,<0.24.0`` and that override is the whole
    point of the mechanism. Asserting the ceiling would fire on every
    sanctioned override, make the converge permanently red on a correct venv,
    and get itself routed around.

    The AC is about FLOORS, and an override raises a ceiling; it never lowers
    a floor. So only the lower bound is asserted.
    """
    rows = packaged_floor_rows(
        {
            "omnibase-infra": _facts("0.38.32", ("omnibase-spi==0.23.3",)),
            "omnibase-spi": _facts("0.23.4"),
        }
    )
    assert [row.verdict for row in rows] == [ReadbackVerdict.MATCH], (
        "an installed version ABOVE a declared exact pin was reported as a "
        "violation; that is the override case, not a defect"
    )


def test_a_version_below_an_exact_pin_is_still_a_floor_violation() -> None:
    """Dropping the ceiling must not drop the floor the same pin implies."""
    rows = packaged_floor_rows(
        {
            "omnibase-infra": _facts("0.38.32", ("omnibase-spi==0.23.3",)),
            "omnibase-spi": _facts("0.23.2"),
        }
    )
    assert [row.verdict for row in rows] == [ReadbackVerdict.MISMATCH]


def test_a_specifier_with_no_lower_bound_is_not_a_floor() -> None:
    """``<0.39.0`` alone says nothing about a minimum, so there is nothing to assert."""
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", ("omnibase-infra<0.39.0",)),
            "omnibase-infra": _facts("0.38.32"),
        }
    )
    assert rows == []


# --------------------------------------------------------------------------
# the capability is required only when there is something to read
# --------------------------------------------------------------------------


def test_no_omni_internal_requirements_needs_no_packaging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A venv with no declared omni-internal floors yields no rows, and asks
    nothing of ``packaging``.

    This shipped wrong and turned a passing integration test red on `dev`
    within the hour. The first version imported ``packaging`` at the top of
    the function and returned an UNREADABLE row when the import failed —
    before looking at whether any floor existed to evaluate. A target venv
    built ``--without-pip`` has no ``packaging``, so a correct repair with
    nothing to assert was reported INDETERMINATE and the readback exited
    non-zero.

    Failing closed is right when the answer matters and cannot be computed.
    It is wrong when there is no question: a capability this function never
    needed must not decide the verdict. The import is attempted only once a
    candidate omni-internal requirement has actually been found.
    """
    import builtins

    from venv_readback import packaged_floor_rows as rows_fn

    real_import = builtins.__import__

    def _refuse_packaging(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("packaging"):
            raise ImportError("packaging is not installed in this interpreter")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _refuse_packaging)

    assert (
        rows_fn(
            {
                "omnimarket": _facts("0.4.119", ()),
                "omnibase-core": _facts("0.47.17", ("pydantic>=2.12.5",)),
            }
        )
        == []
    )


def test_missing_packaging_with_a_real_floor_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a floor DOES exist and cannot be evaluated, that is UNREADABLE.

    The narrowing above must not become a silent pass: an omni-internal
    requirement the interpreter cannot parse is exactly the case this
    readback has to refuse.
    """
    import builtins

    from venv_readback import ReadbackOutcome as Outcome
    from venv_readback import outcome_for as outcome_fn
    from venv_readback import packaged_floor_rows as rows_fn

    real_import = builtins.__import__

    def _refuse_packaging(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("packaging"):
            raise ImportError("packaging is not installed in this interpreter")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _refuse_packaging)

    rows = rows_fn(
        {
            "omnimarket": _facts("0.4.123", ("omnibase-infra>=0.38.31",)),
            "omnibase-infra": _facts("0.38.30"),
        }
    )
    assert rows, "a declared omni-internal floor must still produce a row"
    assert outcome_fn(rows) is not Outcome.IN_SYNC


# --------------------------------------------------------------------------
# the pre-filter decides WHETHER to load a parser; it must not decide the answer
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "omnibase_core>=0.47.17",
        "Omnibase-Core>=0.47.17",
        "omnibase.core>=0.47.17",
        "OMNIBASE_CORE>=0.47.17",
    ],
)
def test_a_floor_is_asserted_whatever_the_name_is_spelled_like(raw: str) -> None:
    """PEP 503 says these are all the same distribution, so all are floors.

    The narrowing above skips a requirement before parsing it, to avoid asking
    for ``packaging`` when there is nothing to evaluate. Deciding that with a
    raw substring test would be fail-OPEN: none of these spellings contains
    the literal ``omnibase-``, every one of them is omni-internal after
    normalization, and a skipped line is a floor that silently stops being
    asserted. That is the failure this readback exists to remove, so the
    pre-filter resolves the normalized name instead.
    """
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", (raw,)),
            "omnibase-core": _facts("0.47.16"),
        }
    )
    assert [row.verdict for row in rows] == [ReadbackVerdict.MISMATCH], (
        f"the floor declared as {raw!r} was not asserted; a spelling the "
        "pre-filter did not recognise dropped a real finding"
    )
    assert outcome_for(rows) is ReadbackOutcome.DRIFTED


def test_an_unparseable_line_that_looks_like_ours_still_fails_closed() -> None:
    """A name the regex cannot locate is refused, not dropped by the pre-filter.

    The pre-filter's fallback exists for exactly this: a line whose name
    cannot be read cannot be judged either, so if it looks omni-internal it
    has to reach the UNREADABLE row rather than be skipped on the way there.
    """
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", ("!!!omnibase_core>=0.47.17",)),
            "omnibase-core": _facts("0.47.16"),
        }
    )
    assert rows, "an unparseable omni-internal-looking line produced no row"
    assert outcome_for(rows) is not ReadbackOutcome.IN_SYNC


def test_a_third_party_name_sharing_a_prefix_substring_is_still_not_ours() -> None:
    """Normalizing the name must not widen what this readback claims authority over."""
    rows = packaged_floor_rows(
        {
            "omnimarket": _facts("0.4.123", ("not-omnibase-core-really>=9.0.0",)),
        }
    )
    assert rows == []
