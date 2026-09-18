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
