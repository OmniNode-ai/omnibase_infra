# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared OMN-18056 fixture support for the evidence-autoclose closer tests.

Not a test module itself (no ``test_`` prefix, not collected by pytest).

WHY THIS EXISTS
---------------
OMN-18056 added two conjuncts inside the single ``if all_verified:`` block
every trigger traverses:

1. **the AC-binding gate** -- every acceptance criterion parsed from the
   ticket body must be named by a check that both DECLARES it (``binds_ac``)
   and VERIFIED. A body with no parseable criteria holds too, because a
   closer that cannot read a ticket's criteria can say nothing about them.
2. **the re-draw** -- the first eligible observation of a verdict ARMS a
   re-draw and writes no Done; a later tick that recomputes the SAME
   fingerprint flips.

Every pre-existing closer test that asserts a FLIP was written against
fixtures that satisfy neither: unlabelled (usually absent) criteria, checks
carrying no bindings, and a single tick. Those 69 failures were the
measurement, not a regression -- so the repair is to make each fixture
DECLARE what the gate now asks for, never to loosen the gate.

The three moves, and every flip-expecting test needs all three:

* ``BOUND_AC_DESCRIPTION`` (or ``bound_ac_description``) on the issue body,
  so exactly one labelled criterion parses;
* ``BOUND_AC_CHECKS`` in the verdict's ``checks`` list, so that criterion is
  named by a verified probative check. The counters are read from the
  verdict's own count fields, never from this list, so adding it moves no
  arithmetic and leaves every other conjunct's fixture untouched;
* two ``handle()`` calls -- the first arms the re-draw, the second flips.
  ``redraw_marker_comment`` is the alternative for a DRY-RUN fixture, where
  the first tick writes nothing and so can never arm anything.
"""

from __future__ import annotations

from typing import Any

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    EnumEvidenceAutocloseDecision,
    _sweep_comment_marker,
    _verdict_fingerprint,
)

#: The canonical label these fixtures bind. One criterion, not several: the
#: counting rule ``len(items) > verified_count`` is a SEPARATE conjunct, and a
#: fixture that accidentally trips it would report a coverage gap where the
#: test means to exercise something else entirely.
BOUND_AC_LABEL = "AC1"

#: A ticket body carrying exactly one labelled, parseable acceptance
#: criterion under a recognised heading. No checkbox markers -- an unchecked
#: ``- [ ]`` is its own coverage rule and would hold the flip before the
#: binding gate ever got to agree with it.
BOUND_AC_DESCRIPTION = (
    "## Acceptance criteria\n"
    "\n"
    "- AC1: the behaviour this fixture exercises is proven by a verified "
    "probative check.\n"
)


def bound_ac_description(*extra_lines: str) -> str:
    """``BOUND_AC_DESCRIPTION`` with ``extra_lines`` appended after it.

    For the fixtures that also need a ``Gate:`` declaration or a cited PR in
    the body. The extra lines land AFTER the criteria section and outside any
    list, so they add no criterion of their own.
    """
    if not extra_lines:
        return BOUND_AC_DESCRIPTION
    return BOUND_AC_DESCRIPTION + "\n" + "\n".join(extra_lines) + "\n"


def bound_ac_check(
    *,
    evidence_id: str = "omn18056-bound-check",
    labels: tuple[str, ...] = (BOUND_AC_LABEL,),
    status: str = "verified",
    proof_class: str = "behavior",
) -> dict[str, Any]:
    """One dod_verify check record declaring the criteria it covers.

    ``status`` and ``labels`` are parameters because "declared by a check that
    did not verify" is a DIFFERENT fact from "declared by nothing", and a
    fixture exercising the near-miss needs to say which one it means.
    """
    return {
        "evidence_id": evidence_id,
        "status": status,
        "proof_class": proof_class,
        "binds_ac": list(labels),
    }


#: The default ``checks`` payload: one verified probative check naming AC1.
BOUND_AC_CHECKS: list[dict[str, Any]] = [bound_ac_check()]


def bound_ac_checks() -> list[dict[str, Any]]:
    """A fresh mutable copy of :data:`BOUND_AC_CHECKS`.

    Callers hand this straight into a verdict dict that other fixtures then
    edit; sharing one list across tests would let one test's edit reach
    another's payload.
    """
    return [bound_ac_check()]


def redraw_marker_comment(
    *,
    total_checks: int,
    verified_count: int,
    failed_count: int = 0,
    non_probative_count: int = 0,
    behavior_proving_count: int,
) -> str:
    """A comment body carrying the re-draw marker for these counters.

    Seeded onto a fixture's comment history so the run under test is the
    SECOND observation of the verdict and may therefore flip. Needed wherever
    a real first tick cannot arm the re-draw itself -- a DRY-RUN, whose whole
    contract is that it writes nothing, so it can never leave the marker a
    later tick reads.

    The marker is built from the production functions, not from a copied
    literal: a fixture that spelled its own marker would keep passing after a
    marker-format change that had silently disarmed the real re-draw.
    """
    fingerprint = _verdict_fingerprint(
        total_checks=total_checks,
        verified_count=verified_count,
        failed_count=failed_count,
        non_probative_count=non_probative_count,
        behavior_proving_count=behavior_proving_count,
    )
    marker = _sweep_comment_marker(
        EnumEvidenceAutocloseDecision.SKIPPED_REDRAW_PENDING,
        (fingerprint,),
    )
    return f"Re-draw armed on an earlier tick (test fixture).\n\n{marker}"
