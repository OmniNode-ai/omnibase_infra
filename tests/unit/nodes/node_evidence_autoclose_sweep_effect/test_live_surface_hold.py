# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106 class (c): a readback against a dead surface is HELD, not judged.

Fourteen tickets in the 2026-08-31 sprint have acceptance criteria that are
live readbacks against ``onex-dev``, which has been in CrashLoopBackOff since
2026-09-02. Their dod_verify checks fail — and until this change the closer
reported that failure as ``GAP_POSTED``, whose comment asserts *"your
acceptance criterion is not met"*. That is a false statement: the check learned
nothing about the criterion, because the thing under test was down. One such
comment accrued per backfill rotation.

The classifier is a message-shape match and is treated as one. Its safety
argument is DIRECTION, not accuracy, and the direction is what these tests pin:
the hold sits after every flip path has already returned, so it is structurally
unreachable from a write. A false positive costs a gap comment that does not
get written; it cannot cost a flip.
"""

from __future__ import annotations

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _LIVE_SURFACE_UNAVAILABLE_SIGNALS,
    _live_surface_unavailable,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)

pytestmark = pytest.mark.unit


def _verdict(*checks: dict[str, object]) -> dict[str, object]:
    return {"checks": list(checks)}


def test_a_crashloop_readback_is_attributed_to_the_surface() -> None:
    evidence_id, signal = _live_surface_unavailable(
        _verdict(
            {
                "evidence_id": "dod-onex-dev-readback",
                "status": "failed",
                "message": (
                    "pod/onex-api-7c9f is in CrashLoopBackOff; readback "
                    "produced no rows"
                ),
            }
        )
    )
    assert evidence_id == "dod-onex-dev-readback"
    assert signal == "crashloopbackoff"


@pytest.mark.parametrize("signal", _LIVE_SURFACE_UNAVAILABLE_SIGNALS)
def test_every_configured_signal_is_reachable(signal: str) -> None:
    """No dead entry in the list: each one must classify something."""
    evidence_id, matched = _live_surface_unavailable(
        _verdict(
            {"evidence_id": "dod-x", "status": "failed", "message": f"boom: {signal}"}
        )
    )
    assert evidence_id == "dod-x"
    assert matched == signal


def test_a_genuine_behaviour_failure_is_not_held() -> None:
    """The direction that matters: a real unmet AC must still read as a gap."""
    assert _live_surface_unavailable(
        _verdict(
            {
                "evidence_id": "dod-behaviour",
                "status": "failed",
                "message": "assert 3 == 4; delegation_events returned 3 rows",
            }
        )
    ) == ("", "")


def test_a_passing_check_is_never_the_attribution() -> None:
    """A check that PASSED proves the surface was reachable.

    Its message may still quote the vocabulary — a test asserting that a pod
    recovers FROM CrashLoopBackOff passes while naming it.
    """
    assert _live_surface_unavailable(
        _verdict(
            {
                "evidence_id": "dod-recovery",
                "status": "verified",
                "message": "pod left CrashLoopBackOff within 90s as required",
            }
        )
    ) == ("", "")


def test_a_skipped_check_is_never_the_attribution() -> None:
    assert _live_surface_unavailable(
        _verdict(
            {
                "evidence_id": "dod-skipped",
                "status": "skipped",
                "message": "connection refused",
            }
        )
    ) == ("", "")


def test_an_unverifiable_check_is_eligible() -> None:
    """UNVERIFIABLE is the other status a dead surface produces."""
    evidence_id, signal = _live_surface_unavailable(
        _verdict(
            {
                "evidence_id": "dod-unver",
                "status": "unverifiable",
                "message": "Unable to connect to the server: dial tcp: i/o timeout",
            }
        )
    )
    assert evidence_id == "dod-unver"
    assert signal


def test_the_first_attributable_check_wins_and_the_rest_are_not_scanned() -> None:
    evidence_id, _ = _live_surface_unavailable(
        _verdict(
            {"evidence_id": "dod-1", "status": "failed", "message": "assert 1 == 2"},
            {
                "evidence_id": "dod-2",
                "status": "failed",
                "message": "connection refused",
            },
            {"evidence_id": "dod-3", "status": "failed", "message": "no such host"},
        )
    )
    assert evidence_id == "dod-2"


@pytest.mark.parametrize(
    "verdict",
    [
        {},
        {"checks": None},
        {"checks": "not a list"},
        {"checks": []},
        {"checks": [None, 7, "x"]},
        {"checks": [{"status": "failed"}]},
        {"checks": [{"status": "failed", "message": None}]},
    ],
)
def test_a_payload_this_classifier_cannot_read_holds_nothing(
    verdict: dict[str, object],
) -> None:
    """Unreadable is not held.

    A hold suppresses the gap comment, so an unreadable payload must fall
    through to the existing behaviour rather than silently swallowing every
    gap the sweep would otherwise report.
    """
    assert _live_surface_unavailable(verdict) == ("", "")


def test_a_nameless_check_still_attributes() -> None:
    evidence_id, signal = _live_surface_unavailable(
        _verdict({"status": "failed", "message": "no such container"})
    )
    assert evidence_id == "<unnamed check>"
    assert signal == "no such container"


def test_the_hold_decision_is_a_skip_not_a_flip_or_a_gap() -> None:
    """The taxonomy invariant, so a later rename cannot make a hold count.

    Every consumer that tallies outcomes branches on this prefix; a decision
    spelled ``gap_*`` would be counted as a posted gap and one spelled
    ``flipped`` would be counted as a close.
    """
    decision = EnumEvidenceAutocloseDecision.SKIPPED_LIVE_SURFACE_UNAVAILABLE
    assert decision.value.startswith("skipped_")
    assert decision is not EnumEvidenceAutocloseDecision.GAP_POSTED
    assert decision is not EnumEvidenceAutocloseDecision.FLIPPED


def test_the_hold_is_unreachable_from_any_write_path() -> None:
    """The safety argument, read off the source rather than asserted in prose.

    The classifier's only call site must appear AFTER every ``FLIPPED`` return
    in `_process_ticket`. If a later edit moves it above one, this fails —
    which is the point: the invariant is placement, not the signal list.
    """
    import inspect

    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
        handler_evidence_autoclose_sweep as sweep_mod,
    )

    source = inspect.getsource(sweep_mod.HandlerEvidenceAutocloseSweep._process_ticket)
    call_sites = [
        index
        for index, line in enumerate(source.splitlines())
        if "_live_surface_unavailable(" in line
    ]
    assert len(call_sites) == 1, "the classifier must have exactly one call site"
    flip_returns = [
        index
        for index, line in enumerate(source.splitlines())
        if "EnumEvidenceAutocloseDecision.FLIPPED" in line
    ]
    assert flip_returns, "expected at least one flip return to order against"
    assert call_sites[0] > max(flip_returns)
