# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""An unattributed run says WHICH route failed, or that none was reached (OMN-18306).

THE RESIDUAL THIS PINS. OMN-18306 landed on 2026-09-13 and fixed the thing it
was opened for: a terminally-failed delegation used to print zero bytes, and
now renders a full receipt carrying every rung. What it left behind is the
SENTENCE that receipt leads with. ``_write_unattributed_run_files`` states the
route is unattributed using one CONSTANT string:

    "no accepted routing attempt: every rung this run attempted was refused,
     errored, or climbed, so no backend can be named as the author of this
     run's output"

That sentence asserts two things it does not check. It claims rungs WERE
attempted, and it names none of them.

Measured 2026-09-15, both halves, on the same day:

* lane ``lakshman-serving-cap-rulings-1456`` had a run reach ``cloud-gemini-pro``
  and be refused there. The constant was correct that rungs were attempted and
  useless about which (ledger ``docs/tracking/ROLLING_WORK_LEDGER.md:8210``).
* this lane drove a run that failed in 265 ms having attempted NOTHING — the
  request was refused before dispatch, zero attempts on the receipt. For that
  shape the constant is simply false: it says every rung attempted was refused
  when no rung was ever attempted.

Telling those apart is the whole diagnostic value. "Four backends turned this
down" and "a guard refused this before it left the machine" are different
failures with different fixes, and the constant renders them identically.

THE REFUSAL ITSELF IS CORRECT AND STAYS. Declining to name a route for a run
that accepted none is right, and AC3 of OMN-18306 pins that no route is ever
synthesised from the last attempted backend. This module asserts only that the
REASON is accurate about what happened.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from omnibase_infra.cli.cli_delegate import _write_unattributed_run_files
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_delegate_terminal import ModelDelegateTerminal
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

# OMN-18810: the two writers now require the addressing facts the files
# record. These suites are about route attribution and carrier shapes, not
# about addressing, so they state one neutral in-process value; the
# addressing keys themselves are pinned by
# tests/unit/cli/test_omn18810_delegate_run_addressing.py.
_ADDRESSING = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.IN_PROCESS,
    bus="inmemory",
)


def _run(result: dict[str, object], state_root: Path) -> dict[str, object]:
    """Drive the real writer and read back the receipt it wrote."""
    run_id = uuid.uuid4()
    correlation_id = uuid.uuid4()
    _write_unattributed_run_files(
        envelope={
            "run_id": str(run_id),
            "correlation_id": str(correlation_id),
            "status": "failed",
        },
        # OMN-18569 typed the writer's terminal. The recorded shapes below are
        # unchanged; they are validated into the model the runtime's own
        # terminal validates into, which is a stricter input than the loose
        # dict this used to hand over, not a weaker one.
        result=ModelDelegateTerminal.model_validate(result),
        state_root=state_root,
        addressing=_ADDRESSING,
        prompt="draft two paragraphs of rationale prose",
        task_type="document",
        task_type_resolution="fallback",
    )
    receipt = json.loads(
        (state_root / "runs" / str(run_id) / "receipt.json").read_text(encoding="utf-8")
    )
    assert isinstance(receipt, dict)
    return receipt


REACHED_TWO_BACKENDS: dict[str, object] = {
    "response": "",
    "error_message": "TASK_MISMATCH: failed covers_dependencies",
    "attempts": [
        {
            "tier": "local",
            "backend_id": "local-heavy-reasoning",
            "model_id": "Qwen3.6-35B-A3B",
            "failure_class": "quality_gate_failed",
            "acceptance_decision": "climb",
            "acceptance_reason": "heuristic_veto",
        },
        {
            "tier": "cheap_cloud",
            "backend_id": "cloud-gemini-pro",
            "model_id": "gemini-2.5-flash",
            "failure_class": "rate_limited",
            "acceptance_decision": "climb",
            "acceptance_reason": "provider_call_failed",
        },
    ],
}

REACHED_NOTHING: dict[str, object] = {
    "response": "",
    "error_message": "acceptance_criteria validation refused the request",
    "attempts": [],
}


class TestTheReasonNamesWhatWasReached:
    def test_a_run_that_reached_backends_names_them(self, tmp_path: Path) -> None:
        """RED before the fix: the reason was a constant naming no backend."""
        receipt = _run(REACHED_TWO_BACKENDS, tmp_path)
        reason = str(receipt["route_unattributed"])
        assert "local-heavy-reasoning" in reason
        assert "cloud-gemini-pro" in reason

    def test_a_run_that_reached_nothing_says_so(self, tmp_path: Path) -> None:
        """RED before the fix: it claimed every attempted rung was refused.

        Zero rungs were attempted. A sentence about what the rungs did is a
        statement about events that did not happen.
        """
        receipt = _run(REACHED_NOTHING, tmp_path)
        reason = str(receipt["route_unattributed"])
        assert "no backend was reached" in reason
        assert "every rung this run attempted was refused" not in reason

    def test_the_two_shapes_do_not_render_identically(self, tmp_path: Path) -> None:
        """The whole point: these are different failures and must read differently."""
        reached = str(_run(REACHED_TWO_BACKENDS, tmp_path)["route_unattributed"])
        unreached = str(_run(REACHED_NOTHING, tmp_path)["route_unattributed"])
        assert reached != unreached

    def test_a_pre_dispatch_refusal_carries_the_reason_it_was_refused(
        self, tmp_path: Path
    ) -> None:
        """A guard that refused before dispatch is named, not left anonymous."""
        receipt = _run(REACHED_NOTHING, tmp_path)
        assert "acceptance_criteria validation refused the request" in str(
            receipt["failure_reason"]
        )


class TestAttributionStaysFailClosed:
    """AC3 of OMN-18306 is unchanged: no route is ever synthesised."""

    def test_no_route_identity_is_written_for_either_shape(
        self, tmp_path: Path
    ) -> None:
        for result in (REACHED_TWO_BACKENDS, REACHED_NOTHING):
            receipt = _run(result, tmp_path)
            assert receipt["route_attributed"] is False
            for forbidden in ("backend_id", "model_id", "endpoint", "tier"):
                assert forbidden not in receipt, (
                    f"{forbidden} at the top level would name a route this run "
                    "never accepted"
                )

    def test_the_last_attempted_backend_is_never_promoted_to_the_route(
        self, tmp_path: Path
    ) -> None:
        """Naming the rungs in the REASON must not become naming a route.

        The rungs appear as evidence of what was tried. The moment the last of
        them is lifted into a route field, the receipt is claiming an author
        for output nobody accepted — the exact lie the refusal exists to
        prevent.
        """
        receipt = _run(REACHED_TWO_BACKENDS, tmp_path)
        assert receipt.get("backend_id") is None
        assert receipt["route_attributed"] is False
        attempts = receipt["attempts"]
        assert isinstance(attempts, list) and len(attempts) == 2


class TestEveryRungKeepsItsOwnAttribution:
    """Per-rung attribution is what makes a failed run diagnosable at all."""

    def test_each_attempt_carries_backend_tier_and_failure_class(
        self, tmp_path: Path
    ) -> None:
        receipt = _run(REACHED_TWO_BACKENDS, tmp_path)
        attempts = receipt["attempts"]
        assert isinstance(attempts, list)
        for attempt in attempts:
            assert attempt["backend_id"]
            assert attempt["tier"]
            assert attempt["failure_class"]
            assert attempt["acceptance_decision"]
