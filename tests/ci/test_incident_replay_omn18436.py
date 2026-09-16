# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18436 incident replay -- the settle budget that could not fit the boot.

THE ARTIFACTS ARE THE RECEIPTS THE INCIDENT EMITTED, byte for byte, fetched from
the GitHub artifact store rather than reconstructed. Both are compose-dev
lab-pass receipts from the .201 dev lane on 2026-09-16, and between them they
carry the whole defect:

``6e41f3e0`` (03:11:20Z-03:39:20Z, artifact 10429237295)
    ``deployed_revision`` **ok**, "converged after 0h24m" -- the lane WAS running
    the merge sha. The settle budget the job had left after that convergence was
    **238 s**, and the lane's measured boot on this same lane is 463-668 s. So
    ``ready_main`` answered 503 with ``is_running: false`` (the runtime serving
    HTTP before its kernel had started) and ``ready_effects`` answered Errno 111.
    The receipt reads FAIL. Nothing was wrong with the lane; the job ran out of
    clock, and rule 24(b) then refused that sha for staging delivery.

``4853e0e1`` (01:41:28Z-02:06:29Z, artifact 10427636135)
    The inverse, and the second defect. ``deployed_revision`` **fail** -- "lane
    at ba3f44b50495, which does not contain merge sha 4853e0e12b19 (relation:
    ancestor on dev)" -- while all four HTTP checks passed "after 0s of a 179s
    settle budget". The PREVIOUS generation answered instantly, because the
    probe step runs under ``if: always()``. Four green reads about a container
    the receipt never names.

Together they are the shape that made the old design fail-always: the HTTP
checks passed only when ``deployed_revision`` failed, and failed exactly when it
succeeded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.ci.lab_pass_receipt import (
    GENERATION_CHECK,
    ModelSettleBudget,
    generation_check,
    parse_generation,
    parse_receipt,
)
from scripts.ci.lane_settle_budget import load_declaration

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "omn18436"
TIMED_OUT = FIXTURES / "lab-pass-receipt-compose-dev-6e41f3e0.json.captured"
UNBOUND = FIXTURES / "lab-pass-receipt-compose-dev-4853e0e1.json.captured"

#: The ceiling the job carried at the incident, and the tail it reserved.
CEILING_AT_THE_INCIDENT_SECONDS = 30 * 60
RESERVED_TAIL_SECONDS = 120
#: The ceiling this change derives from the declared parts.
CEILING_AFTER_THE_FIX_SECONDS = 45 * 60

_BUDGET_RE = re.compile(r"the full (?P<seconds>\d+)s settle budget")


def _budget_the_job_actually_had(path: Path) -> int:
    """Read the grant out of the receipt's OWN evidence, never from prose."""
    receipt = parse_receipt(path.read_text(encoding="utf-8"))
    grants = {
        int(match.group("seconds"))
        for check in receipt.checks
        if (match := _BUDGET_RE.search(check.evidence))
    }
    assert len(grants) == 1, f"{path.name}: expected one grant, saw {sorted(grants)}"
    return grants.pop()


def _elapsed_implied_by(grant_seconds: int, ceiling_seconds: int) -> int:
    """Invert the arithmetic the incident's own job performed.

    The shipped shell computed ``SETTLE = ceiling - elapsed - tail``, so a
    recorded grant and the ceiling that produced it pin the elapsed time. This
    is derived from the captured bytes, not estimated.
    """
    return ceiling_seconds - grant_seconds - RESERVED_TAIL_SECONDS


def test_the_real_guard_reports_the_incident_budget_as_unaffordable() -> None:
    """The guard's verdict on the artifact: REJECT.

    ``6e41f3e0`` converged and then failed on timing. The declared budget was
    not a budget the job could afford, and this is the check that now says so
    in the receipt instead of the shortfall arriving as ``ready_effects: fail``.
    """
    grant = _budget_the_job_actually_had(TIMED_OUT)
    assert grant == 238, "the captured receipt's own recorded grant"

    declared = load_declaration("compose-dev").settle_budget_seconds
    budget = ModelSettleBudget(
        lane="compose-dev",
        declared_seconds=declared,
        affordable_seconds=grant,
        job_ceiling_seconds=CEILING_AT_THE_INCIDENT_SECONDS,
        elapsed_seconds=_elapsed_implied_by(grant, CEILING_AT_THE_INCIDENT_SECONDS),
        reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        source="config/lab_pass_settle_budget.yaml",
    )
    assert budget.sufficient is False
    assert budget.granted_seconds == grant
    assert str(declared) in budget.evidence
    assert str(grant) in budget.evidence
    assert "run out of CLOCK" in budget.evidence


def test_the_same_guard_reports_the_fixed_ceiling_as_affordable() -> None:
    """The discriminator, over the SAME captured bytes with ONE input changed.

    Without it, a guard that called every budget unaffordable would replay this
    incident perfectly and refuse every run forever.
    """
    grant = _budget_the_job_actually_had(TIMED_OUT)
    elapsed = _elapsed_implied_by(grant, CEILING_AT_THE_INCIDENT_SECONDS)
    declared = load_declaration("compose-dev").settle_budget_seconds
    affordable = CEILING_AFTER_THE_FIX_SECONDS - elapsed - RESERVED_TAIL_SECONDS
    budget = ModelSettleBudget(
        lane="compose-dev",
        declared_seconds=declared,
        affordable_seconds=affordable,
        job_ceiling_seconds=CEILING_AFTER_THE_FIX_SECONDS,
        elapsed_seconds=elapsed,
        reserved_tail_seconds=RESERVED_TAIL_SECONDS,
        source="config/lab_pass_settle_budget.yaml",
    )
    assert budget.sufficient is True
    assert budget.granted_seconds == declared


def test_the_lane_in_the_timed_out_receipt_was_booting_not_broken() -> None:
    """Read from the captured bytes, because it is the whole point.

    The receipt FAILED, and a reader had no way to tell this apart from an
    unhealthy lane. ``is_running: false`` on a converged lane inside a
    238-second grant is a lane mid-boot.
    """
    receipt = parse_receipt(TIMED_OUT.read_text(encoding="utf-8"))
    assert receipt.result.value == "FAIL"
    revision = next(c for c in receipt.checks if c.name == "deployed_revision")
    assert revision.ok is True, "the lane WAS running the merge sha"
    ready_main = next(c for c in receipt.checks if c.name == "ready_main")
    assert ready_main.ok is False
    assert '"is_running":false' in ready_main.evidence.replace(" ", "")
    assert _budget_the_job_actually_had(TIMED_OUT) < 463, (
        "the grant must be under the fastest boot this lane has been observed "
        "to take, or the incident had some other cause"
    )


def test_the_unbound_receipt_recorded_greens_from_a_container_it_never_named() -> None:
    """The second defect, replayed over its own captured receipt.

    Convergence failed on an ancestor revision and every HTTP check passed in
    zero seconds -- the previous generation, still serving. The new binding
    check is what turns those greens into a stated failure.
    """
    receipt = parse_receipt(UNBOUND.read_text(encoding="utf-8"))
    assert receipt.result.value == "FAIL"
    revision = next(c for c in receipt.checks if c.name == "deployed_revision")
    assert revision.ok is False
    assert "does not contain merge sha" in revision.evidence
    http_checks = [c for c in receipt.checks if c.name != "deployed_revision"]
    assert all(c.ok for c in http_checks), "all four read green"
    assert all("ready after 0s" in c.evidence for c in http_checks)
    assert GENERATION_CHECK not in {c.name for c in receipt.checks}, (
        "the receipt this ticket replaces carried no binding at all"
    )

    # Driven through the REAL check: with no generation published, the greens
    # are refused rather than recorded as partial evidence for the sha.
    verdict = generation_check(None, _a_generation())
    assert verdict.ok is False
    assert "published no container generation" in verdict.evidence


def _a_generation() -> object:
    return parse_generation(
        {
            "Id": "b" * 64,
            "Name": "/omninode-runtime",
            "Image": "sha256:" + "c" * 64,
            "Config": {
                "Labels": {
                    "org.opencontainers.image.revision": "ba3f44b50495" + "0" * 28
                }
            },
        }
    )


def test_the_fixtures_are_the_bytes_the_incident_emitted() -> None:
    """A capture that no longer parses as the artifact it claims to be is a
    reconstruction, and the whole registry rule exists to tell those apart."""
    for path, sha in (
        (TIMED_OUT, "6e41f3e0c8b36073359407931eadc863ff07f289"),
        (UNBOUND, "4853e0e12b1937c81cc01ab2bca1da5d7ffc127f"),
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["sha"] == sha
        assert payload["lane"] == "compose-dev"
        assert payload["receipt_version"] == "lab_pass_receipt.v1"
