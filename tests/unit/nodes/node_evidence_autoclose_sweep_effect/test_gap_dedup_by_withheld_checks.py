# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The gap-comment fingerprint keys on WHICH checks withheld the flip.

OMN-16808 gave each (ticket, gap class) at most one live statement, keyed on
a digest of the verdict's COUNTERS. That key is wrong in a way OMN-17201
measured: between run 33991898262 (21:06Z) and run 33993316390 (21:35Z) two
of its checks moved from `failed` to `non_probative` with nothing about the
ticket changing. The counters moved (`3/30, 2 failed, 24 non-probative` ->
`3/30, 0 failed, 26 non-probative`), so the digest moved, so a second comment
was written asserting the same unmet criterion a second time.

The statement a gap comment makes is *these checks are what is standing
between this ticket and Done*. That is a SET OF CHECK IDS, not a tuple of
counts. Two verdicts that withhold the flip on the same checks are the same
statement however the verifier happened to grade them this rotation, and the
same statement is not repeated.

The contract version joins the key because the RULE that reads those checks
can change underneath a ticket: a closer that has learned to say something
new must be able to say it, and a version bump is exactly the event that
makes the old statement stale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _GAP_FINGERPRINT_CONTRACT_VERSION,
    HandlerEvidenceAutocloseSweep,
    _withheld_check_ids,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)

from .test_handler_evidence_autoclose_sweep import (
    FakeLinearClient,
    _issue,
    _make_dod_verify_fake,
    _make_gh_fake,
    _merged_pr,
    _request,
)
from .test_live_check_not_executed_hold import _check, _receipt

pytestmark = pytest.mark.unit


def test_the_withheld_set_is_every_check_that_is_not_verified() -> None:
    assert _withheld_check_ids(
        {
            "checks": [
                _check("dod-a", "verified"),
                _check("dod-b", "failed"),
                _check("dod-c", "non_probative"),
                _check("dod-d", "skipped"),
            ]
        }
    ) == ("dod-b", "dod-c", "dod-d")


def test_a_superseded_check_withholds_nothing() -> None:
    """OMN-15382/OMN-15390: superseded is resolved upstream, not outstanding."""
    assert (
        _withheld_check_ids(
            {"checks": [_check("dod-old", "superseded"), _check("dod-new", "verified")]}
        )
        == ()
    )


def test_the_set_is_order_independent() -> None:
    """Check order is a verifier implementation detail, not part of the claim."""
    forward = _withheld_check_ids(
        {"checks": [_check("dod-b", "failed"), _check("dod-a", "failed")]}
    )
    reverse = _withheld_check_ids(
        {"checks": [_check("dod-a", "failed"), _check("dod-b", "failed")]}
    )
    assert forward == reverse == ("dod-a", "dod-b")


@pytest.mark.parametrize(
    "verdict",
    [{}, {"checks": None}, {"checks": "not a list"}, {"checks": []}],
)
def test_an_unreadable_check_list_yields_no_ids(verdict: dict[str, object]) -> None:
    """The caller falls back to counters on an empty result — see below."""
    assert _withheld_check_ids(verdict) == ()


def test_the_pinned_contract_version_is_the_node_contract_version() -> None:
    """The version in the key is the node's own, read from the contract.

    A hand-typed constant that drifts from `contract.yaml` would silence a
    gap the bump was meant to refresh, so the two are pinned together.
    """
    contract = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_evidence_autoclose_sweep_effect"
        / "contract.yaml"
    )
    declared = yaml.safe_load(contract.read_text(encoding="utf-8"))["node_version"]
    assert declared == _GAP_FINGERPRINT_CONTRACT_VERSION


async def _sweep(
    payloads: list[dict[str, object]],
    *,
    ticket: str = "OMN-9999",
) -> tuple[list[Any], FakeLinearClient]:
    linear = FakeLinearClient(issues={ticket: _issue()})
    outcomes: list[Any] = []
    for index, payload in enumerate(payloads, start=1):
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=_make_gh_fake(
                companions=[_merged_pr(index, f"evidence({ticket}): x", ticket)],
                files_by_pr={index: [f"contracts/{ticket}.yaml"]},
            ),
            run_dod_verify_command=_make_dod_verify_fake({ticket: (payload, 0, "")}),
        )
        result = await handler.handle(_request(apply=True))
        outcomes.append(result.outcomes[0])
    return outcomes, linear


def _payload(
    failed: tuple[str, ...], non_probative: tuple[str, ...]
) -> dict[str, object]:
    checks = [_check("dod-green", "verified", proof_class="merge-state")]
    checks += [
        _check(name, "failed", message="binding produced no verdict") for name in failed
    ]
    checks += [
        _check(name, "non_probative", proof_class="surrogate") for name in non_probative
    ]
    return _receipt(checks=checks, verdict_status="failed", behavior_proving=0)


async def test_a_regrade_that_moves_no_check_out_of_the_set_is_a_duplicate() -> None:
    """The OMN-17201 shape: `failed` -> `non_probative`, same three checks.

    Under the counter-keyed digest this posted a second comment.
    """
    outcomes, linear = await _sweep(
        [
            _payload(failed=("dod-1", "dod-2"), non_probative=("dod-3",)),
            _payload(failed=("dod-1",), non_probative=("dod-2", "dod-3")),
        ]
    )

    assert outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
    assert (
        outcomes[1].decision == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
    )
    assert len(linear.comments) == 1


async def test_a_new_check_in_the_set_is_new_information_and_is_posted() -> None:
    """The positive control. Silence must be a property of sameness only."""
    outcomes, linear = await _sweep(
        [
            _payload(failed=("dod-1",), non_probative=("dod-3",)),
            _payload(failed=("dod-1", "dod-4"), non_probative=("dod-3",)),
        ]
    )

    assert outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
    assert outcomes[1].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
    assert len(linear.comments) == 2


async def test_a_check_leaving_the_set_is_also_new_information() -> None:
    """Progress is news too — a shrinking gap earns a fresh statement."""
    outcomes, linear = await _sweep(
        [
            _payload(failed=("dod-1", "dod-2"), non_probative=("dod-3",)),
            _payload(failed=("dod-1",), non_probative=("dod-3",)),
        ]
    )

    assert outcomes[1].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
    assert len(linear.comments) == 2


async def test_a_payload_with_no_readable_checks_still_dedups_on_counters() -> None:
    """Fail-safe, not fail-open.

    A verifier that emits counters and no per-check records is a shape the
    closer must still be idempotent against — otherwise this change would
    trade a duplicate-on-regrade for a duplicate-on-every-tick.
    """
    payload: dict[str, object] = _payload(failed=("dod-1",), non_probative=())
    result = payload["result"]
    assert isinstance(result, dict)
    verdict = result["terminal_payload"]
    assert isinstance(verdict, dict)
    verdict["checks"] = []

    outcomes, linear = await _sweep([payload, payload])

    assert outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
    assert (
        outcomes[1].decision == EnumEvidenceAutocloseDecision.SKIPPED_DUPLICATE_COMMENT
    )
    assert len(linear.comments) == 1
