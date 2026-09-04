# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16961 — ``onex skill dod_verify`` prints TWO receipt arms, not one.

``receipt_mode._run_and_emit`` branches on ``status.is_success_like and
handler_result is not None``:

* success-like  -> ``result`` IS the handler's own model, flat.
  ``result_model = omnimarket...ModelDodVerifyState``; there is no
  ``terminal_payload`` key anywhere in the receipt.
* everything else -> ``result`` is a ``ModelReceiptRuntimeSummary`` and the
  verdict is nested at ``result.terminal_payload``.

The sweep hardcoded the SECOND arm (OMN-16736's constant was derived from a
single ``status: failed`` capture). Because dod_verify's workflow result is
success-like exactly when its verdict is ``verified``, the arm the sweep
could not read was precisely the flip-eligible one: a ticket that satisfies
every OMN-16821 conjunct was recorded ``error_verify_unparseable`` at
``exit_code=0``, and ``tickets_flipped`` was structurally pinned to 0.

Both fixtures here are VERBATIM local captures taken 2026-08-29 against
``onex_change_control@origin/dev`` with the same
``${DISPATCH_VENV}/bin/onex skill dod_verify <ticket>`` argv the sweep
spawns — one ticket from run 33258391128's errored pool (OMN-16831) and one
from its parsed pool (OMN-16865). They are the discriminating measurement
for AC1: the split is deterministic and reproduces single-process with no
concurrency, which refutes the load/timeout direction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    _DOD_VERIFY_STATE_RESULT_MODEL,
    _RECEIPT_SUMMARY_RESULT_MODEL,
    HandlerEvidenceAutocloseSweep,
    _extract_dod_verify_verdict,
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

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "omn16961"


def _captured(name: str) -> dict[str, object]:
    payload = json.loads((_FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _success_arm() -> dict[str, object]:
    return _captured("dod-verify-omn16831.success-arm.skill-result.json.captured")


def _summary_arm() -> dict[str, object]:
    return _captured("dod-verify-omn16865.summary-arm.skill-result.json.captured")


@pytest.mark.unit
class TestTheTwoReceiptArmsAreReal:
    """Pin the shape difference itself, before any handler behaviour."""

    def test_a_verified_verdict_prints_the_flat_success_arm(self):
        captured = _success_arm()
        assert captured["status"] == "success"
        assert captured["exit_code"] == 0
        assert captured["result_model"] == _DOD_VERIFY_STATE_RESULT_MODEL
        result = captured["result"]
        assert isinstance(result, dict)
        # The whole defect in two assertions: the verdict is RIGHT HERE, and
        # the key the sweep insisted on does not exist at all.
        assert "terminal_payload" not in result
        assert result["total_checks"] == 32
        assert result["status"] == "verified"

    def test_a_non_verified_verdict_prints_the_nested_summary_arm(self):
        captured = _summary_arm()
        assert captured["status"] == "failed"
        assert captured["exit_code"] == 1
        assert captured["result_model"] == _RECEIPT_SUMMARY_RESULT_MODEL
        result = captured["result"]
        assert isinstance(result, dict)
        assert "total_checks" not in result
        terminal = result["terminal_payload"]
        assert isinstance(terminal, dict)
        assert terminal["total_checks"] == 8

    def test_the_two_arms_are_discriminated_by_result_model_alone(self):
        """`result_model` is the receipt's own declared type tag.

        Reading it is using the contract, not sniffing the shape. Nothing
        else in the envelope separates the arms without guessing.
        """
        assert _success_arm()["result_model"] != _summary_arm()["result_model"]


@pytest.mark.unit
class TestVerdictExtraction:
    """`_extract_dod_verify_verdict` speaks both declared arms, and nothing else."""

    def test_success_arm_yields_the_flat_verdict(self):
        verdict, refusal = _extract_dod_verify_verdict(_success_arm())
        assert refusal == ""
        assert verdict is not None
        assert verdict["total_checks"] == 32
        assert verdict["verified_count"] == 6
        assert verdict["non_probative_count"] == 26

    def test_summary_arm_yields_the_nested_verdict(self):
        verdict, refusal = _extract_dod_verify_verdict(_summary_arm())
        assert refusal == ""
        assert verdict is not None
        assert verdict["total_checks"] == 8
        assert verdict["status"] == "skipped"

    def test_an_undeclared_result_model_is_refused_by_name(self):
        """No `result_model` means the receipt did not declare its own shape."""
        verdict, refusal = _extract_dod_verify_verdict(
            {"status": "success", "result": {"total_checks": 3}}
        )
        assert verdict is None
        assert "result_model" in refusal

    def test_an_unrecognised_result_model_is_refused_and_named(self):
        """Fail LOUD on drift — never fall back to shape-sniffing.

        AC4: a receipt whose shape this reader does not know is an
        unreachable verdict, and unreachable verdicts never count as proof.
        """
        receipt = {
            "status": "success",
            "result": {"total_checks": 3, "status": "verified"},
            "result_model": "some.other.Model",
        }
        verdict, refusal = _extract_dod_verify_verdict(receipt)
        assert verdict is None
        assert "some.other.Model" in refusal

    def test_a_summary_arm_with_no_terminal_payload_is_refused_specifically(self):
        """A genuinely unreached verdict keeps failing closed, with its own cause."""
        receipt = {
            "status": "failed",
            "result": {"workflow_result": "failed", "error": "boom"},
            "result_model": _RECEIPT_SUMMARY_RESULT_MODEL,
        }
        verdict, refusal = _extract_dod_verify_verdict(receipt)
        assert verdict is None
        assert "terminal_payload" in refusal

    def test_a_declared_arm_missing_total_checks_is_refused(self):
        receipt = {
            "status": "success",
            "result": {"status": "verified"},
            "result_model": _DOD_VERIFY_STATE_RESULT_MODEL,
        }
        verdict, refusal = _extract_dod_verify_verdict(receipt)
        assert verdict is None
        assert "total_checks" in refusal


@pytest.mark.unit
class TestTheSweepReadsTheSuccessArm:
    """End to end: the arm that was unreadable now reaches a real decision."""

    async def _run(self, captured: dict[str, object], exit_code: int):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(7543, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={7543: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake({"OMN-9999": (captured, exit_code, "")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        return await handler.handle(_request(apply=False)), linear

    async def test_the_errored_pool_row_now_carries_the_numbers_it_measured(self):
        """OMN-16831's real receipt: 32 checks, 6 verified, 0 failed, 26 non-probative.

        Before this change the same bytes produced
        ``error_verify_unparseable`` with every counter at 0.
        """
        result, _linear = await self._run(_success_arm(), 0)
        outcome = result.outcomes[0]

        assert (
            outcome.decision != EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE
        )
        assert result.tickets_errored == 0
        assert outcome.dod_verify_total_checks == 32
        assert outcome.dod_verify_verified_count == 6
        assert outcome.dod_verify_failed_count == 0
        assert outcome.dod_verify_non_probative_count == 26

    async def test_the_summary_arm_still_reads_exactly_as_before(self):
        """No regression on the nine tickets that already parsed."""
        result, _linear = await self._run(_summary_arm(), 1)
        outcome = result.outcomes[0]

        assert outcome.decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert outcome.dod_verify_total_checks == 8
        assert outcome.dod_verify_verified_count == 0

    async def test_an_unreachable_verdict_is_still_never_proof(self):
        """AC4 carried forward — the guard is not weakened, only made specific."""
        receipt = {
            "status": "success",
            "result": {"workflow_result": "completed"},
            "result_model": _RECEIPT_SUMMARY_RESULT_MODEL,
        }
        result, linear = await self._run(receipt, 0)
        outcome = result.outcomes[0]

        assert (
            outcome.decision == EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE
        )
        # AC2: the specific cause, not the generic "no verdict was reached".
        assert "terminal_payload" in outcome.reason
        assert _RECEIPT_SUMMARY_RESULT_MODEL in outcome.reason
        assert result.tickets_errored == 1
        assert linear.state_updates == []
        assert linear.comments == []
