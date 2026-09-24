# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A caller of ``onex delegate`` can name the rung, and the receipt says so (OMN-19124).

THE GAP THESE PIN. ``backend_id`` has been a declared optional input on the
delegate node contract since OMN-15156
(``omnimarket/src/omnimarket/nodes/node_delegate_skill_orchestrator/contract.yaml``,
the ``backend_id`` entry: "Optional explicit backend pin... None resolves the
backend via the normal cheapest-first tier_order selection"). The handler
threads it and both dispatch ports accept it. The chain was built end to end
and stopped one layer short of a caller: this CLI declared no flag that
reaches the field, so the ONLY thing a caller could choose was a task class,
and the class decided the rung by cheapest-first walk plus quality-gate
escalation.

Measured consequence, 2026-09-21: two dev-lane delegations (correlations
``f0575167-a1bb-4513-be2d-1a8122929b6d`` and
``a24b7f3c-286f-4e57-a2d7-2e48cf5a9a90``) both terminated on ``local``
Qwen3.8-27B at quality 1.0 against a 0.8 bar with ``escalation_count: 0``.
That is the ladder working as designed, and it is exactly why escalation
cannot be used to reach a specific rung on demand — a rung that is never
walked to is a rung that cannot be exercised at all.

THE SEMANTICS THIS MODULE CHOOSES AND ASSERTS (AC3). The ticket names two
candidate semantics for the flag and requires that one be asserted rather
than inherited from the escalation loop's incidental behaviour. The port's
own note is the reason the incidental behaviour is not acceptable as an
unstated default: a transport failure on the pinned backend excludes the
pinned backend's WHOLE TIER and re-resolves the next hop through the normal
``tier_order`` (``port_local_delegation_dispatch.py``,
``_resolve_initial_backend`` "Escalation-interaction note"). So a pin that is
silent about failure hands the caller an answer from a DIFFERENT, typically
more expensive, rung while the caller believes it ran where it asked.

The chosen semantics is **PIN-OR-REFUSE, at the terminal**: the pin still
only selects the initial attempt (that is the port's contract and this CLI
does not redesign it), but a completed delegation whose ACCEPTED attempt did
not run on the pinned backend is a REFUSAL here — non-zero exit, the defect
named on stderr, and the receipt still written so the run is diagnosable.
The honest limit is stated rather than implied: this does not PREVENT the
escalation spend, it makes it impossible for that spend to be mistaken for
the pinned run. Preventing it requires a port change in ``omnimarket`` and is
not in this change.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli.cli_delegate import (
    _delegate_receipt_evidence_error,
    _validate_backend_pin,
    _write_local_run_files,
    _write_payload,
    delegate_command,
)
from omnibase_infra.cli.model_delegate_run_addressing import ModelDelegateRunAddressing
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.unit

_RESULT_MODEL = (
    "omnimarket.models.delegation.wire."
    "model_delegate_skill_response.ModelDelegateSkillCompleted"
)

_IN_PROCESS = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.IN_PROCESS,
    bus="inmemory",
)

# The cheap_cloud rung this ticket was filed to make reachable. Spelled from
# the routing contract's own declaration (routing_tiers.yaml declares
# ``backend_id: cloud-glm`` for ``id: glm-5.3-flash``) rather than invented
# here, and never resolved by this CLI — an unresolvable pin is the routing
# authority's refusal to raise, not this layer's to pre-empt.
_GLM_BACKEND = "cloud-glm"
_GLM_MODEL = "glm-5.3-flash"


def _attempt(*, tier: str, backend_id: str, model_id: str) -> dict[str, object]:
    return {
        "tier": tier,
        "backend_id": backend_id,
        "model_id": model_id,
        "quality_gate_passed": True,
        "quality_score": 1.0,
        "cost_usd": 0.0,
        "failure_class": None,
        "error_message": "",
        "acceptance_decision": "accept",
        "acceptance_reason": "quality_bar_met",
    }


def _receipt(
    *,
    tier: str = "local",
    backend_id: str = "local-coder",
    model_id: str = "Qwen3.8-27B",
) -> ModelSkillResult[dict[str, object]]:
    return ModelSkillResult(
        skill_name="node_delegate_skill_orchestrator",
        node_name="node_delegate_skill_orchestrator",
        status=EnumSkillResultStatus.SUCCESS,
        correlation_id=uuid4(),
        run_id=uuid4(),
        exit_code=0,
        duration_ms=1200,
        result={
            "status": "completed",
            "task_type": "summarization",
            "model_name": model_id,
            "provider": tier,
            "response": "OK",
            "attempts": [_attempt(tier=tier, backend_id=backend_id, model_id=model_id)],
            "terminal_failure_cause": None,
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _write(
    tmp_path: Path,
    receipt: ModelSkillResult[dict[str, object]],
    *,
    requested_backend_id: str | None,
) -> dict[str, object]:
    _write_local_run_files(
        receipt=receipt,
        state_root=tmp_path,
        prompt="Reply with exactly: OK",
        task_type="summarization",
        task_type_resolution="explicit",
        addressing=_IN_PROCESS,
        requested_backend_id=requested_backend_id,
    )
    run_dir = tmp_path / "runs" / str(receipt.run_id)
    return json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))


def _payload(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    path = _write_payload(
        prompt="Reply with exactly: OK",
        task_type="summarization",
        source="claude-code",
        state_root=tmp_path,
        run_id=str(uuid4()),
        correlation_id=uuid4(),
        max_tokens=None,
        **kwargs,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


class TestAc1ACallerCanNameTheRung:
    """AC1. The flag exists, and it reaches the field the contract declares."""

    def test_the_command_declares_a_backend_pin_option(self) -> None:
        """Read from the parser's OWN option strings, not from the help text.

        The same shape the prod-promotion gate's option tests use: a claim
        about what a CLI accepts is only falsifiable against the declared
        parameters, because help prose can name a flag that does not parse
        and a flag can parse while naming itself nothing.
        """
        declared = {
            opt
            for param in delegate_command.params
            for opt in getattr(param, "opts", ())
        }
        assert "--backend-id" in declared

    def test_the_pin_is_written_into_the_payload_under_the_contract_field(
        self, tmp_path: Path
    ) -> None:
        payload = _payload(tmp_path, backend_id=_GLM_BACKEND)
        assert payload["backend_id"] == _GLM_BACKEND


class TestAc2OmittingTheFlagChangesNothing:
    """AC2. The cheapest-first walk is the behaviour of an unpinned request."""

    def test_no_pin_writes_no_backend_key_at_all(self, tmp_path: Path) -> None:
        """Omitted, not null.

        ``ModelDelegateSkillRequest`` declares ``extra="forbid"``, and every
        optional field this payload builder writes is omitted rather than
        nulled for exactly that reason. A ``backend_id: null`` would be a new
        key on every existing caller's payload, which is a shape change even
        though the value means "unpinned".
        """
        payload = _payload(tmp_path)
        assert "backend_id" not in payload

    def test_an_unpinned_receipt_is_never_refused_for_its_rung(
        self, tmp_path: Path
    ) -> None:
        receipt = _receipt()
        assert (
            _delegate_receipt_evidence_error(receipt, requested_backend_id=None) is None
        )


class TestAc3PinOrRefuse:
    """AC3. The pin-failure semantics is asserted here, not inherited."""

    def test_a_completed_run_on_another_rung_is_refused_by_name(self) -> None:
        """The escalated answer is not silently returned as the pinned one.

        This is the failure the port's escalation note predicts: the pinned
        backend's transport fails, its WHOLE tier is excluded, and the run
        completes on a different — typically more expensive — rung.
        """
        receipt = _receipt(tier="claude", backend_id="claude-sonnet", model_id="sonnet")
        defect = _delegate_receipt_evidence_error(
            receipt, requested_backend_id=_GLM_BACKEND
        )
        assert defect is not None
        assert _GLM_BACKEND in defect
        assert "claude-sonnet" in defect

    def test_a_completed_run_on_the_pinned_rung_is_accepted(self) -> None:
        receipt = _receipt(
            tier="cheap_cloud", backend_id=_GLM_BACKEND, model_id=_GLM_MODEL
        )
        assert (
            _delegate_receipt_evidence_error(receipt, requested_backend_id=_GLM_BACKEND)
            is None
        )

    def test_a_terminally_failed_run_is_not_re_reported_as_a_pin_violation(
        self,
    ) -> None:
        """A run that reached no rung already reports its own cause.

        Turning "nothing answered" into "your pin was violated" would name
        the wrong defect, and the unattributed path exists precisely to say
        what actually failed.
        """
        receipt = _receipt()
        result = receipt.result
        assert isinstance(result, dict)
        result["status"] = "failed"
        result["attempts"] = []
        result["terminal_failure_cause"] = "quality_bar_missed"
        defect = _delegate_receipt_evidence_error(
            receipt, requested_backend_id=_GLM_BACKEND
        )
        assert defect is None or "pin" not in defect.lower()


class TestAc4TheReceiptSaysTheRungWasChosen:
    """AC4. A pinned run is distinguishable from one that walked to the rung."""

    def test_a_pinned_receipt_records_the_pin_and_how_the_rung_was_selected(
        self, tmp_path: Path
    ) -> None:
        receipt = _receipt(
            tier="cheap_cloud", backend_id=_GLM_BACKEND, model_id=_GLM_MODEL
        )
        written = _write(tmp_path, receipt, requested_backend_id=_GLM_BACKEND)
        assert written["requested_backend_id"] == _GLM_BACKEND
        assert written["backend_selection"] == "pinned"
        assert written["backend_pin_honoured"] is True

    def test_an_unpinned_receipt_says_cheapest_first_and_claims_no_pin(
        self, tmp_path: Path
    ) -> None:
        """The positive control. Without it, "pinned" is unfalsifiable."""
        receipt = _receipt()
        written = _write(tmp_path, receipt, requested_backend_id=None)
        assert written["requested_backend_id"] is None
        assert written["backend_selection"] == "cheapest_first"
        assert written["backend_pin_honoured"] is None

    def test_a_violated_pin_is_recorded_on_the_receipt_not_only_on_stderr(
        self, tmp_path: Path
    ) -> None:
        receipt = _receipt(tier="claude", backend_id="claude-sonnet", model_id="sonnet")
        written = _write(tmp_path, receipt, requested_backend_id=_GLM_BACKEND)
        assert written["requested_backend_id"] == _GLM_BACKEND
        assert written["backend_pin_honoured"] is False
        assert written["backend_id"] == "claude-sonnet"


class TestAPinIsNeverSilentlyDropped:
    """A pin the caller believes they set must never become an unpinned run."""

    def test_an_empty_pin_is_a_usage_error_not_a_silent_unpin(self) -> None:
        with pytest.raises(ValueError, match="--backend-id"):
            _validate_backend_pin("   ")

    def test_surrounding_whitespace_is_normalised_rather_than_refused(self) -> None:
        assert _validate_backend_pin(f"  {_GLM_BACKEND}\n") == _GLM_BACKEND

    def test_an_absent_flag_stays_absent(self) -> None:
        assert _validate_backend_pin(None) is None

    def test_the_cli_does_not_keep_its_own_list_of_backend_ids(self) -> None:
        """The routing contract is the vocabulary; this layer holds no copy.

        A CLI-side allowlist would go stale against ``routing_tiers.yaml`` and
        start refusing rungs the contract declares, so an unknown id must
        reach the routing authority and be refused THERE, by name.
        """
        assert _validate_backend_pin("a-backend-this-cli-has-never-heard-of") == (
            "a-backend-this-cli-has-never-heard-of"
        )


class TestAPinNothingReachedIsNotAViolatedPin:
    """Found on the lab, not in this file, and pinned here afterwards.

    OMN-19124 lab arm B (dev lane, 2026-09-22): a pinned run was refused by
    the customer-key terminus guard BEFORE any rung was dispatched, and the
    receipt recorded ``backend_pin_honoured: false``. That reads as "we ran
    somewhere else", which is the precise misreading this key exists to
    prevent — nothing ran anywhere, and the receipt's own
    ``route_unattributed`` says so. ``None`` is the only honest third value,
    and this class is the falsifier the first pass did not have.
    """

    def test_a_run_that_reached_no_rung_reports_the_pin_as_unresolved(
        self, tmp_path: Path
    ) -> None:
        receipt = _receipt()
        result = receipt.result
        assert isinstance(result, dict)
        result["status"] = "failed"
        result["attempts"] = []
        result["terminal_failure_cause"] = "provider_error"
        written = _write(tmp_path, receipt, requested_backend_id=_GLM_BACKEND)
        assert written["requested_backend_id"] == _GLM_BACKEND
        assert written["backend_selection"] == "pinned"
        assert written["backend_pin_honoured"] is None
        assert written["route_attributed"] is False
