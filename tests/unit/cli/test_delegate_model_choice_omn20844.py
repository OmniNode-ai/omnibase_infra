# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate --model`` names the model the customer's own key runs (OMN-20844).

The delegate node's request declares an optional ``model`` (omnimarket
OMN-20844): the in-process port runs the customer's own BYOK route on it for
this call and refuses it for any other route. This CLI is the caller that
reaches the field. The semantics follow the backend pin's (OMN-19124):

* the flag is written into the payload under the contract field, and omitted
  (never nulled) when unset, because the request forbids extra keys;
* MODEL-OR-REFUSE at the terminal: a completed run whose accepted attempt ran
  another model is a refusal naming both, with the receipt still written;
* the receipt records the model asked for and whether it was honoured.
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
    _validate_model_choice,
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
_BYOK = "byok-openrouter"
_CHOSEN = "google/gemini-2.5-flash-lite"


def _receipt(model_id: str) -> ModelSkillResult[dict[str, object]]:
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
            "task_type": "document",
            "model_name": model_id,
            "provider": "cheap_cloud",
            "response": "OK",
            "attempts": [
                {
                    "tier": "cheap_cloud",
                    "backend_id": _BYOK,
                    "model_id": model_id,
                    "quality_gate_passed": True,
                    "quality_score": 1.0,
                    "cost_usd": 0.0,
                    "failure_class": None,
                    "error_message": "",
                    "acceptance_decision": "accept",
                    "acceptance_reason": "quality_bar_met",
                    "substituted_from_backend_id": None,
                }
            ],
            "terminal_failure_cause": None,
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _payload(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    path = _write_payload(
        prompt="Reply with exactly: OK",
        task_type="document",
        source="claude-code",
        state_root=tmp_path,
        run_id=uuid4(),
        correlation_id=uuid4(),
        max_tokens=None,
        **kwargs,
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write(
    tmp_path: Path, model_id: str, requested_model: str | None
) -> dict[str, object]:
    receipt = _receipt(model_id)
    _write_local_run_files(
        receipt=receipt,
        state_root=tmp_path,
        prompt="Reply with exactly: OK",
        task_type="document",
        task_type_resolution="explicit",
        addressing=_IN_PROCESS,
        requested_backend_id=_BYOK,
        requested_model=requested_model,
    )
    run_dir = tmp_path / "runs" / str(receipt.run_id)
    return json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))


class TestTheFlag:
    def test_the_command_declares_a_model_option(self) -> None:
        declared = {
            opt
            for param in delegate_command.params
            for opt in getattr(param, "opts", ())
        }
        assert "--model" in declared

    def test_the_model_is_written_into_the_payload_under_the_contract_field(
        self, tmp_path: Path
    ) -> None:
        payload = _payload(tmp_path, backend_id=_BYOK, model=_CHOSEN)
        assert payload["model"] == _CHOSEN

    def test_no_model_writes_no_model_key_at_all(self, tmp_path: Path) -> None:
        assert "model" not in _payload(tmp_path)

    def test_an_empty_model_is_a_usage_error_not_a_silent_default(self) -> None:
        with pytest.raises(ValueError, match="--model"):
            _validate_model_choice("  ")

    def test_surrounding_whitespace_is_normalised(self) -> None:
        assert _validate_model_choice(f" {_CHOSEN}\n") == _CHOSEN
        assert _validate_model_choice(None) is None


class TestModelOrRefuse:
    def test_a_completed_run_on_another_model_is_refused_by_name(self) -> None:
        defect = _delegate_receipt_evidence_error(
            _receipt("google/gemma-4-31b-it:free"),
            requested_backend_id=_BYOK,
            requested_model=_CHOSEN,
        )
        assert defect is not None
        assert _CHOSEN in defect
        assert "google/gemma-4-31b-it:free" in defect

    def test_a_completed_run_on_the_chosen_model_is_accepted(self) -> None:
        assert (
            _delegate_receipt_evidence_error(
                _receipt(_CHOSEN), requested_backend_id=_BYOK, requested_model=_CHOSEN
            )
            is None
        )


class TestTheReceiptNamesTheChoice:
    def test_the_receipt_names_the_chosen_model_and_that_it_was_honoured(
        self, tmp_path: Path
    ) -> None:
        written = _write(tmp_path, _CHOSEN, _CHOSEN)
        assert written["model"] == _CHOSEN
        assert written["requested_model"] == _CHOSEN
        assert written["model_choice_honoured"] is True

    def test_a_run_with_no_model_asked_for_claims_no_choice(
        self, tmp_path: Path
    ) -> None:
        written = _write(tmp_path, _CHOSEN, None)
        assert written["requested_model"] is None
        assert written["model_choice_honoured"] is None

    def test_a_violated_choice_is_recorded_on_the_receipt(self, tmp_path: Path) -> None:
        written = _write(tmp_path, "google/gemma-4-31b-it:free", _CHOSEN)
        assert written["model_choice_honoured"] is False
