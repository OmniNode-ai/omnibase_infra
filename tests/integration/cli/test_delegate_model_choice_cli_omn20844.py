# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for the customer's model choice (OMN-20844).

The unit module drives ``_write_payload`` and the receipt helpers directly.
This one goes through ``click`` and ``run_delegate`` -- the real option
parsing, the real usage refusal, and the real validator-to-writer wiring --
so a ``--model`` that parses but never reaches the validator or the receipt
fails here.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from click.testing import CliRunner, Result

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import delegate_command, run_delegate
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.integration

_RESULT_MODEL = (
    "omnimarket.models.delegation.wire."
    "model_delegate_skill_response.ModelDelegateSkillCompleted"
)
_BYOK = "byok-openrouter"
_CHOSEN = "google/gemini-2.5-flash-lite"
_OTHER = "google/gemma-4-31b-it:free"


def _invoke(args: list[str]) -> Result:
    """Run the real command with option parsing, stopping before dispatch."""
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


class TestTheModelFlagIsReachableFromACommandLine:
    def test_the_flag_is_listed_in_the_commands_own_help(self) -> None:
        result = _invoke(["--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_the_flag_parses_rather_than_being_an_unknown_option(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--backend-id",
                _BYOK,
                "--model",
                _CHOSEN,
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "No such option" not in result.output
        assert "--model was given an empty value" not in result.output

    def test_an_empty_model_is_refused_at_the_flag_and_names_it(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--model",
                "   ",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert "--model was given an empty value" in result.output


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
            "task_type": "research",
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


def _run_model_delegate(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: ModelSkillResult[dict[str, object]],
) -> tuple[int, str | None, dict[str, object]]:
    """Run ``run_delegate`` with receipt mode reduced to its validator/writer seam."""
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        """\
name: node_delegate_skill_orchestrator
terminal_event: onex.evt.omnimarket.delegate-skill-completed.v1
event_bus:
  publish_topics:
    - onex.evt.omnimarket.delegate-skill-completed.v1
  subscribe_topics:
    - onex.cmd.omnimarket.delegate-skill.v1
""",
        encoding="utf-8",
    )
    verdicts: list[str | None] = []
    calls: list[dict[str, object]] = []

    def _no_drift(**_: object) -> None:
        return None

    def _resolve_contract(_: str) -> Path:
        return contract

    def _fake_receipt_mode(**kwargs: object) -> int:
        calls.append(kwargs)
        receipt_validator = kwargs["receipt_validator"]
        receipt_callback = kwargs["receipt_callback"]
        assert callable(receipt_validator)
        assert callable(receipt_callback)
        verdict = receipt_validator(receipt)
        assert verdict is None or isinstance(verdict, str)
        verdicts.append(verdict)
        if verdict is not None:
            return 1
        receipt_callback(receipt)
        return 0

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", _no_drift)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", _resolve_contract)
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_receipt_mode)

    exit_code = run_delegate(
        prompt="probe",
        task_type="research",
        backend_id=_BYOK,
        model=_CHOSEN,
        max_tokens=None,
        bus="inmemory",
        locus=EnumDelegateLocus.IN_PROCESS,
        state_root=tmp_path / "state",
        timeout=5,
        verbose=False,
        emit_socket=tmp_path / "emit.sock",
    )

    assert len(calls) == 1
    assert verdicts
    return exit_code, verdicts[0], calls[0]


class TestTheChosenModelReachesTheValidatorAndTheReceipt:
    def test_the_chosen_model_answering_is_honoured_on_the_receipt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        receipt = _receipt(_CHOSEN)

        exit_code, defect, _ = _run_model_delegate(
            tmp_path=tmp_path, monkeypatch=monkeypatch, receipt=receipt
        )

        assert exit_code == 0
        assert defect is None
        written = json.loads(
            (
                tmp_path / "state" / "runs" / str(receipt.run_id) / "receipt.json"
            ).read_text(encoding="utf-8")
        )
        assert written["requested_model"] == _CHOSEN
        assert written["model_choice_honoured"] is True

    def test_another_model_answering_is_refused_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        receipt = _receipt(_OTHER)

        exit_code, defect, _ = _run_model_delegate(
            tmp_path=tmp_path, monkeypatch=monkeypatch, receipt=receipt
        )

        assert exit_code != 0
        assert defect is not None
        assert _CHOSEN in defect
        assert _OTHER in defect

    def test_the_chosen_model_is_in_the_dispatched_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _, _, call = _run_model_delegate(
            tmp_path=tmp_path, monkeypatch=monkeypatch, receipt=_receipt(_CHOSEN)
        )

        payload_paths = [
            value
            for value in call.values()
            if isinstance(value, Path) and value.suffix == ".json"
        ]
        assert payload_paths, sorted(call)
        payloads = [json.loads(p.read_text(encoding="utf-8")) for p in payload_paths]
        assert any(isinstance(p, dict) and p.get("model") == _CHOSEN for p in payloads)
