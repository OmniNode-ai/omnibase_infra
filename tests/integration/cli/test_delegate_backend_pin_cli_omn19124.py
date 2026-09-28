# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end CLI coverage for the rung pin (OMN-19124).

The unit module beside this one drives ``_write_payload`` and the receipt
helpers directly. This one goes through ``click`` -- the real command, the
real option parsing, the real refusal path -- because the defect this ticket
closes was reachable ONLY from a command line, and its RED proof is a parser
fact rather than a function fact.

Measured on the .201 dev lane, 2026-09-22, against the released CLI:

    Error: No such option '--backend-id'.   (exit 2)

That is the whole defect. ``backend_id`` was a declared optional input on the
delegate node contract, the handler threaded it and the local dispatch port
honoured it -- and no caller could say it. A payload-writer test cannot
observe that, because the payload writer was never the thing that was
missing.
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
_GLM_BACKEND = "cloud-glm"
_STAND_IN_TASK_CLASS_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18305"
    / "task_class_contracts_vocabulary.yaml"
)


def _invoke(args: list[str]) -> Result:
    """Run the real command with option parsing, stopping before dispatch.

    Dispatch needs a co-installed omnimarket and a live model endpoint,
    neither of which belongs in this gate. Everything asserted here happens
    in FRONT of dispatch: option parsing and pin validation.
    """
    return CliRunner().invoke(delegate_command, args, catch_exceptions=False)


class TestTheRungPinIsReachableFromACommandLine:
    """The lane RED, as a test: the flag must PARSE."""

    def test_the_pin_flag_parses_rather_than_being_an_unknown_option(
        self, tmp_path: Path
    ) -> None:
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--backend-id",
                "cloud-glm",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "No such option" not in result.output, (
            "the released CLI answered exactly this on the dev lane; a pin "
            "nobody can type is the whole of OMN-19124"
        )
        assert "--backend-id" not in result.output or "Usage:" not in result.output

    def test_the_flag_is_listed_in_the_commands_own_help(self) -> None:
        """A caller finds a flag by reading --help, not by reading source."""
        result = _invoke(["--help"])
        assert result.exit_code == 0
        assert "--backend-id" in result.output

    def test_an_empty_pin_is_refused_at_the_flag_and_names_it(
        self, tmp_path: Path
    ) -> None:
        """A pin silently dropped would walk the ladder and answer anyway.

        The refusal must fire in FRONT of the omnimarket drift guard, which
        is the next thing on this command's path and which reports something
        entirely unrelated to the caller's flag value.
        """
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--backend-id",
                "   ",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert result.exit_code != 0
        assert "--backend-id was given an empty value" in result.output
        assert "omnimarket is NOT INSTALLED" not in result.output, (
            "the pin refusal must precede the drift guard, or the caller is "
            "told about a co-install when their flag value is the problem"
        )

    def test_the_unpinned_command_line_is_unchanged(self, tmp_path: Path) -> None:
        """AC2 at the parser. An absent flag must not become a usage error."""
        result = _invoke(
            [
                "summarise this",
                "--task-type",
                "summarization",
                "--state-root",
                str(tmp_path),
            ]
        )
        assert "No such option" not in result.output
        assert "--backend-id was given an empty value" not in result.output


def _attempt(
    *,
    backend_id: str,
    model_id: str,
    substituted_from_backend_id: str | None = None,
) -> dict[str, object]:
    return {
        "tier": "cheap_cloud",
        "backend_id": backend_id,
        "model_id": model_id,
        "quality_gate_passed": True,
        "quality_score": 1.0,
        "cost_usd": 0.0,
        "failure_class": None,
        "error_message": "",
        "acceptance_decision": "accept",
        "acceptance_reason": "quality_bar_met",
        "substituted_from_backend_id": substituted_from_backend_id,
    }


def _receipt(
    *,
    backend_id: str,
    model_id: str,
    substituted_from_backend_id: str | None = None,
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
            "task_type": "research",
            "model_name": model_id,
            "provider": "cheap_cloud",
            "response": "OK",
            "attempts": [
                _attempt(
                    backend_id=backend_id,
                    model_id=model_id,
                    substituted_from_backend_id=substituted_from_backend_id,
                )
            ],
            "terminal_failure_cause": None,
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _run_pinned_delegate(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: ModelSkillResult[dict[str, object]],
) -> tuple[int, str | None]:
    """Run the CLI entry with receipt mode reduced to its validator/writer seam."""
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
    validator_verdicts: list[str | None] = []
    receipt_mode_calls: list[dict[str, object]] = []

    def _no_drift(**_: object) -> None:
        return None

    def _resolve_contract(_: str) -> Path:
        return contract

    def _resolve_task_contract() -> Path:
        return _STAND_IN_TASK_CLASS_CONTRACT

    def _fake_receipt_mode(**kwargs: object) -> int:
        """Call the real validator, then the real writer, as receipt mode does.

        On a refusal the real receipt mode replaces the receipt with its own
        FAILED runtime-summary envelope before the writer runs; that envelope
        is receipt mode's concern, not this ticket's, so the writer is only
        handed the delegation receipt on the accepted path. The refused
        receipt's ``backend_pin_honoured: false`` is pinned by the unit module.
        """
        receipt_mode_calls.append(kwargs)
        receipt_validator = kwargs["receipt_validator"]
        receipt_callback = kwargs["receipt_callback"]
        assert callable(receipt_validator)
        assert callable(receipt_callback)
        verdict = receipt_validator(receipt)
        assert verdict is None or isinstance(verdict, str)
        validator_verdicts.append(verdict)
        if verdict is not None:
            return 1
        receipt_callback(receipt)
        return 0

    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", _no_drift)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", _resolve_contract)
    monkeypatch.setattr(
        cli_delegate, "resolve_task_class_contract_path", _resolve_task_contract
    )
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_receipt_mode)

    exit_code = run_delegate(
        prompt="probe",
        task_type="research",
        backend_id=_GLM_BACKEND,
        max_tokens=None,
        bus="inmemory",
        locus=EnumDelegateLocus.IN_PROCESS,
        state_root=tmp_path / "state",
        timeout=5,
        verbose=False,
        emit_socket=tmp_path / "emit.sock",
    )

    assert len(receipt_mode_calls) == 1
    assert validator_verdicts
    return exit_code, validator_verdicts[0]


def _written_receipt(
    state_root: Path, receipt: ModelSkillResult[dict[str, object]]
) -> dict[str, object]:
    written = json.loads(
        (state_root / "runs" / str(receipt.run_id) / "receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(written, dict)
    return written


class TestAc2PinHonoursTheLocalByokSubstitutionViaRunDelegate:
    """Go through ``run_delegate`` to prove request-to-validator-to-writer wiring, not only the helper."""

    def test_a_pin_answered_via_byok_substitution_is_honoured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        receipt = _receipt(
            backend_id="byok-glm",
            model_id="glm-5.3-flash",
            substituted_from_backend_id=_GLM_BACKEND,
        )

        exit_code, defect = _run_pinned_delegate(
            tmp_path=tmp_path, monkeypatch=monkeypatch, receipt=receipt
        )

        assert exit_code == 0
        assert defect is None
        written = _written_receipt(tmp_path / "state", receipt)
        assert written["requested_backend_id"] == _GLM_BACKEND
        assert written["backend_selection"] == "pinned"
        assert written["backend_pin_honoured"] is True

    def test_a_completed_run_on_another_backend_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        receipt = _receipt(backend_id="claude-sonnet", model_id="sonnet")

        exit_code, defect = _run_pinned_delegate(
            tmp_path=tmp_path, monkeypatch=monkeypatch, receipt=receipt
        )

        assert exit_code != 0
        assert defect is not None
        assert _GLM_BACKEND in defect
        assert "claude-sonnet" in defect
        assert not (tmp_path / "state" / "runs" / str(receipt.run_id)).exists()

    def test_a_substitution_of_a_different_backend_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        receipt = _receipt(
            backend_id="byok-openrouter",
            model_id="house/model:free",
            substituted_from_backend_id="openrouter-qwen3-coder-480b",
        )

        exit_code, defect = _run_pinned_delegate(
            tmp_path=tmp_path, monkeypatch=monkeypatch, receipt=receipt
        )

        assert exit_code != 0
        assert defect is not None
        assert _GLM_BACKEND in defect
        assert "byok-openrouter" in defect
        assert not (tmp_path / "state" / "runs" / str(receipt.run_id)).exists()
