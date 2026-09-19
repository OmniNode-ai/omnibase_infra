# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""End-to-end coverage for the run files a DISPATCHED delegation owes (OMN-18569).

The unit module beside this one drives the unwrap. This one drives the whole
artifact-writing path a customer actually meets — three files on disk, with the
content and the route identity they are supposed to carry, plus the line
printed on stderr — from the verbatim receipt of a real dispatched run against
the ``.201`` dev lane.

That distinction is the whole reason this file exists rather than one more unit
test. The defect was never in a return value: the unwrap returned ``None``
correctly for the shape it was asked about, the writer returned early
correctly, and every unit test of the day passed. What was wrong was the
DIRECTORY — empty, for a run that exited 0 and answered correctly — and a
directory is only observable by writing to one.

The recording is ``tests/fixtures/delegation/omn18569/``; its README carries the
run, the redactions, and what it does not cover.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import _write_local_run_files
from omnibase_infra.cli.delegate_terminal_resolver import (
    DelegateTerminalUnresolvedError,
)
from omnibase_infra.cli.model_delegate_run_addressing import (
    ModelDelegateRunAddressing,
)
from omnibase_infra.cli.model_receipt_runtime_summary import (
    ModelReceiptRuntimeSummary,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.integration

# OMN-18810: this suite replays a REAL dispatched run, so it addresses the
# lane that run actually reached. An in-process value here would have the
# fixture contradict its own subject.
_ADDRESSING = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.DEPLOYED_LANE,
    bus="kafka",
    lane="dev",
    dispatch_target="onex.cmd.omnimarket.delegate-skill.v1 via REDACTED-LAB-BROKER",
)


_RECORDED_DISPATCHED_RECEIPT = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "delegation"
    / "omn18569"
    / "dispatched_envelope_carrier_receipt.json"
)

_PROMPT = (
    "List the first five prime numbers in ascending order, separated by "
    "commas, and nothing else."
)


def _recorded() -> ModelSkillResult[ModelReceiptRuntimeSummary]:
    return ModelSkillResult[ModelReceiptRuntimeSummary].model_validate(
        json.loads(_RECORDED_DISPATCHED_RECEIPT.read_text(encoding="utf-8"))
    )


def _write(state_root: Path) -> Path:
    receipt = _recorded()
    _write_local_run_files(
        receipt=receipt,
        state_root=state_root,
        addressing=_ADDRESSING,
        prompt=_PROMPT,
        task_type="summarization",
        task_type_resolution="explicit",
    )
    return state_root / "runs" / str(receipt.run_id)


class TestTheDirectoryIsNotEmpty:
    """The defect, stated as the thing a customer could see: an empty directory."""

    def test_all_three_files_exist(self, tmp_path: Path) -> None:
        run_dir = _write(tmp_path)
        for name in ("result.txt", "receipt.json", "run.json"):
            assert (run_dir / name).is_file(), f"{name} missing"

    def test_the_paths_reach_the_terminal_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Files nobody was told about are files nobody reads."""
        run_dir = _write(tmp_path)
        printed = capsys.readouterr().err
        assert "delegate artifacts: " in printed
        for name in ("result.txt", "receipt.json", "run.json"):
            assert str(run_dir / name) in printed


class TestTheFilesCarryWhatTheRunProduced:
    def test_the_answer_is_the_answer(self, tmp_path: Path) -> None:
        run_dir = _write(tmp_path)
        assert (run_dir / "result.txt").read_text(encoding="utf-8") == "2, 3, 5, 7, 11"

    def test_the_receipt_names_the_rung_that_answered(self, tmp_path: Path) -> None:
        """Route attribution survives the envelope unwrap, unchanged."""
        run_dir = _write(tmp_path)
        receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["model"] == "Qwen3.6-35B-A3B"
        assert receipt["routing_tier"] == "local"
        assert receipt["backend_id"] == "a3428e79-1694-5248-ab00-8e532196a515"
        assert receipt["correlation_id"] == "83aa8b6c-8189-49f0-953d-80c8f015ed0a"
        assert receipt["status"] == EnumSkillResultStatus.SUCCESS.value

    def test_the_run_metadata_separates_the_lane_from_the_rung(
        self, tmp_path: Path
    ) -> None:
        """OMN-18810: this test used to be the defect, in test form.

        It was named "carries the prompt and the lane" and asserted
        ``lane == "local"`` about a run dispatched to the ``dev`` lane. The
        two facts are now separate keys, and this fixture is the case that
        makes the difference visible: the rung that answered was local, and
        it answered on another machine.
        """
        run_dir = _write(tmp_path)
        run_json = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run_json["prompt"] == _PROMPT
        assert run_json["task_type"] == "summarization"
        assert run_json["routing_tier"] == "local"
        assert run_json["lane"] == "dev"
        assert run_json["locus"] == "deployed-lane"
        assert run_json["correlation_id"] == "83aa8b6c-8189-49f0-953d-80c8f015ed0a"


class TestAnEmptyDirectoryIsNowLoud:
    """The half that keeps the next fork from surviving a release.

    A third carrier shape will eventually arrive, the way the second one did.
    When it does, this path must stop and name what it could not find, rather
    than write nothing and exit 0.
    """

    @staticmethod
    def _receipt_without_a_terminal(
        *, workflow: str
    ) -> ModelSkillResult[ModelReceiptRuntimeSummary]:
        return ModelSkillResult[ModelReceiptRuntimeSummary](
            skill_name=cli_delegate.DELEGATE_NODE_NAME,
            node_name=cli_delegate.DELEGATE_NODE_NAME,
            status=EnumSkillResultStatus.SUCCESS,
            correlation_id=uuid.uuid4(),
            run_id=uuid.uuid4(),
            exit_code=0,
            duration_ms=9,
            result=ModelReceiptRuntimeSummary(
                workflow_result="completed",
                exit_code=0,
                workflow=workflow,
                terminal_payload={"status": "completed", "response": "an answer"},
                handler_result=None,
            ),
            result_model=(
                "omnibase_infra.cli.model_receipt_runtime_summary."
                "ModelReceiptRuntimeSummary"
            ),
            runtime_identity=collect_runtime_identity(config_source="test"),
        )

    def test_an_unresolvable_terminal_refuses_and_names_the_field(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(DelegateTerminalUnresolvedError) as raised:
            _write_local_run_files(
                receipt=self._receipt_without_a_terminal(
                    workflow=(
                        "/site-packages/omnimarket/nodes/"
                        "node_delegate_skill_orchestrator/contract.yaml"
                    )
                ),
                state_root=tmp_path,
                addressing=_ADDRESSING,
                prompt=_PROMPT,
                task_type="summarization",
                task_type_resolution="explicit",
            )
        assert "attempts" in str(raised.value)
        assert not (tmp_path / "runs").exists()

    def test_another_nodes_run_is_still_silent(self, tmp_path: Path) -> None:
        """The refusal did not widen. Negative control on the class above."""
        _write_local_run_files(
            receipt=self._receipt_without_a_terminal(
                workflow="/site-packages/omnimarket/nodes/node_gap_compute/contract.yaml"
            ),
            state_root=tmp_path,
            addressing=_ADDRESSING,
            prompt="proof",
            task_type="summarization",
            task_type_resolution="explicit",
        )
        assert not (tmp_path / "runs").exists()
