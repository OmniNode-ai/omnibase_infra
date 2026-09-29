# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The two files a delegation writes must say WHERE the run was addressed.

OMN-18810. On 2026-09-19 a lane reported that ``onex delegate ... --bus kafka
--lane dev --locus deployed-lane`` had accepted its routing flags and silently
ignored them: exit 0, and a ``run.json`` reading ``lane: "local"``.

The flags were honoured. That run published to the dev lane's broker and
hosted nothing — its own receipt says ``handler_locus: "dispatched"`` and
names the broker and the live consumer groups it found there. What produced
the false report is that ``run.json``'s ``lane`` key never held a lane. It
held the accepted attempt's ROUTING TIER, under a name identical to the flag
that means something else, in a file that recorded no transport, no lane and
no locus at all.

These tests pin the four addressing keys onto both written files and pin the
tier back to its own name. The positive control matters as much as the
dispatched case: a plain local run must still record ``local`` as its tier,
under ``routing_tier``, and must be distinguishable from a dispatched one
without opening the nested envelope.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_infra.cli.cli_delegate import _write_local_run_files
from omnibase_infra.cli.model_delegate_run_addressing import ModelDelegateRunAddressing
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.unit

_DISPATCHED = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.DEPLOYED_LANE,
    bus="kafka",
    lane="dev",
    dispatch_target="onex.cmd.omnimarket.delegate-skill.v1 via lab-broker.invalid:19092",
)

_IN_PROCESS = ModelDelegateRunAddressing(
    locus=EnumDelegateLocus.IN_PROCESS,
    bus="inmemory",
)

_RESULT_MODEL = (
    "omnimarket.models.delegation.wire."
    "model_delegate_skill_response.ModelDelegateSkillCompleted"
)


def _attempt(*, accepted: bool) -> dict[str, object]:
    return {
        "tier": "local",
        "backend_id": "local-coder",
        "model_id": "Qwen3.8-27B",
        "quality_gate_passed": accepted,
        "quality_score": 1.0 if accepted else 0.1,
        "cost_usd": 0.0,
        "failure_class": None if accepted else "quality_bar_missed",
        "error_message": "",
        "acceptance_decision": "accept" if accepted else "climb",
        "acceptance_reason": "quality_bar_met" if accepted else "below_bar",
    }


def _receipt(*, accepted: bool = True) -> ModelSkillResult[dict[str, object]]:
    """A delegation terminal on the bare carrier shape.

    The carrier shape is not what these tests are about — the addressing keys
    are written by the same code on both shapes — so the simpler one is used
    and the envelope-carrier shape stays covered by its own OMN-18569 suite.
    """
    return ModelSkillResult(
        skill_name="node_delegate_skill_orchestrator",
        node_name="node_delegate_skill_orchestrator",
        status=(
            EnumSkillResultStatus.SUCCESS if accepted else EnumSkillResultStatus.FAILED
        ),
        correlation_id=uuid4(),
        run_id=uuid4(),
        exit_code=0 if accepted else 1,
        duration_ms=44177,
        result={
            "status": "completed" if accepted else "failed",
            "task_type": "summarization",
            "model_name": "Qwen3.8-27B",
            "provider": "local",
            "response": "OK" if accepted else "",
            "attempts": [_attempt(accepted=accepted)],
            "terminal_failure_cause": None if accepted else "quality_bar_missed",
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


def _write(
    tmp_path: Path,
    addressing: ModelDelegateRunAddressing,
    *,
    accepted: bool = True,
) -> tuple[Path, ModelSkillResult[dict[str, object]]]:
    receipt = _receipt(accepted=accepted)
    _write_local_run_files(
        receipt=receipt,
        state_root=tmp_path,
        prompt="Reply with exactly: OK",
        task_type="summarization",
        task_type_resolution="explicit",
        addressing=addressing,
    )
    return tmp_path / "runs" / str(receipt.run_id), receipt


def _run_json(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def _receipt_json(run_dir: Path) -> dict[str, object]:
    return json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))


class TestADispatchedRunSaysSoInItsOwnFiles:
    """AC1. The reported defect, as an assertion.

    Every field here is one the 2026-09-19 reporting lane looked for and did
    not find. If this class passes, that report cannot be written again from
    these two files.
    """

    def test_run_json_names_the_locus_the_transport_and_the_lane(
        self, tmp_path: Path
    ) -> None:
        run_dir, _ = _write(tmp_path, _DISPATCHED)
        run_json = _run_json(run_dir)
        assert run_json["locus"] == "deployed-lane"
        assert run_json["bus"] == "kafka"
        assert run_json["lane"] == "dev"

    def test_run_json_names_the_topic_and_broker_the_command_went_to(
        self, tmp_path: Path
    ) -> None:
        run_dir, _ = _write(tmp_path, _DISPATCHED)
        assert _run_json(run_dir)["dispatch_target"] == (
            "onex.cmd.omnimarket.delegate-skill.v1 via lab-broker.invalid:19092"
        )

    def test_receipt_json_carries_the_same_four_keys(self, tmp_path: Path) -> None:
        """One spelling in both files, so neither can be read against the other."""
        run_dir, _ = _write(tmp_path, _DISPATCHED)
        receipt_json = _receipt_json(run_dir)
        assert receipt_json["locus"] == "deployed-lane"
        assert receipt_json["bus"] == "kafka"
        assert receipt_json["lane"] == "dev"
        assert receipt_json["dispatch_target"] == (
            "onex.cmd.omnimarket.delegate-skill.v1 via lab-broker.invalid:19092"
        )

    def test_a_dispatched_run_is_distinguishable_from_a_local_one(
        self, tmp_path: Path
    ) -> None:
        """The whole point, stated as one comparison.

        Both runs below answered on the local tier at zero cost, which is why
        the tier was never going to separate them. The locus does.
        """
        dispatched, _ = _write(tmp_path / "dispatched", _DISPATCHED)
        in_process, _ = _write(tmp_path / "in_process", _IN_PROCESS)
        assert _run_json(dispatched)["routing_tier"] == "local"
        assert _run_json(in_process)["routing_tier"] == "local"
        assert _run_json(dispatched)["locus"] != _run_json(in_process)["locus"]


class TestTheTierIsNoLongerCalledALane:
    """AC2. The rename, and the absence it depends on.

    A duplicate carrying both spellings would satisfy every other test here
    and leave the defect exactly where it was, so the absence is asserted
    directly rather than implied by the presence of the new key.
    """

    def test_the_tier_is_written_under_routing_tier(self, tmp_path: Path) -> None:
        run_dir, _ = _write(tmp_path, _DISPATCHED)
        assert _run_json(run_dir)["routing_tier"] == "local"

    def test_the_lane_key_never_holds_a_tier(self, tmp_path: Path) -> None:
        """On ``--lane dev`` the ``lane`` key says ``dev``, not ``local``."""
        run_dir, _ = _write(tmp_path, _DISPATCHED)
        run_json = _run_json(run_dir)
        assert run_json["lane"] == "dev"
        assert run_json["lane"] != run_json["routing_tier"]

    def test_both_files_spell_the_tier_the_same_way(self, tmp_path: Path) -> None:
        run_dir, _ = _write(tmp_path, _DISPATCHED)
        assert (
            _run_json(run_dir)["routing_tier"] == _receipt_json(run_dir)["routing_tier"]
        )


class TestTheLocalPathIsThePositiveControl:
    """AC3. The default route still works, and now says what it is.

    Standing rule: routing reaches the local tier by default and the served
    local model stays Qwen3.8-27B. Nothing here changes either — this class
    exists to fail if something does.
    """

    def test_a_local_run_records_in_process_and_no_lane(self, tmp_path: Path) -> None:
        run_dir, _ = _write(tmp_path, _IN_PROCESS)
        run_json = _run_json(run_dir)
        assert run_json["locus"] == "in-process"
        assert run_json["bus"] == "inmemory"
        assert run_json["lane"] is None
        assert run_json["dispatch_target"] is None

    def test_a_local_run_still_attributes_its_route(self, tmp_path: Path) -> None:
        run_dir, _ = _write(tmp_path, _IN_PROCESS)
        assert _run_json(run_dir)["routing_tier"] == "local"
        assert _receipt_json(run_dir)["model"] == "Qwen3.8-27B"


class TestAFailedRunStillSaysWhereItRan:
    """AC4. Where a failed run ran is the first question asked about it.

    The route-unattributed branch writes its own three files and had its own
    ``lane: None``, which was accidentally correct and meant nothing. It now
    carries the same four keys as every other run, because a run that cannot
    name its backend is exactly the one whose locus matters.
    """

    def test_the_unattributed_branch_carries_the_addressing_keys(
        self, tmp_path: Path
    ) -> None:
        run_dir, _ = _write(tmp_path, _DISPATCHED, accepted=False)
        run_json = _run_json(run_dir)
        assert run_json["locus"] == "deployed-lane"
        assert run_json["bus"] == "kafka"
        assert run_json["lane"] == "dev"
        assert run_json["dispatch_target"] == (
            "onex.cmd.omnimarket.delegate-skill.v1 via lab-broker.invalid:19092"
        )

    def test_the_unattributed_branch_names_no_tier(self, tmp_path: Path) -> None:
        """Fail-closed attribution is unchanged: no rung answered, none is named."""
        run_dir, _ = _write(tmp_path, _DISPATCHED, accepted=False)
        run_json = _run_json(run_dir)
        assert run_json["route_attributed"] is False
        assert run_json["routing_tier"] is None


class TestTheWriterCannotBeCalledWithoutAddressing:
    """AC5. The chain, not a note on a happy path.

    ``addressing`` is a required keyword argument. A caller that forgets it
    raises rather than writing a file that omits the four keys, which is the
    state this ticket exists to leave behind — the pre-fix writer took no
    such argument and every one of its callers compiled.
    """

    def test_omitting_addressing_is_a_type_error(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            _write_local_run_files(  # type: ignore[call-arg]
                receipt=_receipt(),
                state_root=tmp_path,
                prompt="Reply with exactly: OK",
                task_type="summarization",
                task_type_resolution="explicit",
            )

    def test_the_addressing_model_forbids_an_unknown_field(self) -> None:
        """A fifth fact gets a field, never a smuggled extra key."""
        with pytest.raises(ValueError):
            ModelDelegateRunAddressing(
                locus=EnumDelegateLocus.IN_PROCESS,
                bus="inmemory",
                broker="lab-broker.invalid:19092",  # type: ignore[call-arg]
            )

    def test_the_resolved_locus_is_never_auto(self, tmp_path: Path) -> None:
        """``auto`` is a request. A file reporting it would report no decision.

        The model accepts the enum member, so this is asserted on what reaches
        the file rather than on construction: every addressing value the CLI
        builds comes from a resolved locus decision, which cannot be ``auto``.
        """
        run_dir, _ = _write(tmp_path, _DISPATCHED)
        assert _run_json(run_dir)["locus"] in {"in-process", "deployed-lane"}
