# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23177 (scenario R9): the C11 delegation legs grade offline.

Every test here reads the committed corpus, the committed legs config and a
hand-written observation file, and performs no network I/O. What is asserted is
the grader: the corpus is the declared set, each leg fails when its property does
not hold, and a planted canary turns the verdict red. Whether a deployed lane
produces those observations is the live driver's question and is not asserted
here.
"""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from omnibase_infra.handlers import handler_c11_delegation_legs as legs
from omnibase_infra.models.model_c11_delegation_legs_request import (
    ModelC11DelegationLegsRequest,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "config" / "c11_delegation_legs.yaml"
HEALTHY = REPO / "tests" / "fixtures" / "omn23177" / "legs_healthy.json"
CORPUS = REPO / "config" / "c11_envelope_corpus.json"


def _config() -> legs.LegsConfig:
    return legs.load_legs_config(CONFIG_PATH)


def _healthy() -> dict[str, Any]:
    document = json.loads(HEALTHY.read_text(encoding="utf-8"))
    observations = document["observations"]
    assert isinstance(observations, dict)
    return observations


def _grade(observations: dict[str, Any]) -> list[legs.LegResult]:
    return legs.grade_legs(_config(), observations)


def _handle(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "config_path": str(CONFIG_PATH),
        "replay_path": str(HEALTHY),
    }
    fields.update(overrides)
    return asyncio.run(
        legs.HandlerC11DelegationLegs().handle(ModelC11DelegationLegsRequest(**fields))
    )


@pytest.mark.unit
def test_the_committed_corpus_is_the_declared_set() -> None:
    config = _config()
    corpus = legs.load_corpus(CORPUS, config)
    assert config.corpus_ids == ("E1", "E2", "E3", "E4", "E5", "E6", "E11")
    assert sorted(corpus) == sorted(config.corpus_ids)
    assert config.refusal_failure_code == "ONEX_CORE_345_ENVELOPE_VALIDATION_FAILED"


def _payload(envelope: object) -> dict[str, Any]:
    assert isinstance(envelope, dict)
    payload = envelope.get("payload")
    assert isinstance(payload, dict)
    return payload


@pytest.mark.unit
def test_each_corpus_envelope_carries_the_defect_it_names() -> None:
    corpus = legs.load_corpus(CORPUS, _config())
    assert "prompt" not in _payload(corpus["E1"])
    assert "payload" not in corpus["E2"]
    assert isinstance(_payload(corpus["E3"])["prompt"], int)
    assert not isinstance(_payload(corpus["E4"])["prompt"], str)
    assert "exec" in _payload(corpus["E5"])
    assert _payload(corpus["E6"])["task_type"] == "shell_exec"
    assert _payload(corpus["E11"])["prompt"] == ""


@pytest.mark.unit
def test_a_corpus_missing_an_envelope_is_an_input_error(tmp_path: Path) -> None:
    document = json.loads(CORPUS.read_text(encoding="utf-8"))
    document["envelopes"] = [row for row in document["envelopes"] if row["id"] != "E6"]
    broken = tmp_path / "corpus.json"
    broken.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(legs.ProbeInputError, match="grader demands"):
        legs.load_corpus(broken, _config())


@pytest.mark.unit
def test_a_config_with_equal_rungs_is_an_input_error(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    document["ladder"]["rung_2"] = document["ladder"]["rung_1"]
    broken = tmp_path / "config.yaml"
    broken.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(legs.ProbeInputError, match="distinct"):
        legs.load_legs_config(broken)


@pytest.mark.unit
def test_a_healthy_lane_grades_every_leg_pass() -> None:
    results = _grade(_healthy())
    assert [leg.name for leg in results] == [
        "envelope_corpus",
        "policy_refusal",
        "rung_503_control",
        "canary",
    ]
    assert [leg.reason for leg in results if not leg.ok] == []


@pytest.mark.unit
def test_an_envelope_with_no_terminal_fails_the_corpus_leg() -> None:
    observations = _healthy()
    observations["envelope_corpus"]["E3"] = {"terminals": []}
    leg = _grade(observations)[0]
    assert leg.ok is False
    assert "E3: no terminal observed" in leg.reason


@pytest.mark.unit
def test_an_envelope_missing_from_the_observations_fails_the_corpus_leg() -> None:
    observations = _healthy()
    del observations["envelope_corpus"]["E11"]
    leg = _grade(observations)[0]
    assert leg.ok is False
    assert "E11" in leg.reason


@pytest.mark.unit
def test_an_untyped_or_retryable_refusal_fails_the_corpus_leg() -> None:
    untyped = _healthy()
    untyped["envelope_corpus"]["E1"]["terminals"][0]["failure_code"] = None
    assert _grade(untyped)[0].ok is False

    retryable = _healthy()
    retryable["envelope_corpus"]["E2"]["terminals"][0]["retryable"] = True
    leg = _grade(retryable)[0]
    assert leg.ok is False
    assert "non-retryable" in leg.reason


@pytest.mark.unit
def test_one_wrong_terminal_among_several_for_one_envelope_fails() -> None:
    observations = _healthy()
    good = observations["envelope_corpus"]["E5"]["terminals"][0]
    bad = copy.deepcopy(good)
    bad["failure_code"] = "ONEX_CORE_007_INVALID_INPUT"
    observations["envelope_corpus"]["E5"]["terminals"] = [good, bad]
    assert _grade(observations)[0].ok is False


@pytest.mark.unit
def test_a_policy_refusal_that_climbs_to_rung_two_fails() -> None:
    observations = _healthy()
    observations["policy_refusal"]["attempts"].append(
        {"backend_id": _config().rung_2, "acceptance_decision": "accept"}
    )
    leg = _grade(observations)[1]
    assert leg.ok is False
    assert "climbed" in leg.reason


@pytest.mark.unit
def test_a_policy_refusal_retried_on_rung_one_fails() -> None:
    observations = _healthy()
    attempt = observations["policy_refusal"]["attempts"][0]
    observations["policy_refusal"]["attempts"] = [attempt, dict(attempt), dict(attempt)]
    leg = _grade(observations)[1]
    assert leg.ok is False
    assert "observed 3" in leg.reason


@pytest.mark.unit
def test_a_503_control_that_never_moves_fails() -> None:
    observations = _healthy()
    attempts = observations["rung_503_control"]["attempts"]
    observations["rung_503_control"]["attempts"] = attempts[:1]
    leg = _grade(observations)[2]
    assert leg.ok is False
    assert "did not move" in leg.reason


@pytest.mark.unit
def test_a_503_control_whose_second_rung_is_not_accepted_fails() -> None:
    observations = _healthy()
    observations["rung_503_control"]["attempts"][1]["acceptance_decision"] = "reject"
    assert _grade(observations)[2].ok is False


@pytest.mark.unit
def test_legs_with_no_observation_fail_and_are_never_skipped() -> None:
    assert [leg.ok for leg in _grade({})] == [False, False, False, False]


@pytest.mark.unit
@pytest.mark.parametrize("surface", ["answer", "receipt", "events"])
def test_the_canary_in_any_surface_fails_the_canary_leg(surface: str) -> None:
    config = _config()
    observations = _healthy()
    observations["canary"][surface] = {"nested": [f"x {config.canary_value} y"]}
    leg = _grade(observations)[3]
    assert leg.ok is False
    assert surface in leg.reason


@pytest.mark.unit
def test_a_canary_scan_over_empty_surfaces_proves_nothing() -> None:
    leg = legs.grade_canary(_config(), {"answer": "", "receipt": None, "events": []})
    assert leg.ok is False
    assert "proves nothing" in leg.reason


@pytest.mark.unit
def test_planting_does_not_mutate_the_observations() -> None:
    observations = _healthy()
    before = json.dumps(observations, sort_keys=True)
    legs.plant_canary_in_answer(_config(), observations)
    assert json.dumps(observations, sort_keys=True) == before


@pytest.mark.unit
def test_the_handler_passes_the_healthy_file_and_fails_when_the_canary_is_planted(
    tmp_path: Path,
) -> None:
    record = tmp_path / "legs.json"
    clean = _handle(record_path=str(record))
    assert clean.success is True
    assert clean.outcome == "PASS"
    written = json.loads(record.read_text(encoding="utf-8"))
    assert written["verdict"] == "pass"
    assert written["planted_canary"] is False

    planted = _handle(record_path=str(record), plant_canary_in_answer=True)
    assert planted.success is False
    assert planted.outcome == "FAIL"
    assert planted.reasons == ("canary: the planted canary appears in: answer",)
    written = json.loads(record.read_text(encoding="utf-8"))
    assert written["verdict"] == "fail"
    assert written["planted_canary"] is True
    assert _config().canary_value not in record.read_text(encoding="utf-8")


@pytest.mark.unit
def test_the_handler_could_not_run_when_the_replay_file_is_unreadable(
    tmp_path: Path,
) -> None:
    result = _handle(replay_path=str(tmp_path / "absent.json"))
    assert result.success is False
    assert result.outcome == "COULD_NOT_RUN"
