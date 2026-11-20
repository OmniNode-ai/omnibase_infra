# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""C28 reads the runtime containers a lane's policy leaves running.

A lane that folds the ``effects`` role into its main runtime has no
``-runtime-effects`` container, so the probe must not demand one; a lane that
folds nothing still must have it, and its logs must still be read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_consumer_flow import (
    grade_consumer_flow,
)
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_docker_consumer_flow_target import (
    HandlerDockerConsumerFlowTarget,
    read_folded_profiles,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_lane import (
    ModelConsumerFlowLane,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_observation import (
    ModelConsumerFlowObservation,
)
from omnibase_infra.nodes.node_board_probe_effect.models.model_consumer_flow_request import (
    ModelConsumerFlowRequest,
)
from scripts.ci import c28_consumer_flow_probe as script

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE = Path(__file__).parent / "fixtures/consumer_flow_recorded.json"
LANE = "omnibase-infra-lane"
MAIN = "omninode-runtime"
EFFECTS = "omninode-runtime-effects"
POLICY = REPO_ROOT / "docker" / "runtime-policy.env"


def _policy_lanes() -> dict[str, frozenset[str]]:
    """Every lane the shipped policy names: its compose project and what it folds."""
    values = dict(
        line.split("=", 1)
        for line in POLICY.read_text(encoding="utf-8").splitlines()
        if "=" in line and not line.startswith("#")
    )
    return {
        project: frozenset(
            values.get(
                f"{key.removesuffix('_COMPOSE_PROJECT')}_RUNTIME_MAIN_FOLDED_PROFILES",
                "",
            ).split(",")
        )
        for key, project in values.items()
        if key.endswith("_COMPOSE_PROJECT")
    }


def test_a_lane_that_folds_effects_names_only_the_main_runtime() -> None:
    lane = ModelConsumerFlowLane.from_compose_project(
        LANE, folded_profiles=frozenset({"effects"})
    )
    assert lane.runtime_containers == ("omninode-lane-runtime",)
    assert lane.boot_containers == (
        "omninode-lane-runtime",
        "omnimarket-lane-projection-api",
    )


def test_a_lane_that_folds_nothing_still_names_the_effects_runtime() -> None:
    """Negative control: the effects container is still required without the fold."""
    lane = ModelConsumerFlowLane.from_compose_project(
        LANE, folded_profiles=frozenset({"workers"})
    )
    assert lane.runtime_containers == (
        "omninode-lane-runtime",
        "omninode-lane-runtime-effects",
    )
    assert ModelConsumerFlowLane.from_compose_project(LANE) == lane


def test_the_shipped_policy_names_a_lane_that_folds_effects_and_one_that_does_not() -> (
    None
):
    lanes = _policy_lanes()
    assert any("effects" in folded for folded in lanes.values())
    assert any("effects" not in folded for folded in lanes.values())


def test_every_lane_in_the_shipped_policy_reads_the_containers_it_runs() -> None:
    for project, folded in _policy_lanes().items():
        if not project.startswith("omnibase-infra"):
            continue
        read = read_folded_profiles(REPO_ROOT, project)
        assert ("effects" in read) == ("effects" in folded)
        lane = ModelConsumerFlowLane.from_compose_project(project, folded_profiles=read)
        assert len(lane.runtime_containers) == (1 if "effects" in folded else 2)
        assert lane.runtime_containers[0].endswith("-runtime")


def test_a_project_the_policy_does_not_name_folds_nothing() -> None:
    assert read_folded_profiles(REPO_ROOT, "omnibase-infra-unlisted") == frozenset()


def test_a_checkout_without_a_policy_file_folds_nothing(tmp_path: Path) -> None:
    """The strict reading: every runtime container is expected."""
    assert read_folded_profiles(tmp_path, LANE) == frozenset()


def test_a_policy_path_that_cannot_be_read_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "docker" / "runtime-policy.env").mkdir(parents=True)
    with pytest.raises(OSError):
        read_folded_profiles(tmp_path, LANE)


def test_the_target_reads_the_policy_it_is_given(tmp_path: Path) -> None:
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "runtime-policy.env").write_text(
        f"LAB_COMPOSE_PROJECT={LANE}\nLAB_RUNTIME_MAIN_FOLDED_PROFILES=workers, effects\n",
        encoding="utf-8",
    )
    assert read_folded_profiles(tmp_path, LANE) == {"workers", "effects"}


def test_the_target_reports_an_unreadable_policy_as_unreadable(
    tmp_path: Path,
) -> None:
    """A policy file that cannot be read is INDETERMINATE material, never a guessed lane."""
    (tmp_path / "docker" / "runtime-policy.env").mkdir(parents=True)
    request = ModelConsumerFlowRequest(
        subject_lane="lane",
        compose_project=LANE,
        base_url="http://127.0.0.1:1",
        scratch=tmp_path,
    )
    observation = HandlerDockerConsumerFlowTarget(repo_root=tmp_path)._observe(request)
    assert observation.read_ok is False
    assert "runtime-policy.env" in observation.read_error


def _observation(*, identity: list[str], logs: list[str]) -> dict[str, object]:
    obs = json.loads(FIXTURE.read_text())
    natural = obs["boot"]["natural"]
    template = next(iter(natural.values()))
    obs["boot"]["natural"] = {name: dict(template) for name in logs}
    obs["boot_identity"] = {
        name: {"id": "x", "status": "running", "health": "healthy", "image": "i"}
        for name in identity
    }
    return obs


def _failed_clauses(obs: dict[str, object]) -> set[str]:
    node = grade_consumer_flow(
        ModelConsumerFlowRequest(subject_lane="s", compose_project="omnibase-infra"),
        ModelConsumerFlowObservation(read_ok=True, **obs),
    )
    by_script = {c.name for c in script.grade(obs).checks if not c.ok}
    by_node = (
        set()
        if node.outcome.value == "PASS"
        else {r.split("/", 1)[1].split(":", 1)[0] for r in node.reasons}
    )
    assert by_node == by_script
    return by_node


def test_folded_lane_with_only_main_logs_grades_the_log_clause_green() -> None:
    obs = _observation(identity=[MAIN, "omnimarket-projection-api"], logs=[MAIN])
    assert "runtime_logs_read" not in _failed_clauses(obs)


def test_unfolded_lane_missing_the_effects_logs_still_fails() -> None:
    """Negative control: both containers were inspected, one log is missing."""
    obs = _observation(identity=[MAIN, EFFECTS], logs=[MAIN])
    assert "runtime_logs_read" in _failed_clauses(obs)


def test_folded_lane_with_no_main_logs_still_fails() -> None:
    obs = _observation(identity=[MAIN], logs=[])
    assert "runtime_logs_read" in _failed_clauses(obs)


def test_unfolded_lane_with_both_logs_grades_green() -> None:
    obs = _observation(identity=[MAIN, EFFECTS], logs=[MAIN, EFFECTS])
    assert "runtime_logs_read" not in _failed_clauses(obs)
