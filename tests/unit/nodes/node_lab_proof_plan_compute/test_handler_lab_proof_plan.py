# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""What the foundation_override plan renders, and what it refuses to render.

Ticket: OMN-19572
"""

from __future__ import annotations

import pytest

from omnibase_infra.lab_proof.enum_lab_proof_attribution import (
    EnumLabProofAttribution,
)
from omnibase_infra.lab_proof.enum_lab_proof_step_id import EnumLabProofStepId
from omnibase_infra.lab_proof.enum_lab_proof_step_phase import EnumLabProofStepPhase
from omnibase_infra.lab_proof.lab_proof_profile_registry import (
    load_lab_proof_profile_registry,
)
from omnibase_infra.lab_proof.model_lab_proof_plan import ModelLabProofPlan
from omnibase_infra.nodes.node_lab_proof_plan_compute.handlers.handler_lab_proof_plan import (
    SABOTAGE_LINE,
    HandlerLabProofPlan,
    LabProofPlanError,
)
from tests.unit.nodes.node_lab_proof_plan_compute.lab_proof_fixtures import (
    BASE,
    HEAD,
    INFRA,
    LANE,
    REGISTRY,
    RUN,
    core_plan,
    core_request,
)

pytestmark = pytest.mark.unit

_ID = EnumLabProofStepId
SHELLS = {"sh", "bash", "zsh", "/bin/sh", "/bin/bash"}


def _ids(plan: ModelLabProofPlan) -> list[EnumLabProofStepId]:
    return [step.step_id for step in plan.steps]


def test_plan_is_ordered_setup_then_prove_then_teardown_then_residue() -> None:
    plan = core_plan()
    order = [
        EnumLabProofStepPhase.SETUP,
        EnumLabProofStepPhase.PROVE,
        EnumLabProofStepPhase.TEARDOWN,
        EnumLabProofStepPhase.RESIDUE,
    ]
    phases = [order.index(step.phase) for step in plan.steps]
    assert phases == sorted(phases)
    assert _ids(plan)[0] is _ID.HOST_LOAD
    assert _ids(plan)[-1] is _ID.RESIDUE_POSITIVE_CONTROL


def test_plan_is_deterministic() -> None:
    assert core_plan() == core_plan()


def test_no_step_is_shell_text_and_every_path_stays_in_the_run() -> None:
    plan = core_plan()
    assert plan.workdir == f"{LANE}/{RUN}"
    assert plan.log_dir == f"{LANE}/logs/{RUN}"
    for step in plan.steps:
        assert step.argv[0] not in SHELLS, step.step_id
        assert step.cwd == LANE or step.cwd.startswith(f"{LANE}/{RUN}"), step.step_id
        for arg in step.argv:
            if arg.startswith("/") and "\n" not in arg:
                assert arg.startswith((LANE, "/app/")), (
                    step.step_id,
                    arg,
                )


def test_subject_and_infra_are_fetched_at_exact_commits() -> None:
    plan = core_plan()
    steps = {step.step_id: step for step in plan.steps}
    assert steps[_ID.SUBJECT_REV].expect_stdout_equals == HEAD
    assert steps[_ID.INFRA_REV].expect_stdout_equals == INFRA
    assert "refs/pull/1766/head" in steps[_ID.SUBJECT_FETCH].argv
    assert (
        "https://github.com/OmniNode-ai/omnibase_core.git"
        in steps[_ID.SUBJECT_FETCH].argv
    )
    assert INFRA in steps[_ID.INFRA_FETCH].argv


def test_a_moved_head_stops_the_run_as_a_harness_failure() -> None:
    step = next(s for s in core_plan().steps if s.step_id is _ID.SUBJECT_REV)
    assert step.must_succeed
    assert step.attribution is EnumLabProofAttribution.HARNESS


def test_the_override_installs_the_package_under_test_over_a_run_scoped_base() -> None:
    steps = {step.step_id: step for step in core_plan().steps}
    build = steps[_ID.BUILD_OVERRIDE].argv
    assert "PACKAGE_NAME=omnibase-core" in build
    assert f"PROVED_SHA={HEAD}" in build
    assert f"BASE_IMAGE=omnibase-infra-local-runtime:lab-proof-base-{RUN}" in build
    assert build[build.index("-t") + 1] == "omnibase-infra-local-runtime:latest"
    assert steps[_ID.BUILD_OVERRIDE].attribution is EnumLabProofAttribution.SUBJECT
    assert steps[_ID.BUILD_BASE].attribution is EnumLabProofAttribution.HARNESS
    # up never rebuilds: the derived image must be the one that boots
    assert "--build" not in steps[_ID.UP].argv


def test_identity_is_read_from_both_runtime_containers() -> None:
    steps = {step.step_id: step for step in core_plan().steps}
    for step_id, container in (
        (_ID.IDENTITY_RUNTIME_MAIN, "omnibase-infra-local-omninode-runtime"),
        (_ID.IDENTITY_RUNTIME_EFFECTS, "omnibase-infra-local-runtime-effects"),
    ):
        argv = steps[step_id].argv
        assert argv[:3] == ("docker", "exec", container)
        assert argv[-3:] == ("module", "omnibase_core", "omnibase-core")


def test_container_logs_are_counted_not_copied_into_the_report() -> None:
    steps = {step.step_id: step for step in core_plan().steps}
    for step_id in (_ID.WIRING_LOGS_RUNTIME_MAIN, _ID.WIRING_LOGS_RUNTIME_EFFECTS):
        assert steps[step_id].record_output is False
        assert "Auto-wiring failed for" in steps[step_id].grep_patterns


def test_focused_tests_run_only_the_changed_test_files() -> None:
    steps = {step.step_id: step for step in core_plan().steps}
    argv = steps[_ID.FOCUSED_TESTS].argv
    assert (
        argv[-1]
        == "tests/unit/test_topic_base_dod_verify_duplicate_retired_omn19153.py"
    )
    assert "src/omnibase_core/topics.py" not in argv
    assert steps[_ID.FOCUSED_TESTS].attribution is EnumLabProofAttribution.SUBJECT


def test_no_focused_tests_step_when_the_pr_changes_no_tests() -> None:
    plan = core_plan(changed=("src/omnibase_core/topics.py",))
    assert _ID.FOCUSED_TESTS not in _ids(plan)


def test_teardown_and_residue_cover_every_surface_the_run_creates() -> None:
    ids = _ids(core_plan())
    for step_id in (
        _ID.DOWN,
        _ID.REMOVE_IMAGES,
        _ID.REMOVE_WORKDIR,
        _ID.RESIDUE_CONTAINERS,
        _ID.RESIDUE_VOLUMES,
        _ID.RESIDUE_NETWORKS,
        _ID.RESIDUE_IMAGES,
        _ID.RESIDUE_PORTS,
        _ID.RESIDUE_WORKDIR,
        _ID.RESIDUE_POSITIVE_CONTROL,
    ):
        assert step_id in ids
    positive = next(
        s for s in core_plan().steps if s.step_id is _ID.RESIDUE_POSITIVE_CONTROL
    )
    assert positive.expect_stdout_nonempty
    assert "label=com.docker.compose.project=omnibase-infra-dev-202" in positive.argv


def test_negative_control_sabotages_the_package_before_it_is_hashed() -> None:
    plan = core_plan(negative_control=True)
    ids = _ids(plan)
    assert ids.index(_ID.NEGATIVE_CONTROL_SABOTAGE) < ids.index(_ID.SUBJECT_HASH)
    sabotage = next(s for s in plan.steps if s.step_id is _ID.NEGATIVE_CONTROL_SABOTAGE)
    assert sabotage.argv[-2].endswith("/override/subject/src/omnibase_core/__init__.py")
    assert sabotage.argv[-1] == SABOTAGE_LINE
    assert _ID.NEGATIVE_CONTROL_SABOTAGE not in _ids(core_plan())


def test_base_control_fetches_the_merge_base_by_sha() -> None:
    plan = core_plan(base_control=True)
    steps = {step.step_id: step for step in plan.steps}
    assert BASE in steps[_ID.SUBJECT_FETCH].argv
    assert steps[_ID.SUBJECT_REV].expect_stdout_equals == BASE
    assert plan.subject.is_base_control


def test_refuses_a_subject_from_another_repository() -> None:
    request = core_request(profile_repo="OmniNode-ai/omnibase_spi")
    with pytest.raises(LabProofPlanError, match="not this profile's repository"):
        HandlerLabProofPlan().handle(request)


def test_refuses_a_row_that_runs_by_hand() -> None:
    registry = load_lab_proof_profile_registry(REGISTRY)
    market = registry.profile_for("OmniNode-ai/omnimarket").variants[0]
    with pytest.raises(LabProofPlanError, match="not by nodes"):
        HandlerLabProofPlan().handle(core_request(variant=market))


def test_refuses_a_lane_root_that_is_not_a_prover_directory() -> None:
    with pytest.raises(ValueError, match="lane_root"):
        core_request(lane_root="/home/prover")
    with pytest.raises(ValueError, match="lane_root"):
        core_request(lane_root="/")


def test_refuses_a_non_sha_infra_ref() -> None:
    with pytest.raises(ValueError, match="infra_sha"):
        core_request(infra_sha="dev")


def test_the_override_dockerfile_comes_from_the_harness_not_the_built_infra() -> None:
    steps = {step.step_id: step for step in core_plan().steps}
    assert steps[_ID.OVERRIDE_DOCKERFILE].argv[1] == (
        f"{LANE}/harness/docker/lab_proof/Dockerfile.foundation-override"
    )


def test_refuses_a_harness_outside_the_lane_or_inside_the_run() -> None:
    with pytest.raises(ValueError, match="inside lane_root"):
        core_request(harness_root="/opt/omnibase_infra")
    with pytest.raises(ValueError, match="run directory"):
        core_request(harness_root=f"{LANE}/{RUN}/harness")
