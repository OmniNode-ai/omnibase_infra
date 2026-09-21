# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-18988 — re-emit for ANY binding failure, not only the queued-ahead one.

WHY THE TRIGGER WIDENS
----------------------
The first cut re-emitted for shas that had NO receipt, which is the queued-ahead
case. The receipt for ``430ff3434`` showed the same defect wearing a different
hat: it FAILED on two checks, ``deployed_revision`` INDETERMINATE with
``commands_ahead`` ZERO (the agent answered HTTP 404 for the run's own
correlation id, so the lane's budget never started) and
``probe_generation_bound`` (the probe read a different container generation than
convergence verified). The lane has since converged and runs that sha.

Neither is a statement about the lane. Both are the run failing to BIND its
observation to the thing under test. Frozen as FAIL forever, under the
inheritance rule that verdict is then inherited by every non-runtime-affecting
merge behind it, so the release cut can never unblock by waiting. That is the
OMN-18976 defect class exactly, so the trigger is the CLASS OF FAILURE rather
than the absence of a receipt.

THE LINE THIS MODULE DEFENDS
-----------------------------
A receipt whose health checks genuinely failed is NEVER re-emitted. Re-emitting
one would convert a real finding about the lane into a PASS, which is strictly
worse than the FAIL this ticket removes. So every test here is about telling the
two apart, and the ones most worth having are the ones that could quietly
misclassify:

* ``deployed_revision`` is binding-class ONLY when its outcome is INDETERMINATE.
  The same check with outcome FAIL is the lane having had its whole budget and
  not converged -- a statement about the lane, never re-emitted. Keying on the
  check NAME alone would collapse these two.
* a receipt failing one binding check AND one health check is not re-emittable.
  Any non-binding failure poisons the whole receipt.
* a PASS is never re-emitted, and neither is a receipt with no failing checks.

THE BINDING MUST BE RE-ESTABLISHED, NOT INHERITED
--------------------------------------------------
The converged run could not bind its own observation -- that is why its receipt
failed. So a re-emission cannot copy that run's binding; it reads the lane's
actual state itself, across three independent surfaces, and records which one
answered. A surface that cannot be read is not a refusal as long as another
answers; all three silent IS a refusal, because then nothing has been bound.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scripts.ci.lab_pass_receipt import (
    EnumLabLane,
    EnumLabPassCheckOutcome,
    ModelLabPassCheck,
    build_receipt,
    is_binding_only_failure,
    resolve_lane_binding,
)

pytestmark = pytest.mark.unit

_S = "b" * 40
_X = "d" * 40


def _receipt(*checks: ModelLabPassCheck):
    return build_receipt(
        sha=_S,
        lane=EnumLabLane.COMPOSE_DEV,
        started_at=datetime(2026, 9, 21, 9, 18, tzinfo=UTC),
        finished_at=datetime(2026, 9, 21, 9, 55, tzinfo=UTC),
        checks=list(checks),
        agent_command_id=None,
    )


def _ok(name: str) -> ModelLabPassCheck:
    return ModelLabPassCheck(name=name, ok=True, evidence="read fine")


def _indeterminate(name: str) -> ModelLabPassCheck:
    return ModelLabPassCheck(
        name=name,
        ok=False,
        evidence="INDETERMINATE: the agent reports no job for this correlation",
        indeterminate=True,
    )


def _failed(name: str) -> ModelLabPassCheck:
    return ModelLabPassCheck(name=name, ok=False, evidence="503 not ready")


class TestOnlyABindingFailureIsReEmittable:
    def test_the_430ff343_shape_is_re_emittable(self) -> None:
        # Both failures are binding-class: an unestablished convergence and a
        # probe bound to the wrong container generation.
        receipt = _receipt(
            _indeterminate("deployed_revision"),
            _failed("probe_generation_bound"),
            _ok("ready_main"),
            _ok("migrations_applied"),
        )
        assert is_binding_only_failure(receipt) is True

    def test_an_unestablished_convergence_alone_is_re_emittable(self) -> None:
        assert (
            is_binding_only_failure(
                _receipt(_indeterminate("deployed_revision"), _ok("ready_main"))
            )
            is True
        )

    def test_a_convergence_the_lane_actually_failed_is_NOT_re_emittable(self) -> None:
        # The discriminator that keying on the check NAME would destroy. Same
        # check, outcome FAIL rather than INDETERMINATE: the lane had its whole
        # budget and did not converge. That is a finding about the lane.
        assert (
            is_binding_only_failure(
                _receipt(_failed("deployed_revision"), _ok("ready_main"))
            )
            is False
        )

    @pytest.mark.parametrize(
        "name",
        [
            "ready_main",
            "ready_effects",
            "health_dimensions",
            "migrations_applied",
            "consumer_group_lag",
            "timed_out_before_ready",
        ],
    )
    def test_a_genuine_health_failure_is_never_re_emittable(self, name: str) -> None:
        assert (
            is_binding_only_failure(_receipt(_failed(name), _ok("some_other_check")))
            is False
        )

    def test_one_health_failure_poisons_a_mostly_binding_receipt(self) -> None:
        # The case most likely to be got wrong: mixing them must refuse, not
        # re-emit the binding part and ignore the rest.
        assert (
            is_binding_only_failure(
                _receipt(
                    _indeterminate("deployed_revision"),
                    _failed("probe_generation_bound"),
                    _failed("ready_main"),
                )
            )
            is False
        )

    def test_a_passing_receipt_is_not_re_emittable(self) -> None:
        # Nothing to fix, and re-emitting would churn an artifact for no reason.
        assert is_binding_only_failure(_receipt(_ok("ready_main"))) is False


class TestTheBindingIsReEstablishedAcrossThreeSurfaces:
    """The converged run could not bind; a re-emission must bind for itself."""

    def test_the_container_revision_label_binds(self) -> None:
        binding = resolve_lane_binding(
            sha=_S,
            container_revision=_X,
            agent_loaded_code_sha=None,
            ready_version_revision=None,
            contains=lambda candidate, observed: candidate == _S and observed == _X,
        )
        assert binding is not None
        assert binding.surface == "container-revision"
        assert binding.observed == _X

    def test_the_agent_loaded_code_sha_binds_when_the_label_is_silent(self) -> None:
        binding = resolve_lane_binding(
            sha=_S,
            container_revision=None,
            agent_loaded_code_sha=_X,
            ready_version_revision=None,
            contains=lambda candidate, observed: True,
        )
        assert binding is not None
        assert binding.surface == "agent-loaded-code-sha"

    def test_the_ready_version_binds_when_the_first_two_are_silent(self) -> None:
        binding = resolve_lane_binding(
            sha=_S,
            container_revision=None,
            agent_loaded_code_sha=None,
            ready_version_revision=_X,
            contains=lambda candidate, observed: True,
        )
        assert binding is not None
        assert binding.surface == "ready-version"

    def test_all_three_silent_is_a_refusal(self) -> None:
        # Nothing bound means nothing may be claimed. Emitting here would be
        # the exact fabrication this whole ticket exists to avoid.
        assert (
            resolve_lane_binding(
                sha=_S,
                container_revision=None,
                agent_loaded_code_sha=None,
                ready_version_revision=None,
                contains=lambda candidate, observed: True,
            )
            is None
        )

    def test_a_surface_that_does_NOT_contain_the_sha_is_a_refusal(self) -> None:
        # The lane is running something that does not carry this sha. That is
        # not a binding; it is the absence of one.
        assert (
            resolve_lane_binding(
                sha=_S,
                container_revision=_X,
                agent_loaded_code_sha=_X,
                ready_version_revision=_X,
                contains=lambda candidate, observed: False,
            )
            is None
        )

    def test_a_readable_but_non_containing_surface_does_not_block_a_later_one(
        self,
    ) -> None:
        # Deliberate: the container label can lag a restart while the agent has
        # already recorded the new code. The first surface to ESTABLISH
        # containment wins; one that merely answers does not veto the others.
        seen: list[str] = []

        def _contains(candidate: str, observed: str) -> bool:
            seen.append(observed)
            return observed == "c" * 40

        binding = resolve_lane_binding(
            sha=_S,
            container_revision=_X,
            agent_loaded_code_sha="c" * 40,
            ready_version_revision=None,
            contains=_contains,
        )
        assert binding is not None
        assert binding.surface == "agent-loaded-code-sha"
        assert seen == [_X, "c" * 40]

    def test_the_binding_names_the_surface_for_converged_via(self) -> None:
        binding = resolve_lane_binding(
            sha=_S,
            container_revision=_X,
            agent_loaded_code_sha=None,
            ready_version_revision=None,
            contains=lambda candidate, observed: True,
        )
        assert binding is not None
        # converged_via must say WHICH surface answered, not merely that one did.
        assert binding.converged_via == f"{_X} via container-revision"
