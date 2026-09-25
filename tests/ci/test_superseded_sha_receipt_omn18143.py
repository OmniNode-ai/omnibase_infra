# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A superseded sha may not receipt a silent PASS (OMN-18143).

WHY THIS EXISTS, AND WHY IT IS NOT REDUNDANT WITH ``deployed_revision``
------------------------------------------------------------------------
The deploy agent now runs the NEWEST foldable rebuild command for a lane and
records the ones it replaced. Convergence has been CONTAINMENT since OMN-18388
-- a lane running a descendant of the merge sha has exercised the change -- so
a superseded sha's own verify job watches the lane converge onto the sha that
ran, and ``deployed_revision`` reports ``ok``.

That verdict is true and it is about the LANE. It is not the statement rule
24(b) gates on, which is about ONE sha. A superseded command's compose content
was never generated: the agent pins the build to the commit it is running, and
that commit is the newer one. So the receipt for the superseded sha would have
carried four green checks for a tree nothing ever built, and delivery would
have taken it on the strength of a pass another commit earned.

THE VERDICT IS FAIL, NOT PASS AND NOT INDETERMINATE
----------------------------------------------------
FAIL because the fact is ESTABLISHED and negative: the agent's own job record
names the commit that ran instead, so this is not a check that could not be
resolved. ``indeterminate`` is reserved here for the case where the agent
could not be asked at all, which is a question about the hop rather than about
the sha -- the distinction OMN-18573 put into the receipt vocabulary.

What the FAIL costs is bounded and was checked against the delivery gate
before being chosen: ``evaluate_gate`` passes on ANY lane's PASS receipt, and
the delivering workflow's own boot gate emits an ``onex-lab`` receipt for the
same sha in the same run. A superseded sha therefore stays deliverable on the
lab lane that did exercise it, and loses only the compose-dev claim it never
earned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.ci import check_dev_lane_staleness as staleness_module

pytestmark = pytest.mark.unit

SHA = "a5d95fa24a25b25f33e5b86f1b09093758e6ae16"
RUNNER_SHA = "6b3cfb0d47f249500c0dbb9e537c0d94f357ae9c"
CORRELATION = "52e430b2-1f1e-4d2a-9a3e-8f4b2c1d0e77"
RUNNER_CORRELATION = "6402141d-3b2c-4e5f-8a91-0c7d6e5f4a3b"

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github/workflows/runtime-rebuild-trigger.yml"
)


@pytest.fixture(scope="module")
def staleness() -> Any:
    return staleness_module


def _opener(status: int, body: str) -> Any:
    def _fetch(url: str, timeout: float) -> tuple[int, str]:
        return status, body

    return _fetch


class TestReadingTheSupersession:
    def test_a_superseded_record_is_reported_with_the_sha_that_ran(
        self, staleness: Any
    ) -> None:
        probe = staleness.read_agent_supersession(
            agent_url="http://agent:8098",
            correlation_id=CORRELATION,
            opener=_opener(
                200,
                json.dumps(
                    {
                        "status": "superseded",
                        "accepted_at": "2026-09-19T02:58:00Z",
                        "superseded_by_sha": RUNNER_SHA,
                        "superseded_by_correlation_id": RUNNER_CORRELATION,
                    }
                ),
            ),
        )
        assert probe.superseded is True
        assert probe.superseded_by_sha == RUNNER_SHA
        assert probe.readable is True
        evidence = staleness.supersession_evidence(probe, SHA)
        assert RUNNER_SHA in evidence
        assert SHA in evidence

    def test_a_job_that_ran_normally_is_not_superseded(self, staleness: Any) -> None:
        probe = staleness.read_agent_supersession(
            agent_url="http://agent:8098",
            correlation_id=CORRELATION,
            opener=_opener(
                200,
                json.dumps(
                    {
                        "status": "success",
                        "accepted_at": "2026-09-19T02:58:00Z",
                        "superseded_by_sha": None,
                        "superseded_by_correlation_id": None,
                    }
                ),
            ),
        )
        assert probe.superseded is False
        assert probe.readable is True

    def test_an_agent_too_old_to_serve_the_field_is_not_superseded(
        self, staleness: Any
    ) -> None:
        """An agent that cannot coalesce has not coalesced.

        The live .201 agent self-updates at a job boundary, so for one job
        after this lands the endpoint predates the field. Treating an absent
        field as unreadable would put an indeterminate check -- and therefore
        a non-PASS receipt -- on a sha that was built on its own tree.
        """
        probe = staleness.read_agent_supersession(
            agent_url="http://agent:8098",
            correlation_id=CORRELATION,
            opener=_opener(200, json.dumps({"accepted_at": "2026-09-19T02:58:00Z"})),
        )
        assert probe.superseded is False
        assert probe.readable is True

    def test_a_command_the_agent_has_never_seen_is_not_superseded(
        self, staleness: Any
    ) -> None:
        """A 404 is a real answer here, unlike on the acceptance read.

        A command with no job record was never dequeued, so it cannot have
        been folded. ``read_agent_acceptance`` treats the same 404 as
        unresolved because it is asking a different question -- when the
        lane's budget started -- which genuinely has no answer yet.
        """
        probe = staleness.read_agent_supersession(
            agent_url="http://agent:8098",
            correlation_id=CORRELATION,
            opener=_opener(404, "{}"),
        )
        assert probe.superseded is False
        assert probe.readable is True

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (500, "upstream exploded"),
            (200, "not json at all"),
            (200, json.dumps({"superseded_by_sha": "a5d95fa2"})),
        ],
    )
    def test_an_unreadable_answer_is_never_reported_as_not_superseded(
        self, staleness: Any, status: int, body: str
    ) -> None:
        probe = staleness.read_agent_supersession(
            agent_url="http://agent:8098",
            correlation_id=CORRELATION,
            opener=_opener(status, body),
        )
        assert probe.superseded is False
        assert probe.readable is False, (
            "an unread answer and a negative answer must not be the same "
            "value; the emitter turns the first into an indeterminate check"
        )
        assert "could not be established" in staleness.supersession_evidence(probe, SHA)

    def test_an_unreachable_agent_is_unreadable_rather_than_fatal(
        self, staleness: Any
    ) -> None:
        def _boom(url: str, timeout: float) -> tuple[int, str]:
            raise OSError("connection refused")

        probe = staleness.read_agent_supersession(
            agent_url="http://agent:8098", correlation_id=CORRELATION, opener=_boom
        )
        assert probe.readable is False
        assert "connection refused" in probe.reason

    def test_a_superseded_sha_fails_the_check_rather_than_passing_it(
        self, staleness: Any
    ) -> None:
        """The decision this change turns on, asserted rather than implied.

        ``deployed_revision`` is ``ok`` on exactly these runs, because
        convergence is containment and the lane IS running a commit that
        contains this sha. If this check were ``ok`` too, the receipt would
        be a PASS and rule 24(b) would deliver a commit whose tree nothing
        built.
        """
        assert (
            staleness.supersession_check_outcome(
                staleness.ModelSupersessionProbe(
                    superseded_by_sha=RUNNER_SHA,
                    superseded_by_correlation_id=RUNNER_CORRELATION,
                ),
                SHA,
            )
            == "fail"
        )

    def test_an_unaskable_agent_is_indeterminate_and_a_clean_record_is_ok(
        self, staleness: Any
    ) -> None:
        """A FAIL is a statement about the sha; an INDETERMINATE is about the hop."""
        assert (
            staleness.supersession_check_outcome(
                staleness.ModelSupersessionProbe(reason="agent unreachable"), SHA
            )
            == "indeterminate"
        )
        assert (
            staleness.supersession_check_outcome(
                staleness.ModelSupersessionProbe(), SHA
            )
            == "ok"
        )

    def test_the_evidence_is_always_one_line_and_never_empty(
        self, staleness: Any
    ) -> None:
        """``ModelLabPassCheck`` refuses empty evidence, and GITHUB_OUTPUT is
        line-oriented, so an unwritten receipt is the failure mode here."""
        for probe in (
            staleness.ModelSupersessionProbe(),
            staleness.ModelSupersessionProbe(
                superseded_by_sha=RUNNER_SHA,
                superseded_by_correlation_id=RUNNER_CORRELATION,
            ),
            staleness.ModelSupersessionProbe(reason="line one\nline two"),
        ):
            evidence = staleness.supersession_evidence(probe, SHA)
            assert evidence
            assert "\n" not in evidence
            assert "`" not in evidence


class TestASameRefCoalesceOMN19499:
    """A fold into the receipt's OWN commit is not a supersession (OMN-19499).

    Since the OMN-19270 lineage fence, the agent records a command as
    superseded by the RUNNING build when that build already carries the ref,
    naming the running infra sha and no correlation id. When that running sha
    IS the receipt sha, the lane runs exactly this commit's tree -- the claim
    ``superseded_by_newer_rebuild`` exists to protect is true, so the check
    reads ``ok``. Live case: c74a2fb01, run 36050688911 attempt 3, job
    107863057541, agent command 5edc3798, FAIL on this check alone.
    """

    def test_a_same_ref_coalesce_reads_ok(self, staleness: Any) -> None:
        probe = staleness.ModelSupersessionProbe(superseded_by_sha=SHA)
        assert staleness.supersession_check_outcome(probe, SHA) == "ok"

    def test_a_same_ref_coalesce_is_compared_case_insensitively(
        self, staleness: Any
    ) -> None:
        probe = staleness.ModelSupersessionProbe(superseded_by_sha=SHA)
        assert staleness.supersession_check_outcome(probe, SHA.upper()) == "ok"

    def test_the_same_ref_evidence_says_this_commit_ran(self, staleness: Any) -> None:
        evidence = staleness.supersession_evidence(
            staleness.ModelSupersessionProbe(superseded_by_sha=SHA), SHA
        )
        assert "same commit" in evidence
        assert "did not build" not in evidence

    def test_a_fold_into_a_different_commit_still_fails(self, staleness: Any) -> None:
        """Positive control: the fix must not pass every supersession."""
        probe = staleness.ModelSupersessionProbe(
            superseded_by_sha=RUNNER_SHA,
            superseded_by_correlation_id=RUNNER_CORRELATION,
        )
        assert staleness.supersession_check_outcome(probe, SHA) == "fail"
        assert "did not build" in staleness.supersession_evidence(probe, SHA)

    def test_an_abbreviated_receipt_sha_is_not_a_same_ref_match(
        self, staleness: Any
    ) -> None:
        """Rule 24(b) gates on the exact sha, so a prefix proves nothing here."""
        probe = staleness.ModelSupersessionProbe(superseded_by_sha=SHA)
        assert staleness.supersession_check_outcome(probe, SHA[:12]) == "fail"

    def test_the_verdict_is_keyed_on_the_receipt_sha(self) -> None:
        """The guard must hand the receipt sha to the verdict, or a caller
        that forgets it silently reverts to the one-argument FAIL."""
        source = (
            Path(__file__).resolve().parents[2]
            / "scripts/ci/check_dev_lane_staleness.py"
        ).read_text()
        assert "supersession_check_outcome(supersession, expected)" in source


class TestTheEmittedCheck:
    def _verify_job(self) -> dict[str, Any]:
        workflow = yaml.safe_load(WORKFLOW.read_text())
        return workflow["jobs"]["verify-lane-converged"]

    def _emit_step(self) -> dict[str, Any]:
        return next(
            step
            for step in self._verify_job()["steps"]
            if "lab_pass_receipt.py emit" in str(step.get("run", ""))
        )

    def test_the_receipt_carries_the_superseded_check(self) -> None:
        run = str(self._emit_step()["run"])
        assert '--check "superseded_by_newer_rebuild:' in run, (
            "without this check a superseded sha's receipt is four green "
            "checks about a tree nothing built"
        )

    def test_the_outcome_and_evidence_come_from_the_guard_through_env(self) -> None:
        step = self._emit_step()
        rendered = yaml.dump(step)
        assert "steps.converge.outputs.superseded_outcome" in rendered
        assert "steps.converge.outputs.superseded_evidence" in rendered
        assert "steps.converge.outputs.superseded_evidence" not in str(step["run"]), (
            "expression-interpolating a value into a run block is the script "
            "injection shape; pass it through env and dereference the variable"
        )

    def test_an_unwritten_outcome_falls_back_to_indeterminate(self) -> None:
        """A guard that died before deciding established nothing."""
        run = str(self._emit_step()["run"])
        assert "SUPERSEDED_OUTCOME=indeterminate" in run
        assert "ok|fail|indeterminate)" in run

    def test_the_guard_writes_all_three_outputs(self) -> None:
        source = (
            Path(__file__).resolve().parents[2]
            / "scripts/ci/check_dev_lane_staleness.py"
        ).read_text()
        for name in ("superseded_by", "superseded_outcome", "superseded_evidence"):
            assert f'_write_output("{name}"' in source
