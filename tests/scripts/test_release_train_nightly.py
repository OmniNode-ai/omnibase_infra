# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the scheduled release train's decision surface (OMN-18595).

Two obligations, the same pair every decision surface in this repo carries.

1. RED reproduces the state that motivated the ticket. ``test_live_*`` feeds the
   evaluator the exact facts read from the canonical clones on 2026-09-17 --
   omnibase_infra at tag v0.38.30 with zero unreleased release-relevant commits,
   omniintelligence 67 commits behind a tag dated 2026-05-26, omniclaude 65
   behind one dated 2026-06-02 -- and asserts the train renders the verdict and
   the reason a reader has to act on, rather than staying silent.
2. GREEN is discriminating. Every "this cuts" assertion has a sibling that flips
   exactly one input and proves the same decision stops cutting, so a passing
   test means "checked and matched" rather than "saw nothing".

The lab-evidence obligation gets its own class. Rule 24(b)'s reader is
fail-closed with a single verdict; the train has to tell a reader WHICH way the
premise failed, because a missing receipt is a question for the rebuild trigger
and a FAIL receipt is a question for the lane. Collapsing them is the defect
OMN-18573 removed one layer down, and it is not reintroduced here.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_POLICY = _REPO_ROOT / "config" / "release_train_policy.yaml"


def _load() -> Any:
    """Import the module under test by path, as its siblings in this dir do."""
    name = "release_train_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load {_MODULE}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load()


@pytest.fixture(autouse=True)
def _green_ci_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test that is not ABOUT the CI premise gets a green one.

    Without this the default seams would reach the network, so a unit test
    would silently depend on live branch protection. Tests that exercise the
    premise pass their own seam explicitly and are unaffected, and the
    "one input flipped" tests below prove the green default is not masking it.
    """
    monkeypatch.setattr(
        rt, "default_required_contexts", lambda repo, branch: ["CI Summary"]
    )
    monkeypatch.setattr(rt, "default_gating_sha", lambda repo, sha: "g" * 40)
    monkeypatch.setattr(
        rt,
        "default_check_runs",
        lambda repo, sha: [
            {"name": "CI Summary", "status": "completed", "conclusion": "success"}
        ],
    )


# --------------------------------------------------------------------------- #
# Fixtures modelled on the live 2026-09-17 readings.                            #
# --------------------------------------------------------------------------- #
_INFRA_SHA = "2a4e49511d421c368adb5097131c5d1b985272c3"
_RECEIPTED_SHA = "3a8802c89f1a768894e8ed156158a2f3dcfd8f2f"


def _policy(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "repo": "omnibase_infra",
        "package": "omnibase-infra",
        "default_branch": "dev",
        "release_relevant_paths": ("src", "pyproject.toml"),
        "lab_evidence": rt.EnumLabEvidence.COMPOSE_DEV,
        "lab_evidence_note": "the compose dev lane on the lab host",
        "mode": rt.EnumTrainMode.CUT,
        "mode_note": "",
    }
    fields.update(overrides)
    return rt.ModelRepoReleasePolicy(**fields)


def _facts(**overrides: Any) -> Any:
    fields: dict[str, Any] = {
        "repo": "omnibase_infra",
        "latest_tag": "v0.38.30",
        "dev_version": "0.38.31",
        "dev_head_sha": _INFRA_SHA,
        "unreleased_count": 4,
        "unreleased_subjects": ("feat(OMN-18567): schedule the convergence (#3683)",),
    }
    fields.update(overrides)
    return rt.ModelRepoFacts(**fields)


def _receipt(sha: str, result: Any, *, lane: Any = None) -> Any:
    """A minimal valid lab-pass receipt, built through the real model."""
    lab = rt.lab_pass_receipt
    checks = (
        lab.ModelLabPassCheck(
            name="ready_main",
            ok=result is lab.EnumLabPassResult.PASS,
            evidence="probe evidence",
        ),
    )
    return lab.ModelLabPassReceipt(
        sha=sha,
        lane=lane or lab.EnumLabLane.COMPOSE_DEV,
        started_at=datetime(2026, 9, 17, 14, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 17, 14, 5, tzinfo=UTC),
        result=result,
        checks=checks,
        agent_command_id="cmd-1",
    )


def _lab_seam(
    *,
    artifacts: list[dict[str, Any]] | None = None,
    receipt: Any = None,
    list_error: Exception | None = None,
    download_error: Exception | None = None,
) -> dict[str, Any]:
    def _list(repo: str, name: str) -> list[dict[str, Any]]:
        if list_error is not None:
            raise list_error
        return artifacts if artifacts is not None else []

    def _download(repo: str, artifact_id: int) -> Any:
        if download_error is not None:
            raise download_error
        return receipt

    return {"list_artifacts": _list, "download_receipt": _download}


# --------------------------------------------------------------------------- #
# The live state on 2026-09-17.                                                 #
# --------------------------------------------------------------------------- #
class TestLiveState:
    def test_live_infra_has_no_unreleased_release_relevant_work_and_says_so(
        self,
    ) -> None:
        """`git rev-list --count v0.38.30..origin/dev -- src pyproject.toml` => 0.

        The four commits on dev are scripts, docs and workflows. The train must
        report that reason by name and must not cut.
        """
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(unreleased_count=0, unreleased_subjects=()),
            **_lab_seam(),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.NO_UNRELEASED_RELEASE_RELEVANT_WORK
        assert decision.repo == "omnibase_infra"
        assert decision.latest_tag == "v0.38.30"

    def test_live_omniintelligence_backlog_is_reported_not_cut(self) -> None:
        """67 unreleased commits since a tag dated 2026-05-26, report-only.

        Cutting a patch release over three months of accumulated change is the
        hazard OMN-18010 declared out of scope. The train still reports it every
        night, which is the whole point of a reported skip.
        """
        decision = rt.decide(
            policy=_policy(
                repo="omniintelligence",
                package="omniintelligence",
                lab_evidence=rt.EnumLabEvidence.NONE,
                lab_evidence_note="no lab lane of its own; no rebuild trigger",
                mode=rt.EnumTrainMode.REPORT_ONLY,
                mode_note="67 unreleased commits need a deliberate minor release first",
            ),
            facts=_facts(
                repo="omniintelligence",
                latest_tag="v0.24.0",
                dev_version="0.24.0",
                unreleased_count=67,
            ),
            **_lab_seam(),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.MODE_REPORT_ONLY
        assert decision.unreleased_count == 67
        assert "deliberate minor release" in decision.detail

    def test_the_shipped_policy_no_longer_records_the_cascade_as_blocked(
        self,
    ) -> None:
        """omniintelligence's entry recorded a cascade blocker that has cleared.

        The entry named OMN-18634 -- uv lock could not resolve this repo's
        dependency graph, so the cascade had never moved its omnibase-infra pin
        off 0.38.21 -- as one of two reasons the repo is report_only. That
        blocker cleared on 2026-09-19 (omniintelligence#918, squash 274f097f;
        proof run 35412730230, job "Cascade / Bump omniintelligence", success).
        A policy note that keeps describing a cleared blocker as open is a
        reader telling the next lane not to look.

        The second blocker is a different fact and is NOT cleared, so the mode
        stays report_only and this test asserts the surviving reason is still
        stated -- a note that lost both halves would read as armed-and-silent.

        A negative assertion on absent strings passes just as happily against a
        typo in the matcher, so the pre-fix text is carried here as the positive
        control: every marker this test demands be gone has to be findable in
        the text it was written against.
        """
        markers = ("OMN-18634", "uv lock cannot resolve", "0.38.21")
        pre_fix_note = (
            "TWO blockers. OMN-18634: uv lock cannot resolve this repo's Python "
            "3.15 / win32 marker split, so the dependency cascade has never "
            "moved its omnibase-infra pin off 0.38.21 and a release here could "
            "not propagate"
        )
        for marker in markers:
            assert marker in pre_fix_note, (
                f"positive control failed: {marker!r} is not in the pre-fix "
                "note, so this test would report a clean absence against any "
                "text at all"
            )

        note = rt.load_policy(_POLICY)["omniintelligence"].mode_note
        for marker in markers:
            assert marker not in note, (
                f"omniintelligence's mode_note still carries {marker!r}; the "
                "cascade blocker it describes cleared on 2026-09-19 and the "
                "entry is the only durable record a later reader has"
            )
        assert "deliberate minor release" in note, (
            "the surviving blocker -- the unreleased-commit backlog that makes "
            "a patch cut the wrong shape -- must stay stated, because it is "
            "what still holds this repo at report_only"
        )

    def test_the_shipped_policy_file_parses_and_covers_the_seven_repos(self) -> None:
        policies = rt.load_policy(_POLICY)
        assert set(policies) == {
            "omnibase_core",
            "omnibase_infra",
            "omnibase_spi",
            "omnimemory",
            "omniintelligence",
            "omnimarket",
            "omniclaude",
        }
        assert "omnibase_compat" not in policies, (
            "omnibase_compat is excluded by operator ruling and must not appear"
        )

    def test_exactly_the_three_reviewed_repos_are_armed_to_cut(self) -> None:
        """Arming a repo is one field, and this test is what keeps it a
        deliberate, visible act rather than a side effect of another change.

        The set grew from one under the operator's roll-out ruling of
        2026-09-17. Each addition was verified live BEFORE arming, and the
        evidence is written into that repo's own policy entry rather than left
        in a session that is now gone: main is an ancestor of dev, so the
        release fast-forward can succeed, and every required context on dev
        reported success for the candidate's gating commit.

        omnimemory IS here, and an earlier revision of this docstring said it
        was not. That exclusion rested on a skipped required context which was
        this reader's own bug twice over -- an arbitrary pick among check runs
        sharing a name, then a fetch that projected away the timestamps the fix
        ordered on. Both are corrected and pinned by their own tests, and every
        required context on that repo's gating commit reports success.
        """
        policies = rt.load_policy(_POLICY)
        armed = sorted(
            name for name, p in policies.items() if p.mode is rt.EnumTrainMode.CUT
        )
        assert armed == [
            "omnibase_core",
            "omnibase_infra",
            "omnibase_spi",
            "omnimemory",
        ]

    def test_every_armed_repo_states_the_premise_it_cuts_on(self) -> None:
        """A cut taken on a premise nobody wrote down is a cut nobody can audit.

        Asserted on the policy entry, because the entry is what outlives the
        session that armed it.
        """
        policies = rt.load_policy(_POLICY)
        for name, policy in sorted(policies.items()):
            if policy.mode is not rt.EnumTrainMode.CUT:
                continue
            if policy.lab_evidence is rt.EnumLabEvidence.NONE:
                assert "green dev CI" in policy.mode_note, (
                    f"{name} is armed to cut with no lab surface and its policy "
                    "entry does not name the premise it cuts on"
                )
                assert "ancestor of dev" in policy.mode_note, (
                    f"{name} is armed to cut without its ancestry stated; a cut "
                    "whose fast-forward cannot succeed fails at the last step"
                )

    def test_every_repo_declaring_no_lab_surface_carries_its_reason(self) -> None:
        policies = rt.load_policy(_POLICY)
        for name, policy in policies.items():
            if policy.lab_evidence is rt.EnumLabEvidence.NONE:
                assert policy.lab_evidence_note.strip(), (
                    f"{name} declares no lab evidence with no reason; the "
                    "declaration is the record of the gap, so an empty one "
                    "records nothing"
                )


# --------------------------------------------------------------------------- #
# The cut path, and a discriminating sibling for each input.                    #
# --------------------------------------------------------------------------- #
class TestCutPath:
    def test_unreleased_work_with_a_passing_receipt_cuts(self) -> None:
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.PASS),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert decision.reason is rt.EnumTrainReason.UNRELEASED_WORK_LAB_PROVEN
        assert decision.candidate_version == "0.38.31"
        assert decision.candidate_sha == _INFRA_SHA

    def test_the_same_inputs_without_unreleased_work_do_not_cut(self) -> None:
        """One input flipped. Proves the cut above is not vacuous."""
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(unreleased_count=0),
            **_lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.PASS),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP

    def test_a_dev_version_level_with_the_tag_cuts_the_next_patch(self) -> None:
        """The one-ahead invariant does not always hold; today it does not.

        omnibase_infra sat at dev 0.38.30 against tag v0.38.30 while its
        post-release bump PR was open. The train has to bump rather than assume.
        """
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(dev_version="0.38.30"),
            **_lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.PASS),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert decision.needs_bump is True
        assert decision.candidate_version == "0.38.31"

    def test_a_dev_version_ahead_of_the_tag_needs_no_bump(self) -> None:
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(dev_version="0.38.31"),
            **_lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.PASS),
            ),
        )
        assert decision.needs_bump is False
        assert decision.candidate_version == "0.38.31"

    def test_a_declared_absent_lab_surface_cuts_and_names_the_declaration(
        self,
    ) -> None:
        """The stated deviation, exercised only where a policy declares it.

        No repo is armed this way today. The path exists so that arming one is a
        policy edit whose consequence is already tested, not new code written in
        a hurry against a live backlog.
        """
        decision = rt.decide(
            policy=_policy(
                repo="omnibase_spi",
                package="omnibase-spi",
                lab_evidence=rt.EnumLabEvidence.NONE,
                lab_evidence_note="library repo; no lab lane and no rebuild trigger",
            ),
            facts=_facts(repo="omnibase_spi", latest_tag="v0.23.3", unreleased_count=1),
            **_lab_seam(),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert (
            decision.reason
            is rt.EnumTrainReason.UNRELEASED_WORK_NO_LAB_SURFACE_DECLARED
        )
        assert "no lab lane" in decision.detail


# --------------------------------------------------------------------------- #
# AC5: every way the lab premise fails is its own outcome, and none cuts.       #
# --------------------------------------------------------------------------- #
class TestLabEvidenceIsNotCollapsed:
    def test_no_receipt_artifact_is_its_own_reason(self) -> None:
        decision = rt.decide(policy=_policy(), facts=_facts(), **_lab_seam())
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT

    def test_a_failing_receipt_is_its_own_reason(self) -> None:
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_lab_seam(
                artifacts=[{"id": 2, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.FAIL),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_FAIL

    def test_a_name_payload_disagreement_is_its_own_reason(self) -> None:
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_lab_seam(
                artifacts=[{"id": 3, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_RECEIPTED_SHA, lab.EnumLabPassResult.PASS),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_NAME_PAYLOAD_DISAGREE
        assert _RECEIPTED_SHA in decision.detail

    def test_an_unreadable_surface_is_its_own_reason(self) -> None:
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_lab_seam(list_error=RuntimeError("gh api exited 1: Bad credentials")),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_UNREADABLE
        assert "Bad credentials" in decision.detail

    def test_an_undownloadable_receipt_is_unreadable_not_absent(self) -> None:
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_lab_seam(
                artifacts=[{"id": 4, "created_at": "2026-09-17T14:07:37Z"}],
                download_error=RuntimeError("artifact 4 is not a readable zip"),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_UNREADABLE

    def test_the_four_failure_modes_produce_four_distinct_reasons(self) -> None:
        """The criterion itself, asserted as one fact rather than four.

        Four different questions for four different owners. A single collapsed
        reason sends every one of them to the wrong place.
        """
        lab = rt.lab_pass_receipt
        seams = [
            _lab_seam(),
            _lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.FAIL),
            ),
            _lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_RECEIPTED_SHA, lab.EnumLabPassResult.PASS),
            ),
            _lab_seam(list_error=RuntimeError("boom")),
        ]
        reasons = {
            rt.decide(policy=_policy(), facts=_facts(), **seam).reason for seam in seams
        }
        assert len(reasons) == 4
        verdicts = {
            rt.decide(policy=_policy(), facts=_facts(), **seam).verdict
            for seam in seams
        }
        assert verdicts == {rt.EnumTrainVerdict.SKIP}

    def test_the_newest_receipt_wins_when_a_sha_has_several(self) -> None:
        """A re-run of the emitting job supersedes an earlier attempt.

        Same ordering rule the rule-24(b) reader uses one layer down. Without it
        a re-run that fixed a lane would still be judged on the stale attempt.
        """
        lab = rt.lab_pass_receipt
        seen: list[int] = []

        def _download(repo: str, artifact_id: int) -> Any:
            seen.append(artifact_id)
            return _receipt(_INFRA_SHA, lab.EnumLabPassResult.PASS)

        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            list_artifacts=lambda repo, name: [
                {"id": 10, "created_at": "2026-09-17T09:00:00Z"},
                {"id": 20, "created_at": "2026-09-17T14:07:37Z"},
            ],
            download_receipt=_download,
        )
        assert seen == [20]
        assert decision.verdict is rt.EnumTrainVerdict.CUT


# --------------------------------------------------------------------------- #
# Configuration errors refuse; they are never reported as "nothing to do".      #
# --------------------------------------------------------------------------- #
class TestConfigurationErrorsRefuse:
    def test_an_unreadable_dev_version_refuses(self) -> None:
        decision = rt.decide(
            policy=_policy(), facts=_facts(dev_version="0.38.31rc1"), **_lab_seam()
        )
        assert decision.verdict is rt.EnumTrainVerdict.REFUSE
        assert decision.reason is rt.EnumTrainReason.VERSION_UNREADABLE

    def test_a_repo_missing_from_the_policy_refuses_rather_than_defaulting(
        self, tmp_path: Path
    ) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "repos:\n"
            "  omnibase_infra:\n"
            "    package: omnibase-infra\n"
            "    default_branch: dev\n"
            "    release_relevant_paths: [src, pyproject.toml]\n"
            "    lab_evidence: compose-dev\n"
            "    lab_evidence_note: the compose dev lane\n"
            "    mode: cut\n",
            encoding="utf-8",
        )
        policies = rt.load_policy(policy_file)
        with pytest.raises(rt.ReleaseTrainConfigError) as excinfo:
            rt.policy_for(policies, "omnimemory")
        assert "omnimemory" in str(excinfo.value)

    def test_a_none_lab_evidence_declaration_without_a_reason_refuses_at_load(
        self, tmp_path: Path
    ) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "repos:\n"
            "  omnimemory:\n"
            "    package: omnimemory\n"
            "    default_branch: dev\n"
            "    release_relevant_paths: [src]\n"
            "    lab_evidence: none\n"
            "    lab_evidence_note: ''\n"
            "    mode: report_only\n"
            "    mode_note: pending an operator decision\n",
            encoding="utf-8",
        )
        with pytest.raises(rt.ReleaseTrainConfigError):
            rt.load_policy(policy_file)

    def test_a_report_only_declaration_without_a_reason_refuses_at_load(
        self, tmp_path: Path
    ) -> None:
        policy_file = tmp_path / "policy.yaml"
        policy_file.write_text(
            "repos:\n"
            "  omnimemory:\n"
            "    package: omnimemory\n"
            "    default_branch: dev\n"
            "    release_relevant_paths: [src]\n"
            "    lab_evidence: compose-dev\n"
            "    lab_evidence_note: the compose dev lane\n"
            "    mode: report_only\n"
            "    mode_note: ''\n",
            encoding="utf-8",
        )
        with pytest.raises(rt.ReleaseTrainConfigError):
            rt.load_policy(policy_file)


# --------------------------------------------------------------------------- #
# The report is durable and complete.                                           #
# --------------------------------------------------------------------------- #
class TestReport:
    def test_the_report_carries_every_repo_and_its_reason(self) -> None:
        decisions = [
            rt.decide(
                policy=_policy(), facts=_facts(unreleased_count=0), **_lab_seam()
            ),
            rt.decide(
                policy=_policy(
                    repo="omniclaude",
                    package="omniclaude",
                    lab_evidence=rt.EnumLabEvidence.NONE,
                    lab_evidence_note="no lab lane",
                    mode=rt.EnumTrainMode.REPORT_ONLY,
                    mode_note="65 unreleased commits need a deliberate minor release",
                ),
                facts=_facts(
                    repo="omniclaude", latest_tag="v0.25.1", dev_version="0.25.1"
                ),
                **_lab_seam(),
            ),
        ]
        payload = json.loads(rt.render_report_json(decisions))
        assert [row["repo"] for row in payload["decisions"]] == [
            "omnibase_infra",
            "omniclaude",
        ]
        for row in payload["decisions"]:
            assert row["reason"], "a decision with no reason reports nothing"
            assert row["verdict"] in {"CUT", "SKIP", "REFUSE"}
        assert payload["cut_count"] == 0

    def test_the_human_summary_names_the_sha_for_every_row(self) -> None:
        """A verdict with no commit named is unactionable at 3am."""
        decision = rt.decide(
            policy=_policy(), facts=_facts(unreleased_count=0), **_lab_seam()
        )
        text = rt.render_report_human([decision])
        assert _INFRA_SHA[:12] in text
        assert "omnibase_infra" in text
        assert decision.reason.value in text


# --------------------------------------------------------------------------- #
# The workflow's own shape, asserted so it cannot drift back.                   #
# --------------------------------------------------------------------------- #
class TestWorkflowShape:
    WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"

    def test_the_workflow_exists_and_is_scheduled(self) -> None:
        body = self.WORKFLOW.read_text(encoding="utf-8")
        assert "schedule:" in body
        assert "workflow_dispatch:" in body

    def test_the_workflow_never_falls_back_to_the_default_token(self) -> None:
        """AC6. An App-token mint that silently degrades is the OMN-18273 confound.

        A job that pushes as the default workflow token has its downstream
        workflows suppressed, so the release PR would open and nothing would
        ever tag it. The mint must fail the job instead.
        """
        body = self.WORKFLOW.read_text(encoding="utf-8")
        assert "secrets.GITHUB_TOKEN" not in body, (
            "the release train must never authenticate a write as the default "
            "workflow token; mint the App token and fail closed if it cannot"
        )
        assert "|| secrets." not in body, (
            "a fallback expression substituting another credential for the "
            "App token reintroduces the confound OMN-18273 corrected"
        )

    def test_the_workflow_uploads_its_decision_under_always(self) -> None:
        body = self.WORKFLOW.read_text(encoding="utf-8")
        assert "upload-artifact" in body
        assert "if: always()" in body, (
            "a decision artifact that appears only on success cannot "
            "distinguish a failed decision from an unrun one"
        )

    def test_the_workflow_declares_no_option_that_forces_a_cut(self) -> None:
        """There is no override. A premise is met or the train does not cut.

        Asserted against the workflow's DECLARED INPUTS rather than against its
        text. A text match cannot tell an override apart from a comment saying
        there is no override, which is the same class of mistake as a gate that
        fires on prose describing the gate.
        """
        import yaml as _yaml

        parsed = _yaml.safe_load(self.WORKFLOW.read_text(encoding="utf-8"))
        # `on` parses as the boolean True in YAML 1.1, which is how GitHub
        # workflow files are read by every YAML 1.1 loader.
        triggers = parsed.get("on", parsed.get(True, {}))
        inputs = (triggers.get("workflow_dispatch") or {}).get("inputs") or {}
        assert set(inputs) == {"repos"}, (
            "the release train declares an input other than the repo filter; "
            "there is no override, and adding one is this test going red"
        )
        assert inputs["repos"].get("default", "") == "", (
            "the repo filter must default to every declared repo, so the "
            "scheduled run decides the whole fleet"
        )

    def test_the_workflow_opens_the_release_pr_against_the_decided_base(
        self,
    ) -> None:
        """The base branch is a policy fact the decision carries, not a literal.

        config/release_train_policy.yaml declares `default_branch` per repo
        precisely because the train fans out across repositories, and a branch
        name baked into the automation is correct only for as long as every
        repository agrees. That is the defect OMN-18588 corrected elsewhere in
        this repo on 2026-09-17, in automation that cloned every repo at a
        hardcoded `dev`. Landing a fresh copy of it here would be a regression
        of a lesson this repo learned the same day.
        """
        body = self.WORKFLOW.read_text(encoding="utf-8")
        assert "--base dev" not in body, (
            "the release train hardcodes the base branch; it must open the "
            "release PR against the branch the decision carries, which is the "
            "repo's own declared default_branch"
        )
        assert "matrix.decision.base_branch" in body, (
            "the cut job must take its base branch from the decision row, so a "
            "repo declaring a different default_branch is honoured rather than "
            "silently retargeted"
        )

    def test_every_decision_row_carries_its_base_branch(self) -> None:
        """The workflow can only read a field the decision actually emits.

        Asserted on a repo whose declared default_branch is NOT `dev`, so the
        field is proven to carry the declaration rather than a constant that
        happens to agree with it today.
        """
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(default_branch="trunk"),
            facts=_facts(),
            **_lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.PASS),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert decision.to_dict()["base_branch"] == "trunk"


# --------------------------------------------------------------------------- #
# The train must not re-enter on its own bookkeeping.                           #
# --------------------------------------------------------------------------- #
class TestReleaseBookkeepingIsNotUnreleasedWork:
    """The post-release bump edits pyproject.toml, which is a packaged path.

    Counted naively it is one unreleased release-relevant commit the morning
    after every release, so the train would cut an empty release every night
    forever. This is the same self-re-entry the release-on-merge decision guards
    with its own subject marker; the shapes differ because the triggers differ.
    """

    def test_the_post_release_bump_subject_is_bookkeeping(self) -> None:
        assert rt.is_release_bookkeeping_subject(
            "chore(OMN-13912): post-release dev version bump to 0.38.30 (#3677)"
        )

    def test_a_release_pr_subject_is_bookkeeping(self) -> None:
        assert rt.is_release_bookkeeping_subject(
            "chore: release omnibase-infra 0.38.30 (OMN-18565) (#3691)"
        )

    def test_an_ordinary_fix_is_not_bookkeeping(self) -> None:
        """The discriminating half: the marker must not swallow real work."""
        assert not rt.is_release_bookkeeping_subject(
            "fix(OMN-18565): the kernel projection seam carries the envelope tenant (#3679)"
        )

    def test_a_subject_merely_mentioning_a_release_is_not_bookkeeping(self) -> None:
        assert not rt.is_release_bookkeeping_subject(
            "docs(OMN-18010): describe how the release train decides to cut (#1)"
        )


# --------------------------------------------------------------------------- #
# The workflow must not hand-roll a second loader for the module it ships with. #
# --------------------------------------------------------------------------- #
class TestTheDeclaredRepoListingComesFromTheCLI:
    """Run 35247876101 is why this class exists.

    The first dispatched run of the train died in its very first step. The
    workflow listed the declared repos by loading ``release_train.py`` through
    an inline ``importlib`` heredoc, and that copy omitted the
    ``sys.modules[name] = module`` registration this test file's own loader has.
    Without it ``@dataclass`` raises ``AttributeError: 'NoneType' object has no
    attribute '__dict__'`` at import, because ``dataclasses`` resolves the
    defining module out of ``sys.modules`` and finds nothing there.

    The lesson is not "add the missing line". Two loaders for one module is the
    defect; the tests passed the whole time precisely because the loader under
    test was the correct one and the shipped one was never exercised. So the
    module grows a subcommand, the workflow calls it like any other CLI, and
    the second loader stops existing.
    """

    WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"

    def test_the_declared_subcommand_lists_every_declared_repo(self) -> None:
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = rt.main(["declared", "--policy", str(_POLICY)])
        assert code == 0
        listed = buffer.getvalue().split()
        assert listed == sorted(listed), "listed in a stable order, not dict order"
        assert set(listed) == set(rt.load_policy(_POLICY)), (
            "the subcommand and the policy loader must agree on the declared set; "
            "a listing that drifts from the policy silently drops a repo from "
            "every nightly decision"
        )

    def test_the_subcommand_runs_as_a_subprocess_the_way_the_workflow_calls_it(
        self,
    ) -> None:
        """The reproduction of run 35247876101, at the boundary it failed on.

        Invoked as a real child process, so the module is imported exactly as
        the runner imports it. The in-process test above cannot catch the
        original defect at all: by the time it runs, the module is already in
        ``sys.modules`` because this file put it there.
        """
        import subprocess

        result = subprocess.run(
            [sys.executable, str(_MODULE), "declared", "--policy", str(_POLICY)],
            capture_output=True,
            text=True,
            check=False,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        assert set(result.stdout.split()) == set(rt.load_policy(_POLICY))

    def test_the_workflow_does_not_hand_roll_a_module_loader(self) -> None:
        body = self.WORKFLOW.read_text(encoding="utf-8")
        assert "spec_from_file_location" not in body, (
            "the workflow hand-rolls a loader for a module it ships beside; that "
            "second copy is what died in run 35247876101. Call the CLI instead"
        )
        assert "release_train.py declared" in body, (
            "the workflow must list the declared repos through the module's own "
            "subcommand, so the listing is exercised by the same tests as the "
            "rest of the module"
        )


# --------------------------------------------------------------------------- #
# The green-CI premise (OMN-18595, second pass).                                #
# --------------------------------------------------------------------------- #
def _ci_seam(
    *,
    required: list[str] | None = None,
    runs: list[dict[str, Any]] | None = None,
    required_error: Exception | None = None,
    runs_error: Exception | None = None,
    gate: str | None = None,
    gate_error: Exception | None = None,
) -> dict[str, Any]:
    def _required(repo: str, branch: str) -> list[str]:
        if required_error is not None:
            raise required_error
        return list(required if required is not None else ["CI Summary"])

    def _runs(repo: str, sha: str) -> list[dict[str, Any]]:
        if runs_error is not None:
            raise runs_error
        if runs is not None:
            return runs
        return [{"name": "CI Summary", "status": "completed", "conclusion": "success"}]

    def _gate(repo: str, sha: str) -> str:
        if gate_error is not None:
            raise gate_error
        return gate if gate is not None else "d8ab1e719e0a" + "0" * 28

    return {
        "required_contexts": _required,
        "check_runs": _runs,
        "gating_sha": _gate,
    }


class TestTheGreenCIPremiseGuardsEveryCutPath:
    """The premise this file's first pass did not have at all.

    Before it, a repo declaring `lab_evidence: none` -- five of the seven --
    would CUT on unreleased commits alone: no lab receipt, and no CI check of
    any kind. That is materially weaker than the design described, and weaker
    than anyone would assume from the phrase "the cut path".
    """

    def test_a_no_lab_surface_repo_cuts_only_when_ci_is_green(self) -> None:
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert "required context" in decision.detail, (
            "a cut taken on the CI premise must SAY what it proved, or the "
            "decision record cannot be told apart from one taken on nothing"
        )

    def test_the_same_repo_does_not_cut_when_a_required_context_failed(self) -> None:
        """One input flipped. Proves the cut above is not vacuous."""
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(
                runs=[
                    {
                        "name": "CI Summary",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ]
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.CI_NOT_GREEN

    def test_a_skipped_required_context_is_its_own_refusal(self) -> None:
        """The trap. GitHub treats a skipped REQUIRED context as satisfying
        protection, so the commit is mergeable while that gate never ran. A
        train that folded this into "not green" would still be right to refuse,
        but nobody reading the artifact would learn why."""
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(
                runs=[
                    {
                        "name": "CI Summary",
                        "status": "completed",
                        "conclusion": "skipped",
                    }
                ]
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.CI_REQUIRED_CONTEXT_SKIPPED

    def test_a_required_context_absent_from_the_sha_refuses(self) -> None:
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(runs=[]),
        )
        assert decision.reason is rt.EnumTrainReason.CI_REQUIRED_CONTEXT_MISSING

    def test_a_still_running_required_context_refuses(self) -> None:
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(
                runs=[
                    {"name": "CI Summary", "status": "in_progress", "conclusion": None}
                ]
            ),
        )
        assert decision.reason is rt.EnumTrainReason.CI_REQUIRED_CONTEXT_PENDING

    def test_an_empty_required_set_is_not_green(self) -> None:
        """An ungated branch is not a passing one, and reading it as green would
        make this premise vacuous exactly where it matters most."""
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(required=[]),
        )
        assert decision.reason is rt.EnumTrainReason.CI_PROTECTION_UNREADABLE

    def test_an_unreadable_protection_read_refuses(self) -> None:
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(required_error=RuntimeError("404")),
        )
        assert decision.reason is rt.EnumTrainReason.CI_PROTECTION_UNREADABLE

    def test_an_unreadable_check_run_read_refuses(self) -> None:
        decision = rt.decide(
            policy=_policy(lab_evidence=rt.EnumLabEvidence.NONE, lab_evidence_note="n"),
            facts=_facts(),
            **_lab_seam(),
            **_ci_seam(runs_error=RuntimeError("boom")),
        )
        assert decision.reason is rt.EnumTrainReason.CI_PROTECTION_UNREADABLE

    def test_the_premise_also_guards_the_lab_receipt_arm(self) -> None:
        """A premise that guarded only the no-lab arm would leave the repo that
        HAS a lab lane cutting without it."""
        lab = rt.lab_pass_receipt
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_lab_seam(
                artifacts=[{"id": 1, "created_at": "2026-09-17T14:07:37Z"}],
                receipt=_receipt(_INFRA_SHA, lab.EnumLabPassResult.PASS),
            ),
            **_ci_seam(
                runs=[
                    {
                        "name": "CI Summary",
                        "status": "completed",
                        "conclusion": "failure",
                    }
                ]
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.CI_NOT_GREEN

    def test_the_premise_is_read_against_the_candidate_sha_and_branch(self) -> None:
        """Not against "the repo" in the abstract.

        A premise resolved against some other ref would pass while the commit
        being cut had never been gated at all.
        """
        seen: list[tuple[str, str]] = []
        gate_asked: list[tuple[str, str]] = []
        GATE = "d8ab1e719e0a" + "0" * 28

        def _gate(repo: str, sha: str) -> str:
            gate_asked.append((repo, sha))
            return GATE

        def _runs(repo: str, sha: str) -> list[dict[str, Any]]:
            seen.append((repo, sha))
            return [
                {"name": "CI Summary", "status": "completed", "conclusion": "success"}
            ]

        branches: list[tuple[str, str]] = []

        def _required(repo: str, branch: str) -> list[str]:
            branches.append((repo, branch))
            return ["CI Summary"]

        rt.decide(
            policy=_policy(
                default_branch="trunk",
                lab_evidence=rt.EnumLabEvidence.NONE,
                lab_evidence_note="n",
            ),
            facts=_facts(dev_head_sha=_INFRA_SHA),
            **_lab_seam(),
            required_contexts=_required,
            check_runs=_runs,
            gating_sha=_gate,
        )
        # The gating sha is resolved FROM the candidate, and the check runs are
        # read against the GATING sha, not the candidate. Asking the post-merge
        # commit for a PR-time gate is unsatisfiable by construction.
        assert gate_asked == [("omnibase_infra", _INFRA_SHA)]
        assert seen == [("omnibase_infra", GATE)]
        assert branches == [("omnibase_infra", "trunk")]


class TestThereIsNoWayToAssertTheCIFact:
    def test_no_entrypoint_declares_an_option_that_asserts_ci(self) -> None:
        """Same shape the health fact has on the k3s prod gate.

        Asserted against the parser's own declared option strings, so adding a
        bypass later is a red test rather than a review catch.
        """
        options: list[str] = []
        for action in rt.build_parser()._actions:
            options.extend(action.option_strings)
        for sub in rt.build_parser()._subparsers._group_actions:
            for parser in getattr(sub, "choices", {}).values():
                for action in parser._actions:
                    options.extend(action.option_strings)
        forbidden = [
            o
            for o in options
            if any(
                t in o.lower() for t in ("ci-status", "ci-green", "skip-ci", "force")
            )
        ]
        assert not forbidden, (
            f"the train declares {forbidden}, which would let a caller assert or "
            "bypass the CI fact it is supposed to resolve itself"
        )


# --------------------------------------------------------------------------- #
# Version source: two manifests, one answer (OMN-18595 roll-out).               #
# --------------------------------------------------------------------------- #
class TestTheVersionSourceIsDiscoveredNotAssumed:
    """The fleet is not all Python.

    omnidash and omniweb carry a package.json and no pyproject.toml, so a train
    that could only read ``[project].version`` would refuse them every night
    with version_unreadable -- fail-closed, but pure noise.

    Fixtures are real git repositories rather than mocks, because the thing
    under test is how the reader behaves against `git show` on a ref, and a
    mock of `git show` would be a mock of the exact surface that broke.
    """

    @staticmethod
    def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
        import os
        import subprocess

        from omnibase_core.validators.no_unguarded_git_subprocess import (
            scrub_git_location_env,
        )

        root = tmp_path / "repo"
        root.mkdir()

        def git(*args: str) -> None:
            # The scrub is INLINE, not hoisted into a variable: the OMN-14891
            # guard reads the call site, and a hoisted binding reads to it as an
            # ambient environment. Keeping it here means the guard can see it.
            subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                env=scrub_git_location_env(os.environ),
            )

        git("init", "--initial-branch", "dev")
        git("config", "user.email", "t@example.invalid")
        git("config", "user.name", "t")
        for name, body in files.items():
            (root / name).write_text(body, encoding="utf-8")
            git("add", name)
        git("commit", "-m", "init")
        # collect_repo_facts reads origin/<branch>, so give it one.
        git("update-ref", "refs/remotes/origin/dev", "HEAD")
        return root

    def test_a_python_repo_still_reads_its_project_version(
        self, tmp_path: Path
    ) -> None:
        root = self._repo(
            tmp_path, {"pyproject.toml": '[project]\nversion = "1.2.3"\n'}
        )
        assert rt.resolve_declared_version("r", root, "origin/dev") == "1.2.3"

    def test_a_node_repo_reads_its_package_json_version(self, tmp_path: Path) -> None:
        """The omnidash fixture: real shape, real version."""
        root = self._repo(
            tmp_path, {"package.json": '{"name":"omnidash","version":"1.1.3"}'}
        )
        assert rt.resolve_declared_version("omnidash", root, "origin/dev") == "1.1.3"

    def test_the_omniweb_fixture_reads_too(self, tmp_path: Path) -> None:
        root = self._repo(
            tmp_path, {"package.json": '{"name":"omniweb","version":"0.1.0"}'}
        )
        assert rt.resolve_declared_version("omniweb", root, "origin/dev") == "0.1.0"

    def test_neither_manifest_refuses(self, tmp_path: Path) -> None:
        """A repo whose version cannot be read is a repo whose next release is
        already broken; reporting that as "nothing to do" is the original
        disease this train was built against."""
        root = self._repo(tmp_path, {"README.md": "hi\n"})
        with pytest.raises(rt.ReleaseTrainConfigError, match="neither pyproject"):
            rt.resolve_declared_version("r", root, "origin/dev")

    def test_both_manifests_refuse_rather_than_preferring_one(
        self, tmp_path: Path
    ) -> None:
        """A silent preference cuts whichever number the other manifest is not
        tracking. The two would drift and only one would ever be tagged."""
        root = self._repo(
            tmp_path,
            {
                "pyproject.toml": '[project]\nversion = "1.2.3"\n',
                "package.json": '{"name":"x","version":"9.9.9"}',
            },
        )
        with pytest.raises(rt.ReleaseTrainConfigError, match="BOTH"):
            rt.resolve_declared_version("r", root, "origin/dev")

    def test_a_versionless_package_json_refuses(self, tmp_path: Path) -> None:
        root = self._repo(tmp_path, {"package.json": '{"name":"x"}'})
        with pytest.raises(rt.ReleaseTrainConfigError, match="declares no version"):
            rt.resolve_declared_version("r", root, "origin/dev")

    def test_an_unparseable_package_json_refuses(self, tmp_path: Path) -> None:
        root = self._repo(tmp_path, {"package.json": "{not json"})
        with pytest.raises(rt.ReleaseTrainConfigError, match="does not parse"):
            rt.resolve_declared_version("r", root, "origin/dev")

    def test_an_unreadable_ref_raises_rather_than_reading_as_absent(
        self, tmp_path: Path
    ) -> None:
        """The distinction the helper exists for.

        git reports BOTH "path not in ref" and "ref is not a thing" as exit
        128, so an implementation keyed on the exit code alone would report a
        repo nobody could read as a repo with nothing to release.
        """
        root = self._repo(tmp_path, {"package.json": '{"name":"x","version":"1.0.0"}'})
        with pytest.raises(rt.ReleaseTrainConfigError):
            rt.resolve_declared_version("r", root, "origin/no-such-branch")


class TestADuplicatedContextResolvesToItsLatestRun:
    """Observed on omnimemory, 2026-09-17, and it nearly cost a real repo.

    The required context 'pr-title / check-title' carried FOUR check runs on its
    gating commit: two skipped and two success, two of them inside a SINGLE
    workflow run, because a conditional job emits a skipped leg beside the real
    one. Seven other required names on that same sha carried duplicates too, so
    this is the normal shape rather than an anomaly.

    A first-wins reader picked the skipped leg and reported the repo blocked by
    the very trap this premise exists to catch. That verdict was wrong, and a
    premise whose refusals cannot be trusted is worse than no premise: it
    teaches the next reader to route around it.
    """

    OBSERVED = [
        {
            "name": "pr-title / check-title",
            "status": "completed",
            "conclusion": "skipped",
            "started_at": "2026-09-17T13:43:59Z",
            "completed_at": "2026-09-17T13:43:59Z",
        },
        {
            "name": "pr-title / check-title",
            "status": "completed",
            "conclusion": "success",
            "started_at": "2026-09-17T13:44:03Z",
            "completed_at": "2026-09-17T13:44:08Z",
        },
    ]

    def test_the_later_success_wins_over_the_earlier_skip(self) -> None:
        reason, detail = rt.classify_ci_green(
            "omnimemory",
            "s" * 40,
            "dev",
            required_contexts=lambda repo, branch: ["pr-title / check-title"],
            check_runs=lambda repo, sha: list(self.OBSERVED),
            gating_sha=lambda repo, sha: "g" * 40,
        )
        assert reason is None, detail

    def test_input_order_does_not_change_the_verdict(self) -> None:
        """The bug was order-dependence, so the fix is asserted that way."""
        reason, _ = rt.classify_ci_green(
            "omnimemory",
            "s" * 40,
            "dev",
            required_contexts=lambda repo, branch: ["pr-title / check-title"],
            check_runs=lambda repo, sha: list(reversed(self.OBSERVED)),
            gating_sha=lambda repo, sha: "g" * 40,
        )
        assert reason is None

    def test_a_later_skip_still_refuses(self) -> None:
        """Latest-wins is not skip-blindness. If the newest run IS the skip, the
        context is skipped and the premise must still refuse."""
        newest_skipped = [
            dict(
                self.OBSERVED[1],
                conclusion="success",
                completed_at="2026-09-17T13:44:08Z",
            ),
            dict(
                self.OBSERVED[0],
                conclusion="skipped",
                completed_at="2026-09-17T13:45:00Z",
            ),
        ]
        reason, _ = rt.classify_ci_green(
            "omnimemory",
            "s" * 40,
            "dev",
            required_contexts=lambda repo, branch: ["pr-title / check-title"],
            check_runs=lambda repo, sha: newest_skipped,
            gating_sha=lambda repo, sha: "g" * 40,
        )
        assert reason is rt.EnumTrainReason.CI_REQUIRED_CONTEXT_SKIPPED

    def test_an_unfinished_run_sorts_last_and_is_not_masked(self) -> None:
        """A still-running leg must not be hidden by an older finished one."""
        pending_newest = [
            dict(self.OBSERVED[1]),
            {
                "name": "pr-title / check-title",
                "status": "in_progress",
                "conclusion": None,
                "started_at": "2026-09-17T13:50:00Z",
                "completed_at": None,
            },
        ]
        reason, _ = rt.classify_ci_green(
            "omnimemory",
            "s" * 40,
            "dev",
            required_contexts=lambda repo, branch: ["pr-title / check-title"],
            check_runs=lambda repo, sha: pending_newest,
            gating_sha=lambda repo, sha: "g" * 40,
        )
        assert reason is rt.EnumTrainReason.CI_REQUIRED_CONTEXT_PENDING


class TestTheCheckRunProjectionKeepsWhatTheResolverSortsOn:
    """The second half of the same bug, and the better-hidden half.

    Fixing first-wins to latest-wins is useless if the fetch projects the
    timestamps away: every run then looks equally recent and an arbitrary one
    still wins. That is exactly what happened -- the jq projection asked for
    name, status and conclusion only, so the corrected resolver kept returning
    the same wrong verdict and omnimemory still read as blocked by a gate that
    had in fact passed.

    Asserted on the shipped SOURCE rather than on behaviour, because the seam
    the other tests inject replaces the real fetch entirely and so can never
    see its projection.
    """

    SOURCE = _MODULE.read_text(encoding="utf-8")

    def test_the_fetch_requests_the_fields_recency_is_computed_from(self) -> None:
        jq_lines = [
            line
            for line in self.SOURCE.splitlines()
            if ".check_runs[]" in line and "{" in line
        ]
        assert jq_lines, "no check-runs projection found in the shipped source"
        for field in ("started_at", "completed_at"):
            assert all(field in line for line in jq_lines), (
                f"the check-runs projection does not fetch {field!r}, which the "
                "duplicate-context resolver orders on. Without it every run "
                "sorts equal and an arbitrary one wins, which is first-wins "
                "again with a longer fuse"
            )

    def test_the_resolver_orders_on_exactly_those_fields(self) -> None:
        """A rename on one side and not the other reintroduces it silently."""
        assert "_recency" in self.SOURCE
        recency = self.SOURCE[self.SOURCE.index("def _recency") :][:400]
        for field in ("completed_at", "started_at"):
            assert field in recency, (
                f"the recency key does not read {field!r}; the fetch and the "
                "resolver must agree on the field names or the ordering is "
                "computed from nothing"
            )


class TestTheDecideJobCanReadWhatThePremiseNeeds:
    """Found by the first real cut attempt, run 35286327829, not by review.

    The green-CI premise asks which contexts are REQUIRED on a repo's default
    branch. That is branch-protection metadata, and the only endpoint carrying
    it needs `administration: read`. Check runs are readable without it, but
    WHICH of them gate a merge is not.

    Without that permission the premise fails closed on every repo with
    `ci_protection_unreadable` -- the correct refusal, and also a train that can
    never cut anything. The unit tests could not catch it: they inject a seam
    that replaces the real fetch, so the token the real fetch would use is not
    exercised anywhere in them. These assertions are the only place the
    credential the real fetch runs under is checked at all, which is why they
    read the workflow rather than the module.

    Three facts have to hold together and each is its own test: the permission
    is requested, the App it is requested from is the one that carries it, and
    nothing writable is requested alongside it. Any one alone is satisfiable in
    a way that is wrong -- a request from an App without the grant fails the
    mint, and the right App with write scopes hands a read-only premise a token
    that could rewrite the gates it checks.
    """

    WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"

    WRITE_SCOPES = ("write", "admin")

    @staticmethod
    def _decide_mints(workflow: Path) -> list[dict[str, Any]]:
        """Every App-token mint declared by the decide job, parsed not grepped.

        Parsed because a text match cannot tell a request apart from a comment
        explaining why a request is absent, and this file carries several
        paragraphs of exactly that prose. Same class of mistake as a gate that
        fires on documentation about the gate.
        """
        import yaml as _yaml

        parsed = _yaml.safe_load(workflow.read_text(encoding="utf-8"))
        decide = parsed["jobs"]["decide"]
        mints = [
            step
            for step in decide["steps"]
            if "create-github-app-token" in str(step.get("uses", ""))
        ]
        assert mints, "the decide job mints no App token"
        return mints

    def test_the_deciding_job_requests_the_permission_its_premise_needs(
        self,
    ) -> None:
        """Inverted BACK, and which App it is minted from is why.

        This assertion has been both ways round inside one day, and neither
        reading was wrong at the time. #3733 asserted the request was present,
        because the premise cannot read protection without it. #3737 asserted it
        was absent, because the onexbot-occ-writer installation does not carry
        it and requesting an ungranted permission fails the MINT -- dispatch run
        35301341015 died at the token step and the job reported nothing per repo
        at all, which is strictly worse than the ci_protection_unreadable
        refusal it replaced.

        What changed is not the grant. No REST endpoint widens an App's granted
        set, so waiting for one was waiting on an org-level change. It is the
        APP: the deciding job now mints from `onexbot`, whose installation
        already carries `administration: read` with `repository_selection: all`,
        and which evidence-autoclose-sweep.yml in this repository has minted for
        the same reason since OMN-16832.

        So the premise and the identity that can satisfy it are asserted
        together, and the paired test below pins that identity. Moving the mint
        back to an App without the grant turns one of the two red.
        """
        for mint in self._decide_mints(self.WORKFLOW):
            assert mint["with"].get("permission-administration") == "read", (
                "the decide job must request administration:read -- the "
                "required-context membership its green-CI premise reads lives "
                "only behind branch protection, and without the request the "
                "train refuses every repo under ci_protection_unreadable"
            )

    def test_the_deciding_job_mints_from_the_app_that_carries_the_grant(
        self,
    ) -> None:
        """A permission request is only satisfiable against the right App.

        Asserted on the SECRET NAMES rather than on a comment naming the App,
        because the secrets are what the mint actually authenticates with. The
        two App credentials differ by one path segment in their secret names,
        which is exactly the kind of difference a review misses and a mint
        failure reports an hour later.
        """
        for mint in self._decide_mints(self.WORKFLOW):
            assert "ONEXBOT_APP_ID" in str(mint["with"].get("app-id", "")), (
                "the decide job must mint from the onexbot App, the one whose "
                "installation carries administration:read; onexbot-occ-writer "
                "does not carry it and the mint fails outright"
            )
            assert "ONEXBOT_APP_PRIVATE_KEY" in str(
                mint["with"].get("private-key", "")
            ), "the decide job's private key must match the App id it declares"
            assert "ONEXBOT_OCC_APP_ID" not in str(mint["with"].get("app-id", "")), (
                "ONEXBOT_OCC_APP_ID is the write-capable App used by the cut "
                "job; the deciding job writes nothing and must not hold it"
            )

    def test_the_deciding_job_requests_no_write_scope_at_all(self) -> None:
        """The decide job writes nothing to GitHub, so it holds nothing that could.

        It clones, plans, writes a step summary and uploads an artifact. Every
        write in this workflow is in the separate `cut` job, under its own token
        narrowed to the one repository being cut. Asserted over EVERY declared
        permission rather than over a named list, so a scope added later is
        covered by a test written before it existed.
        """
        for mint in self._decide_mints(self.WORKFLOW):
            granted = {
                key: value
                for key, value in mint["with"].items()
                if str(key).startswith("permission-")
            }
            assert granted, "the decide job's mint declares no permissions at all"
            offenders = {
                key: value
                for key, value in granted.items()
                if str(value).strip().lower() in self.WRITE_SCOPES
            }
            assert not offenders, (
                f"the decide job requests write scopes {sorted(offenders)}; it "
                "reads to decide and writes nothing, and a token that can "
                "rewrite the gates it is checking is the one thing this premise "
                "must not hold. Writes belong in the cut job"
            )
