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

        omnimemory is deliberately NOT here despite being the same shape as the
        other two. Its dev HEAD's gating commit carries a SKIPPED required
        context, which GitHub treats as satisfying protection, so a cut would
        fast-forward main across a commit whose gate never ran.
        """
        policies = rt.load_policy(_POLICY)
        armed = sorted(
            name for name, p in policies.items() if p.mode is rt.EnumTrainMode.CUT
        )
        assert armed == ["omnibase_core", "omnibase_infra", "omnibase_spi"]

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
