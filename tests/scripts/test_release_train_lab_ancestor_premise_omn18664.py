# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The release train's lab premise inherits a receipt across non-runtime commits.

RED STATE THIS REPRODUCES (OMN-18664)
-------------------------------------
Release-train run 35316314098 skipped omnibase_infra with ``lab_receipt_absent``.
The train asked for a ``lab-pass-receipt-compose-dev-<sha>`` naming the EXACT dev
head, but ``runtime-rebuild-trigger.yml`` emits that artifact only when the merge
was runtime-affecting (``published == 'true' && runtime_lane == 'dev'``). Seven of
the eight dev merges preceding that run therefore have no receipt and never will
-- by design, not by failure. A workflow-only or ``scripts/ci``-only merge landing
last makes the cut premise unsatisfiable until some unrelated runtime merge
arrives, while adding nothing at all to the release it blocks.

THE FIX IS IN THE TRAIN, NOT THE TRIGGER
----------------------------------------
Emitting a receipt for a sha the lane never rebuilt would mean fabricating
``deployed_revision`` or inventing a "no rebuild needed" verdict -- the train's
decision relocated, not removed. The honest statement is already available: if no
commit between R and HEAD is runtime-affecting, the lane running R is the lane
running HEAD's runtime. So the premise resolves R and inherits its receipt.

EVERY GREEN ASSERTION HERE HAS A FLIPPED SIBLING
------------------------------------------------
"inherits across workflow-only commits" is paired with "refuses across a
``docker/migrations/**`` commit", and "uses the merge commit" is paired with
"never asks for the pull request head sha". A premise that only ever says yes is
not a premise.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULE = _REPO_ROOT / "scripts" / "ci" / "release_train.py"
_POLICY_PATH = _REPO_ROOT / "config" / "release_train_policy.yaml"
_CLASSIFIER = _REPO_ROOT / "scripts" / "runtime_change_classifier.py"
_TRIGGER = _REPO_ROOT / "scripts" / "trigger_rebuild_on_merge.py"


def _load(path: Path, name: str) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        msg = f"cannot load {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rt = _load(_MODULE, "release_train_under_test_omn18664")

# Shas modelled on the live 2026-09-18 reading: `22745243f` was dev HEAD with a
# rebuild in flight, `5be72e12f…` was the last sha carrying a PASS receipt, and
# the commits between them were workflow and scripts/ci only.
_HEAD = "22745243f49b6d108f5c8b292c5243b886e0f262"
_WORKFLOW_ONLY_1 = "011d3c3ca000000000000000000000000000000a"
_WORKFLOW_ONLY_2 = "334c4dcc0000000000000000000000000000000b"
_RECEIPTED = "5be72e12f1e9cb8168271a2dd31f8f6c5656e896"
_MIGRATION_COMMIT = "abcdef1234567890abcdef1234567890abcdef12"
#: The PULL REQUEST head sha of the merge that produced ``_HEAD``. Different by
#: construction on a squash-only repo, and the sha a naive query would use.
_PR_HEAD = "817923ed25b2000000000000000000000000000c"


@pytest.fixture(autouse=True)
def _green_ci_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about the lab premise; give them a green CI one."""
    monkeypatch.setattr(
        rt, "default_required_contexts", lambda repo, branch: ["CI Summary"]
    )
    monkeypatch.setattr(rt, "default_gating_sha", lambda repo, sha: _PR_HEAD)
    monkeypatch.setattr(
        rt,
        "default_check_runs",
        lambda repo, sha: [
            {"name": "CI Summary", "status": "completed", "conclusion": "success"}
        ],
    )


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
        "dev_head_sha": _HEAD,
        "unreleased_count": 4,
        "unreleased_subjects": ("feat(OMN-18567): schedule the convergence",),
    }
    fields.update(overrides)
    return rt.ModelRepoFacts(**fields)


def _receipt(sha: str, result: Any = None) -> Any:
    lab = rt.lab_pass_receipt
    outcome = result or lab.EnumLabPassResult.PASS
    return lab.ModelLabPassReceipt(
        sha=sha,
        lane=lab.EnumLabLane.COMPOSE_DEV,
        started_at=datetime(2026, 9, 18, 2, 48, tzinfo=UTC),
        finished_at=datetime(2026, 9, 18, 2, 53, tzinfo=UTC),
        result=outcome,
        checks=(
            lab.ModelLabPassCheck(
                name="ready_main",
                ok=outcome is lab.EnumLabPassResult.PASS,
                evidence="lane at 25e15ca25b71, which contains merge sha",
            ),
        ),
        agent_command_id="79d743db-92a1-4c19-9695-af92d9f9939d",
    )


def _seams(
    *,
    receipted: dict[str, Any] | None = None,
    branch: list[str] | None = None,
    runtime_shas: set[str] | None = None,
    pending: set[str] | None = None,
    queried: list[str] | None = None,
) -> dict[str, Any]:
    """Build the four injectable seams `decide` takes for the lab premise.

    ``receipted`` maps a sha to the receipt the artifact store holds for it;
    every other sha resolves to zero artifacts, which is the real behaviour of an
    exact-name artifact query.
    """
    have = receipted or {}
    runtime = runtime_shas or set()
    is_pending = pending or set()

    def _list(repo: str, name: str) -> list[dict[str, Any]]:
        if queried is not None:
            queried.append(name)
        for sha in have:
            if name.endswith(sha):
                return [{"id": 1, "created_at": "2026-09-18T02:53:14Z"}]
        return []

    def _download(repo: str, artifact_id: int) -> Any:
        return next(iter(have.values()))

    def _branch_commits() -> list[str]:
        return branch if branch is not None else [_HEAD]

    def _runtime_affecting(sha: str) -> bool:
        return sha in runtime

    def _rebuild_pending(repo: str, sha: str) -> bool:
        return sha in is_pending

    return {
        "list_artifacts": _list,
        "download_receipt": _download,
        "branch_commits": _branch_commits,
        "runtime_affecting": _runtime_affecting,
        "rebuild_pending": _rebuild_pending,
    }


# --------------------------------------------------------------------------- #
# AC1 -- the nearest runtime-affecting ancestor.                                #
# --------------------------------------------------------------------------- #
class TestNearestRuntimeAffectingAncestor:
    def test_workflow_only_head_inherits_the_ancestors_receipt(self) -> None:
        """RED before OMN-18664: this is the live 2026-09-18 state.

        HEAD is workflow-only, two workflow-only commits sit above a receipted
        runtime commit, and nothing between them can change what the lane runs.
        """
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_seams(
                receipted={_RECEIPTED: _receipt(_RECEIPTED)},
                branch=[_HEAD, _WORKFLOW_ONLY_1, _WORKFLOW_ONLY_2, _RECEIPTED],
                runtime_shas={_RECEIPTED},
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert decision.reason is rt.EnumTrainReason.UNRELEASED_WORK_LAB_PROVEN
        # Both shas named, and the count of what was skipped, because a reader
        # of a CUT row has to be able to check the inheritance themselves.
        assert _HEAD[:12] in decision.detail
        assert _RECEIPTED[:12] in decision.detail
        assert "3" in decision.detail, (
            "the detail must state how many non-runtime commits the premise "
            f"walked past; got {decision.detail!r}"
        )
        # The release is still cut at HEAD. Inheriting a lane proof never moves
        # the commit being released.
        assert decision.candidate_sha == _HEAD

    def test_a_runtime_commit_in_the_gap_voids_the_inheritance(self) -> None:
        """The flipped sibling. One migration commit and the premise refuses."""
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_seams(
                receipted={_RECEIPTED: _receipt(_RECEIPTED)},
                branch=[_HEAD, _MIGRATION_COMMIT, _RECEIPTED],
                runtime_shas={_MIGRATION_COMMIT, _RECEIPTED},
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT
        assert _MIGRATION_COMMIT[:12] in decision.detail, (
            "the refusal must name the runtime-affecting commit that has no "
            f"receipt, not the head it was reached from; got {decision.detail!r}"
        )

    def test_an_exact_head_receipt_is_preferred_over_any_ancestor(self) -> None:
        """A receipted, runtime-affecting HEAD resolves to itself, not a walk."""
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_seams(
                receipted={_HEAD: _receipt(_HEAD)},
                branch=[_HEAD, _WORKFLOW_ONLY_1, _RECEIPTED],
                runtime_shas={_HEAD, _RECEIPTED},
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert decision.reason is rt.EnumTrainReason.UNRELEASED_WORK_LAB_PROVEN
        assert _RECEIPTED[:12] not in decision.detail, (
            "an exact-head receipt must not be described as inherited from an "
            f"ancestor; got {decision.detail!r}"
        )

    def test_no_runtime_affecting_ancestor_within_the_walk_refuses(self) -> None:
        """Exhausting the walk is not a pass. It resolves nothing and refuses."""
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_seams(
                receipted={_RECEIPTED: _receipt(_RECEIPTED)},
                branch=[_HEAD, _WORKFLOW_ONLY_1, _WORKFLOW_ONLY_2],
                runtime_shas=set(),
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT

    def test_an_unreadable_classifier_does_not_inherit(self) -> None:
        """Fail closed: a classifier that raises falls back to the exact head."""

        def _boom(sha: str) -> bool:
            raise RuntimeError("omniclaude validator not checked out")

        seams = _seams(
            receipted={_RECEIPTED: _receipt(_RECEIPTED)},
            branch=[_HEAD, _RECEIPTED],
            runtime_shas={_RECEIPTED},
        )
        seams["runtime_affecting"] = _boom
        decision = rt.decide(policy=_policy(), facts=_facts(), **seams)
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT
        assert _HEAD[:12] in decision.detail

    def test_omitting_the_ancestry_seams_keeps_the_exact_head_premise(self) -> None:
        """No seams supplied is the pre-OMN-18664 behaviour, unchanged."""
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            list_artifacts=lambda repo, name: [],
            download_receipt=lambda repo, artifact_id: None,
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT


# --------------------------------------------------------------------------- #
# AC2 -- one classifier, imported.                                              #
# --------------------------------------------------------------------------- #
class TestOneClassifier:
    def test_the_train_carries_no_copy_of_the_runtime_path_list(self) -> None:
        body = _MODULE.read_text(encoding="utf-8")
        for literal in (
            "run-forward-migrations",
            "docker/migrations",
            "docker/runtime-policy.env",
            "scripts/deploy-agent/deploy_agent",
            "docker/onex-api",
            "LANE_STATE_PATH_PATTERNS",
        ):
            assert literal not in body, (
                f"{literal!r} appears in release_train.py. The runtime-affecting "
                "predicate has exactly one definition, in "
                "scripts/runtime_change_classifier.py; a second list here is how "
                "the train inherits a receipt across a commit the trigger "
                "considers runtime-affecting"
            )

    def test_the_trigger_re_exports_the_same_objects_not_copies(self) -> None:
        classifier = _load(_CLASSIFIER, "runtime_change_classifier_omn18664")
        trigger_body = _TRIGGER.read_text(encoding="utf-8")
        assert "runtime_change_classifier" in trigger_body, (
            "trigger_rebuild_on_merge.py must load the shared classifier rather "
            "than defining its own"
        )
        assert classifier.LANE_STATE_PATH_PATTERNS, (
            "the shared module must carry the lane-state patterns"
        )
        # The patterns the trigger's own comment block names are still here.
        joined = "\n".join(classifier.LANE_STATE_PATH_PATTERNS)
        assert "scripts/run-forward-migrations.sh" in joined
        assert "docker/migrations/**" in joined

    def test_adding_a_pattern_changes_the_trains_verdict_without_editing_it(
        self, tmp_path: Path
    ) -> None:
        """A new lane-state pattern reaches the train through the shared module.

        The train never restates the list, so widening it is a one-file change
        and this proves the train follows it.
        """
        classifier = _load(_CLASSIFIER, "runtime_change_classifier_omn18664")
        validator = tmp_path / "validate_pr_deploy_required.py"
        validator.write_text(
            "def find_runtime_paths(changed_files, *a, **k):\n    return []\n",
            encoding="utf-8",
        )
        canonical = classifier.load_runtime_path_classifier(validator)

        changed = ["config/brand_new_lane_state.toml"]
        assert classifier.classify_runtime_paths(changed, canonical) == []

        widened = (*classifier.LANE_STATE_PATH_PATTERNS, "config/brand_new_*.toml")
        original = classifier.LANE_STATE_PATH_PATTERNS
        try:
            classifier.LANE_STATE_PATH_PATTERNS = widened
            assert classifier.classify_runtime_paths(changed, canonical) == changed
        finally:
            classifier.LANE_STATE_PATH_PATTERNS = original


# --------------------------------------------------------------------------- #
# AC3 -- a pending rebuild is not an absent receipt.                            #
# --------------------------------------------------------------------------- #
class TestPendingIsNotAbsent:
    def test_an_in_flight_rebuild_reports_pending(self) -> None:
        """The race that produced run 35316314098's SKIP, told truthfully.

        The receipt WILL exist; it did not yet. Calling that "the sha has not
        been exercised on the compose dev lane" sent a lane on a four-hour
        investigation of a working system.
        """
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_seams(
                branch=[_HEAD],
                runtime_shas={_HEAD},
                pending={_HEAD},
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_PENDING
        assert "has not been exercised" not in decision.detail

    def test_no_run_and_no_artifact_is_still_absent(self) -> None:
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_seams(branch=[_HEAD], runtime_shas={_HEAD}, pending=set()),
        )
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT

    def test_a_pending_probe_that_raises_stays_absent(self) -> None:
        """Fail closed toward the SKIP that already refuses to cut."""
        seams = _seams(branch=[_HEAD], runtime_shas={_HEAD})

        def _boom(repo: str, sha: str) -> bool:
            raise RuntimeError("actions api unreadable")

        seams["rebuild_pending"] = _boom
        decision = rt.decide(policy=_policy(), facts=_facts(), **seams)
        assert decision.verdict is rt.EnumTrainVerdict.SKIP
        assert decision.reason is rt.EnumTrainReason.LAB_RECEIPT_ABSENT


# --------------------------------------------------------------------------- #
# AC4 -- the policy note stops lying.                                           #
# --------------------------------------------------------------------------- #
class TestPolicyNote:
    def test_no_compose_dev_note_claims_a_receipt_for_every_merge(self) -> None:
        policy = yaml.safe_load(_POLICY_PATH.read_text(encoding="utf-8"))
        for name, entry in policy["repos"].items():
            if entry.get("lab_evidence") != "compose-dev":
                continue
            note = entry["lab_evidence_note"]
            assert "for every merge" not in note, (
                f"{name}'s lab_evidence_note claims a receipt for every merge. "
                "The trigger emits one only for runtime-affecting merges, and "
                "this sentence prints on every decision row for this repo"
            )
            assert "runtime-affecting" in note, (
                f"{name}'s lab_evidence_note must state WHICH merges emit a "
                "receipt, since the reader of a SKIP row acts on it"
            )
            assert "ancestor" in note, (
                f"{name}'s lab_evidence_note must name the inheritance rule: a "
                "reader seeing a receipt for a sha that is not head needs to "
                "know that is intended"
            )


# --------------------------------------------------------------------------- #
# AC5 -- receipt queries are keyed by the merge commit.                         #
# --------------------------------------------------------------------------- #
class TestReceiptKeyedByMergeCommit:
    def test_the_pull_request_head_sha_is_never_queried(self) -> None:
        """A squash merge's run carries the PR head; the receipt carries neither.

        Proven live: run 35257835048 has headSha 0e5014b53d74 and emitted
        lab-pass-receipt-compose-dev-5be72e12f1e9…, four hex characters of which
        appear nowhere in that head sha. A query keyed by a run's headSha is a
        false zero by construction.
        """
        queried: list[str] = []
        decision = rt.decide(
            policy=_policy(),
            facts=_facts(),
            **_seams(
                receipted={_RECEIPTED: _receipt(_RECEIPTED)},
                branch=[_HEAD, _WORKFLOW_ONLY_1, _RECEIPTED],
                runtime_shas={_RECEIPTED},
                queried=queried,
            ),
        )
        assert decision.verdict is rt.EnumTrainVerdict.CUT
        assert queried, "the premise must actually query the artifact surface"
        for name in queried:
            assert _PR_HEAD not in name, (
                f"the receipt was looked up by the pull request head sha: {name}"
            )
        assert any(name.endswith(_RECEIPTED) for name in queried), (
            "the receipt must be looked up by the branch commit it is named for"
        )

    def test_the_workflow_supplies_the_classifier_and_never_fails_on_its_absence(
        self,
    ) -> None:
        """The premise needs omniclaude's validator, and degrades without it.

        Fetching it must not be able to fail the run: a train that cannot reach
        a sibling repo should still decide every repo on the narrower premise,
        which refuses rather than inherits.
        """
        workflow = yaml.safe_load(
            (
                _REPO_ROOT / ".github" / "workflows" / "release-train-nightly.yml"
            ).read_text(encoding="utf-8")
        )
        steps = workflow["jobs"]["decide"]["steps"]
        fetch = [
            s
            for s in steps
            if str(s.get("with", {}).get("repository", "")).endswith("omniclaude")
        ]
        assert fetch, (
            "the decide job must check out omniclaude's deploy-gate validator; "
            "without it the lab premise cannot classify a commit at all"
        )
        assert fetch[0].get("continue-on-error") is True, (
            "a failed sibling checkout must degrade the premise, not fail the "
            "whole decision"
        )
        plan = next(s for s in steps if s.get("id") == "plan")
        assert "--runtime-path-validator" in plan["run"]
        assert "validate_pr_deploy_required.py" in plan["run"]

    def test_the_cli_exposes_the_validator_and_no_override_of_the_premise(self) -> None:
        """A flag that supplies evidence is fine; one that asserts it is not."""
        parser = rt.build_parser()
        plan_actions = {
            option
            for action in parser._subparsers._group_actions[0].choices["plan"]._actions
            for option in action.option_strings
        }
        assert "--runtime-path-validator" in plan_actions
        for forbidden in ("--force", "--skip-lab", "--lab-receipt", "--assume-proven"):
            assert forbidden not in plan_actions, (
                f"{forbidden} would let a caller assert the lab premise instead "
                "of the gate resolving it"
            )

    def test_the_train_never_derives_a_receipt_name_from_a_run_head_sha(self) -> None:
        body = _MODULE.read_text(encoding="utf-8")
        for forbidden in ("head_sha", "headSha"):
            assert f"artifact_name({forbidden}" not in body
        # The gating sha resolver exists for the CI premise and must not leak
        # into the lab one.
        lab_section = body.split("def classify_lab_receipt", 1)[1].split(
            "\ndef decide", 1
        )[0]
        assert "gating_sha" not in lab_section, (
            "classify_lab_receipt must not reach for the gating (pull request "
            "head) sha; the receipt is named for the merge commit"
        )


# --------------------------------------------------------------------------- #
# The pending probe's own wire shape (OMN-18664 follow-up).                     #
# --------------------------------------------------------------------------- #
class TestPendingProbeWireShape:
    """A probe that 404s on every call reports "never pending" and says nothing.

    Measured live 2026-09-18: ``default_rebuild_pending`` passed its filter
    through ``gh api -f head_sha=…``. ``gh`` switches to POST the moment any
    ``-f`` is present, that endpoint has no POST, and the 404 was caught one
    frame up -- so the probe silently answered "not pending" for a sha whose
    trigger run was ``in_progress`` at that exact moment. AC3 was wired, tested
    through its seam, and dead on the real API.
    """

    def test_the_probe_issues_a_get_with_the_filter_in_the_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[list[str]] = []

        class _Completed:
            stdout = "in_progress\n"

        def _fake_run(argv: list[str], **kwargs: Any) -> Any:
            seen.append(argv)
            return _Completed()

        monkeypatch.setattr(rt, "default_gating_sha", lambda repo, sha: _PR_HEAD)
        monkeypatch.setattr(rt.subprocess, "run", _fake_run)

        assert rt.default_rebuild_pending("omnibase_infra", _HEAD) is True
        argv = seen[-1]
        assert "-f" not in argv and "-F" not in argv, (
            "gh api switches to POST as soon as -f/-F is present, and the "
            "workflow-runs endpoint has no POST; the filter belongs in the URL"
        )
        url = next(a for a in argv if a.startswith("repos/"))
        assert f"head_sha={_PR_HEAD}" in url, (
            f"the head sha must be a URL query parameter; got {url!r}"
        )

    def test_a_concluded_run_is_not_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The flipped sibling: every run completed means the receipt is absent."""

        class _Completed:
            stdout = "completed\ncompleted\n"

        monkeypatch.setattr(rt, "default_gating_sha", lambda repo, sha: _PR_HEAD)
        monkeypatch.setattr(rt.subprocess, "run", lambda *a, **k: _Completed())
        assert rt.default_rebuild_pending("omnibase_infra", _HEAD) is False

    def test_no_merged_pull_request_is_not_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rt, "default_gating_sha", lambda repo, sha: "")
        assert rt.default_rebuild_pending("omnibase_infra", _HEAD) is False
