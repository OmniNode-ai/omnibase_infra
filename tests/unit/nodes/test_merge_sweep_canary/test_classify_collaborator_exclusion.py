# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Collaborator exclusion in the merge-sweep classifier (OMN-18823).

A pull request handed to a person — assigned to them, or with them as a
requested reviewer — must never be admitted by the sweep. Before this module
no admission rule anywhere read either field, and on 2026-09-19T12:33:45Z a
manual merge drain squash-merged omniweb#423 (OMN-18794), a pull request whose
assignee and requested reviewer were both a collaborator and whose body said in
prose that it was not to be merged. Prose is not a gate.

The roster is supplied on the typed input. It is deliberately NOT declared in
this repository: omnibase_infra is public, and the accounts live in the private
vocabulary home beside the public-repo hygiene denylist. Every login in this
module is therefore synthetic — a public test fixture naming real collaborators
would be the same disclosure one layer down.
"""

from __future__ import annotations

import inspect
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_merge_sweep_classify_compute.handlers import (
    handler_classify_prs as classify_module,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.handlers.handler_classify_prs import (
    HandlerClassifyPRs,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.enum_classify_skip_reason import (
    EnumClassifySkipReason,
)
from omnibase_infra.nodes.node_merge_sweep_classify_compute.models.model_classify_input import (
    ModelClassifyInput,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)

pytestmark = pytest.mark.unit

_ROSTER = ("collab-one", "collab-two")


def _make_pr(**overrides: object) -> ModelPRInfo:
    """A merge-ready PR: green CI, approved, mergeable, nobody assigned."""
    defaults: dict[str, object] = {
        "number": 1,
        "repo": "OmniNode-ai/test",
        "title": "test PR",
        "mergeable": "MERGEABLE",
        "review_decision": "APPROVED",
        "ci_status": "SUCCESS",
        "is_draft": False,
        "has_auto_merge": False,
        "assignees": (),
        "requested_reviewers": (),
    }
    defaults.update(overrides)
    return ModelPRInfo(**defaults)  # type: ignore[arg-type]


async def _classify(pr: ModelPRInfo, roster: tuple[str, ...] = _ROSTER):
    return await HandlerClassifyPRs().handle(
        ModelClassifyInput(
            prs=(pr,), correlation_id=uuid4(), collaborator_logins=roster
        )
    )


class TestCollaboratorExclusion:
    """The exclusion fires on either field, before any track determination."""

    @pytest.mark.asyncio
    async def test_assignee_on_the_roster_is_skipped(self) -> None:
        """An otherwise merge-ready PR assigned to a collaborator is SKIP."""
        result = await _classify(_make_pr(assignees=("collab-one",)))

        assert len(result.track_a) == 0
        assert len(result.track_b) == 0
        assert len(result.skipped) == 1
        skipped = result.skipped[0]
        assert skipped.track == "SKIP"
        assert skipped.skip_reason is EnumClassifySkipReason.COLLABORATOR_EXCLUDED
        assert skipped.excluded_account == "collab-one"
        assert "collab-one" in skipped.reason

    @pytest.mark.asyncio
    async def test_requested_reviewer_on_the_roster_is_skipped(self) -> None:
        """Requested review is a hand-off too, even with nobody assigned."""
        result = await _classify(_make_pr(requested_reviewers=("collab-two",)))

        assert len(result.skipped) == 1
        skipped = result.skipped[0]
        assert skipped.skip_reason is EnumClassifySkipReason.COLLABORATOR_EXCLUDED
        assert skipped.excluded_account == "collab-two"

    @pytest.mark.asyncio
    async def test_both_fields_empty_proceeds_to_its_normal_track(self) -> None:
        """Nobody assigned, nobody asked to review: the PR classifies as before."""
        result = await _classify(_make_pr())

        assert len(result.skipped) == 0
        assert len(result.track_a) == 1
        assert result.track_a[0].skip_reason is EnumClassifySkipReason.NOT_SKIPPED

    @pytest.mark.asyncio
    async def test_non_collaborator_assignee_proceeds(self) -> None:
        """An assignee who is not on the roster is not an exclusion."""
        result = await _classify(
            _make_pr(assignees=("someone-else",), requested_reviewers=("bot-account",))
        )

        assert len(result.skipped) == 0
        assert len(result.track_a) == 1

    @pytest.mark.asyncio
    async def test_empty_roster_excludes_nobody(self) -> None:
        """A declared-but-empty roster is a real answer, not a wildcard."""
        result = await _classify(_make_pr(assignees=("collab-one",)), roster=())

        assert len(result.skipped) == 0
        assert len(result.track_a) == 1

    @pytest.mark.asyncio
    async def test_exclusion_beats_a_track_b_pull_request(self) -> None:
        """The exclusion runs before track determination, so red CI still SKIPs.

        Without this ordering a collaborator's PR would land in Track B and be
        picked up by the polish path, which is a mutation of their branch.
        """
        result = await _classify(
            _make_pr(ci_status="FAILURE", assignees=("collab-one",))
        )

        assert len(result.track_b) == 0
        assert len(result.skipped) == 1
        assert (
            result.skipped[0].skip_reason
            is EnumClassifySkipReason.COLLABORATOR_EXCLUDED
        )

    @pytest.mark.asyncio
    async def test_draft_still_reports_draft_not_collaborator(self) -> None:
        """Draft is checked first and keeps its own reason; both are SKIP."""
        result = await _classify(_make_pr(is_draft=True, assignees=("collab-one",)))

        assert result.skipped[0].skip_reason is EnumClassifySkipReason.DRAFT

    @pytest.mark.asyncio
    async def test_auto_merge_already_enabled_keeps_its_own_reason(self) -> None:
        """Auto-merge is checked before the exclusion, per the declared order."""
        result = await _classify(
            _make_pr(has_auto_merge=True, assignees=("collab-one",))
        )

        assert (
            result.skipped[0].skip_reason is EnumClassifySkipReason.AUTO_MERGE_ENABLED
        )

    @pytest.mark.asyncio
    async def test_mixed_batch_separates_the_excluded_from_the_admitted(self) -> None:
        """Two admissible PRs and two hand-offs in one batch."""
        prs = (
            _make_pr(number=1),
            _make_pr(number=2, assignees=("collab-one",)),
            _make_pr(number=3, requested_reviewers=("collab-two",)),
            _make_pr(number=4, assignees=("someone-else",)),
        )
        result = await HandlerClassifyPRs().handle(
            ModelClassifyInput(
                prs=prs, correlation_id=uuid4(), collaborator_logins=_ROSTER
            )
        )

        assert result.total_classified == 4
        assert len(result.track_a) == 2
        assert len(result.skipped) == 2
        assert {c.excluded_account for c in result.skipped} == {
            "collab-one",
            "collab-two",
        }


class TestTheRosterIsStatedRatherThanAssumed:
    """An unstated roster must not read as an empty one."""

    def test_collaborator_logins_is_required_on_the_input(self) -> None:
        """Omitting the roster is a ValidationError, not a silent empty tuple."""
        with pytest.raises(ValidationError):
            ModelClassifyInput(prs=(), correlation_id=uuid4())  # type: ignore[call-arg]

    def test_assignees_and_requested_reviewers_are_required_on_the_pr(self) -> None:
        """A PR record that never observed the fields cannot be built at all."""
        with pytest.raises(ValidationError):
            ModelPRInfo(number=1, repo="OmniNode-ai/test")  # type: ignore[call-arg]

    def test_a_pr_carrying_both_fields_round_trips(self) -> None:
        """Positive control for the two refusals above."""
        pr = _make_pr(assignees=("a",), requested_reviewers=("b",))
        assert pr.assignees == ("a",)
        assert pr.requested_reviewers == ("b",)

    def test_the_handler_module_hardcodes_no_roster(self) -> None:
        """The roster arrives on the input; the node declares no default set.

        A module-level collection of logins here would be the hardcoded roster
        the contract exists to replace — and, in a public repository, a
        disclosure. This asserts the absence by shape rather than by naming any
        account.
        """
        source = inspect.getsource(classify_module)
        assert "collaborator_logins" in source, "the handler must read the roster"
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "collaborator" not in stripped.lower():
                continue
            assert not stripped.startswith(
                ("_COLLABORATORS", "COLLABORATORS", "_ROSTER", "ROSTER")
            ), f"module-level roster constant found: {stripped}"


class TestThePrListEffectObservesTheFields:
    """The exclusion cannot fire on a field the scan never asked GitHub for."""

    def test_gh_field_string_requests_both(self) -> None:
        """A field absent from the --json list comes back absent, silently."""
        from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.handlers import (
            handler_pr_list,
        )

        fields = handler_pr_list._GH_JSON_FIELDS.split(",")
        assert "assignees" in fields
        assert "reviewRequests" in fields

    def test_a_gh_payload_maps_onto_the_model(self) -> None:
        """User logins, team slugs and a bare name all survive the mapping."""
        from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.handlers.handler_pr_list import (
            _pr_json_to_model,
        )

        pr = _pr_json_to_model(
            {
                "number": 423,
                "title": "a PR handed to someone",
                "assignees": [{"login": "collab-one"}],
                "reviewRequests": [{"login": "collab-two"}, {"slug": "platform-leads"}],
            },
            "OmniNode-ai/test",
        )

        assert pr.assignees == ("collab-one",)
        assert pr.requested_reviewers == ("collab-two", "platform-leads")

    def test_absent_keys_map_to_empty_rather_than_raising(self) -> None:
        """gh omits an empty list; that is observed-and-empty, not unobserved."""
        from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.handlers.handler_pr_list import (
            _pr_json_to_model,
        )

        pr = _pr_json_to_model({"number": 1}, "OmniNode-ai/test")
        assert pr.assignees == ()
        assert pr.requested_reviewers == ()


class TestTheContractDeclaresThePolicy:
    """The rule is contract-declared, and the accounts are not in this repo."""

    def _contract(self) -> dict:
        import pathlib

        import yaml

        path = (
            pathlib.Path(__file__).resolve().parents[4]
            / "src"
            / "omnibase_infra"
            / "nodes"
            / "node_merge_sweep_classify_compute"
            / "contract.yaml"
        )
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_the_exclusion_is_declared_as_a_skip_rule(self) -> None:
        rules = self._contract()["admission_policy"]["skip_rules"]
        by_name = {r["name"]: r for r in rules}
        rule = by_name["collaborator_exclusion"]
        assert rule["skip_reason"] == EnumClassifySkipReason.COLLABORATOR_EXCLUDED.value
        assert set(rule["fields"]) == {"assignees", "requested_reviewers"}

    def test_the_rule_runs_before_any_track_is_determined(self) -> None:
        """Declared order is draft, auto-merge, then the exclusion."""
        rules = self._contract()["admission_policy"]["skip_rules"]
        assert [r["name"] for r in rules] == [
            "draft",
            "auto_merge_enabled",
            "collaborator_exclusion",
        ]

    def test_the_contract_names_the_roster_source_and_declares_no_accounts(
        self,
    ) -> None:
        """This repository is public; it declares the policy, never the people."""
        rules = {
            r["name"]: r for r in self._contract()["admission_policy"]["skip_rules"]
        }
        roster = rules["collaborator_exclusion"]["roster"]
        assert roster["declared_here"] is False
        assert roster["input_field"] == "collaborator_logins"
        assert roster["source_path"]
        assert "collaborator_logins" not in roster or isinstance(
            roster.get("collaborator_logins"), type(None)
        )
