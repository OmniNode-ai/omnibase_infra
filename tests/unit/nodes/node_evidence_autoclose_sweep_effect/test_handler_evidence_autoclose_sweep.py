# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerEvidenceAutocloseSweep — fake gh/dod_verify/Linear.

Covers OMN-16106 first slice: companion-merge -> dod_verify -> governed
Done flip, with every fail-closed path (missing binding, ambiguous
binding, dod_verify non-zero exit, dod_verify unparseable, Linear API
error, close-if-done label, already-Done ticket) plus the two happy
paths (flip, gap) in both DRY-RUN and apply modes, plus the kill switch.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _ac_coverage_gap,
    _acceptance_criteria_items,
    _extract_ticket_binding,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_evidence_autoclose_decision import (
    EnumEvidenceAutocloseDecision,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)

_OCC_REPO = "OmniNode-ai/onex_change_control"


def _merged_pr(number: int, title: str, ticket: str) -> dict[str, object]:
    # Merge timestamp must stay inside the handler's real-clock lookback
    # window (default 24h, see `_request()` below) regardless of when the
    # test suite runs — a fixed wall-clock literal here is a time bomb that
    # ages out of the window and starts failing days after being written.
    recently_merged = (datetime.now(tz=UTC) - timedelta(hours=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {
        "number": number,
        "html_url": f"https://github.com/{_OCC_REPO}/pull/{number}",
        "title": title,
        "updated_at": recently_merged,
        "merged_at": recently_merged,
    }


_RECEIPT_SUMMARY_MODEL = (
    "omnibase_infra.cli.model_receipt_runtime_summary.ModelReceiptRuntimeSummary"
)
_DOD_VERIFY_STATE_MODEL = (
    "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState"
)


def _dod_verify_ok(
    *,
    total: int,
    verified: int,
    failed: int,
    skipped: int = 0,
    behavior_proving: int = 1,
    verdict_status: str | None = None,
) -> dict[str, object]:
    """A ModelSkillResult shaped like the one `onex skill dod_verify` prints.

    OMN-16736: on a NON-success run the verification counts live under
    ``result.terminal_payload``, not flat on ``result``. ``result`` there
    carries the dispatch outcome and the node's own terminal state is nested
    one level below it.

    OMN-16961: that is only ONE of the two arms the CLI prints, and this
    double used to emit it unconditionally — including for a verified verdict,
    where it also stamped ``status: "success"`` and
    ``result_model: ModelDodVerifyState`` on a body that carried neither. That
    receipt cannot exist. ``receipt_mode`` puts the handler's own model FLAT on
    ``result`` whenever the run is success-like, and a verified dod_verify run
    is exactly that. So the double emitting an impossible hybrid is why the
    live 10-of-19 ``error_verify_unparseable`` split had no failing test: the
    only arm production could ever flip on was the one no test constructed.

    The arm is therefore derived from the verdict here, exactly as the CLI
    derives it — verified -> flat success arm, anything else -> nested summary
    arm. Both are pinned against verbatim captures in tests/fixtures/omn16961/.

    OMN-15911 added ``behavior_proving_count`` to the verdict. It defaults
    to 1 here so the pre-existing cases keep exercising the path they were
    written for (a flip on green counts); the cases that exercise the
    proof-class guard itself pass it explicitly. It is NOT optional in the
    payload — a verdict that omits the key entirely models a pre-OMN-15911
    verifier and is deliberately an ERROR, covered in
    test_omn_15911_behavior_proof_gate.py.
    """
    status = verdict_status or ("verified" if failed == 0 else "failed")
    verdict: dict[str, object] = {
        "correlation_id": str(uuid4()),
        "ticket_id": "OMN-9999",
        "status": status,
        "dry_run": False,
        "checks": [],
        "total_checks": total,
        "verified_count": verified,
        "failed_count": failed,
        "skipped_count": skipped,
        "superseded_count": 0,
        "behavior_proving_count": behavior_proving,
        "error_message": None,
    }
    success_like = status == "verified"
    envelope: dict[str, object] = {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success" if success_like else "failed",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0 if success_like else 1,
        "duration_ms": 1,
    }
    if success_like:
        envelope["result"] = verdict
        envelope["result_model"] = _DOD_VERIFY_STATE_MODEL
    else:
        envelope["result"] = {
            "workflow_result": "failed",
            "exit_code": 1,
            "workflow": "<OMNI_HOME>/omnimarket/nodes/node_dod_verify/contract.yaml",
            "terminal_payload": verdict,
        }
        envelope["result_model"] = _RECEIPT_SUMMARY_MODEL
    return envelope


def _verdict_of(receipt: dict[str, object]) -> dict[str, object]:
    """The verdict body of a ``_dod_verify_ok`` receipt, whichever arm it is in."""
    result = receipt["result"]
    assert isinstance(result, dict)
    if receipt["result_model"] == _RECEIPT_SUMMARY_MODEL:
        nested = result["terminal_payload"]
        assert isinstance(nested, dict)
        return nested
    return result


class FakeLinearClient:
    """Fake Linear client — canned issue payloads, records mutation calls."""

    def __init__(
        self,
        issues: dict[str, dict[str, object]] | None = None,
        done_state_id: str | None = "state-done-id",
    ) -> None:
        self._issues = issues or {}
        self._done_state_id = done_state_id
        self.state_updates: list[tuple[str, str]] = []
        self.comments: list[tuple[str, str]] = []
        self.fetch_issue_calls: list[str] = []

    async def fetch_issue(self, ticket_id: str) -> dict[str, object] | None:
        self.fetch_issue_calls.append(ticket_id)
        return self._issues.get(ticket_id)

    async def fetch_done_state_id(self, team_id: str) -> str | None:
        return self._done_state_id

    async def update_issue_state(self, issue_id: str, state_id: str) -> bool:
        self.state_updates.append((issue_id, state_id))
        return True

    async def create_comment(self, issue_id: str, body: str) -> bool:
        self.comments.append((issue_id, body))
        return True

    async def fetch_comment_bodies(self, issue_id: str) -> tuple[str, ...] | None:
        # OMN-16808: the sweep reads what it has already said on a ticket
        # before it says anything else. This double serves its own write log
        # back, so a second run over the same window is distinguishable from a
        # first — a fake that always returned () would let the duplicate-post
        # defect pass unnoticed here.
        return tuple(body for target, body in self.comments if target == issue_id)


def _issue(
    *,
    issue_id: str = "issue-uuid-1",
    state_type: str = "started",
    labels: tuple[str, ...] = (),
    description: str | None = None,
) -> dict[str, object]:
    return {
        "id": issue_id,
        "identifier": "OMN-9999",
        "state": {"id": "state-1", "name": "In Progress", "type": state_type},
        "labels": {"nodes": [{"name": label} for label in labels]},
        "team": {"id": "team-1"},
        # Linear returns JSON null, not "", for an empty description.
        "description": description,
    }


def _make_gh_fake(
    companions: list[dict[str, object]],
    files_by_pr: dict[int, list[str]],
    files_error_by_pr: dict[int, str] | None = None,
):
    files_error_by_pr = files_error_by_pr or {}

    async def fake_run_gh_command(args: list[str], timeout: float):
        path = args[2]
        if "/files" in path:
            number = int(path.split("/pulls/")[1].split("/files")[0])
            if number in files_error_by_pr:
                return None, files_error_by_pr[number]
            files = files_by_pr.get(number, [])
            # OMN-16736: the real GitHub REST payload for
            # `GET /repos/{owner}/{repo}/pulls/{number}/files` keys each entry
            # on "filename". This double previously emitted {"path": ...},
            # which is the Contents/trees key and appears nowhere in this
            # endpoint's response — so the whole suite stayed green while the
            # production reader matched nothing and returned ([], "") for
            # every PR ever scanned. The double must speak the real API.
            return [{"filename": f} for f in files], ""
        # PR-list page.
        page = int(path.rsplit("page=", 1)[1])
        if page == 1:
            return companions, ""
        return [], ""

    return fake_run_gh_command


def _make_dod_verify_fake(responses: dict[str, tuple[dict | None, int, str]]):
    async def fake_run_dod_verify(ticket_id: str, cwd: str, timeout: int):
        return responses[ticket_id]

    return fake_run_dod_verify


def _request(**overrides: object) -> ModelEvidenceAutocloseSweepRequest:
    defaults: dict[str, object] = {
        "correlation_id": uuid4(),
        "occ_repo": _OCC_REPO,
        "lookback_hours": 24,
        "apply": False,
    }
    defaults.update(overrides)
    return ModelEvidenceAutocloseSweepRequest(**defaults)


@pytest.mark.unit
class TestExtractTicketBinding:
    def test_binding_from_contract_file(self):
        ticket_id, ambiguous = _extract_ticket_binding(
            "docs(OMN-14750): OCC companion",
            ["contracts/OMN-14750.yaml", "drift/dod_receipts/OMN-14750/x.yaml"],
        )
        assert ticket_id == "OMN-14750"
        assert ambiguous is False

    def test_binding_from_title_only(self):
        ticket_id, ambiguous = _extract_ticket_binding(
            "evidence(OMN-9999): OCC companion for X#1", []
        )
        assert ticket_id == "OMN-9999"
        assert ambiguous is False

    def test_missing_binding(self):
        ticket_id, ambiguous = _extract_ticket_binding(
            "chore: bump deps", ["README.md"]
        )
        assert ticket_id is None
        assert ambiguous is False

    def test_ambiguous_binding_file_vs_title(self):
        ticket_id, ambiguous = _extract_ticket_binding(
            "evidence(OMN-1111): mismatched title", ["contracts/OMN-2222.yaml"]
        )
        assert ticket_id is None
        assert ambiguous is True

    def test_ambiguous_binding_two_contract_files(self):
        ticket_id, ambiguous = _extract_ticket_binding(
            "evidence(OMN-1111): two contracts",
            ["contracts/OMN-1111.yaml", "contracts/OMN-2222.yaml"],
        )
        assert ticket_id is None
        assert ambiguous is True


@pytest.mark.unit
class TestKillSwitch:
    async def test_kill_switch_env_var_short_circuits(self, monkeypatch):
        monkeypatch.setenv("ONEX_AUTOCLOSE_DISABLED", "1")
        gh_calls: list[object] = []

        async def fake_gh(args, timeout):
            gh_calls.append(args)
            return [], ""

        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=fake_gh
        )
        result = await handler.handle(_request())
        assert result.kill_switch_engaged is True
        assert result.companions_scanned == 0
        assert gh_calls == []

    async def test_kill_switch_constructor_override(self):
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), autoclose_disabled=True
        )
        result = await handler.handle(_request())
        assert result.kill_switch_engaged is True


@pytest.mark.unit
class TestBindingSkips:
    async def test_missing_binding_is_skipped_never_guessed(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "chore: bump deps", "")],
            files_by_pr={1: ["README.md"]},
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=gh_fake
        )
        result = await handler.handle(_request())
        assert result.companions_scanned == 1
        assert result.bindings_extracted == 0
        assert len(result.outcomes) == 1
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_NO_BINDING
        )

    async def test_ambiguous_binding_is_skipped_never_guessed(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-1111): x", "")],
            files_by_pr={1: ["contracts/OMN-2222.yaml"]},
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=gh_fake
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING
        )

    async def test_real_github_files_payload_shape_yields_contract_paths(self):
        """Pin the live GitHub response shape: entries are keyed on ``filename``.

        OMN-16736. This is the test the suite did not have. ``_fetch_pr_files``
        read ``item["path"]`` while ``GET /repos/{owner}/{repo}/pulls/{number}
        /files`` keys every entry on ``filename`` — verified live 2026-08-27,
        the response object's keys are exactly the set asserted below and
        ``path`` is not among them. The reader therefore returned ``([], "")``
        for every PR ever scanned, and the old test double emitted the same
        wrong key, so the defect was invisible: the double and the reader
        agreed with each other and disagreed with GitHub.

        This exercises the real reader against a verbatim-shaped payload
        rather than the shared ``_make_gh_fake`` helper, so it stays honest
        even if that helper is edited again.
        """
        live_shape_keys = {
            "sha",
            "filename",
            "status",
            "additions",
            "deletions",
            "changes",
            "blob_url",
            "raw_url",
            "contents_url",
            "patch",
        }
        assert "path" not in live_shape_keys

        payload = [
            {key: f"<{key}>" for key in live_shape_keys} | {"filename": name}
            for name in (
                "contracts/OMN-16682.yaml",
                "drift/dod_receipts/OMN-16682/occ-self-bind-pr-7267/command.yaml",
            )
        ]

        async def fake_run_gh_command(args: list[str], timeout: float):
            return payload, ""

        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=fake_run_gh_command
        )
        files, error = await handler._fetch_pr_files(_OCC_REPO, 7267, 30)
        assert error == ""
        assert files == [
            "contracts/OMN-16682.yaml",
            "drift/dod_receipts/OMN-16682/occ-self-bind-pr-7267/command.yaml",
        ]

    async def test_uninterpretable_files_payload_is_an_error_not_empty(self):
        """Entries present but no usable path key must fail closed (OMN-16736).

        The failure this guards is the one that actually shipped: a non-empty
        payload whose entries this reader cannot interpret silently became an
        empty changed-file list, which then fell through to a title-only
        binding. "The API returned rows I cannot read" is a fetch failure, not
        "this companion touched zero files".
        """

        async def fake_run_gh_command(args: list[str], timeout: float):
            return [{"path": "contracts/OMN-1234.yaml"}], ""

        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=fake_run_gh_command
        )
        files, error = await handler._fetch_pr_files(_OCC_REPO, 1234, 30)
        assert files == []
        assert "filename" in error

    async def test_two_contract_files_beat_a_single_id_title(self):
        """The live #7267 signature: title names one ticket, files name two.

        OMN-16736. onex_change_control#7267 is titled ``evidence(OMN-16682):
        ...`` but touches ``contracts/OMN-16682.yaml`` AND ``contracts/
        OMN-16691.yaml``. With the file listing dead it bound to OMN-16682 and
        reported no ambiguity — a mis-targeted Done flip under ``--apply``.
        The file listing must win and the sweep must refuse to guess.
        """
        gh_fake = _make_gh_fake(
            companions=[
                _merged_pr(
                    7267,
                    "evidence(OMN-16682): OCC Evidence-Source autobind for X#327",
                    "",
                )
            ],
            files_by_pr={
                7267: [
                    "contracts/OMN-16682.yaml",
                    "contracts/OMN-16691.yaml",
                    "drift/dod_receipts/OMN-16682/occ-self-bind-pr-7267/command.yaml",
                ]
            },
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=gh_fake
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_AMBIGUOUS_BINDING
        )
        assert result.outcomes[0].ticket_id == ""
        assert result.bindings_extracted == 0

    async def test_file_fetch_failure_fails_closed_never_falls_back_to_title(self):
        """A GitHub file-fetch failure must never be treated as 'zero files'.

        Regression coverage: _fetch_pr_files silently returning [] on error
        would let _extract_ticket_binding fall through to a title-only match
        that a real file listing might have disambiguated or contradicted.
        """
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={},
            files_error_by_pr={1: "gh api rate limited"},
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=gh_fake
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.ERROR_GITHUB_API
        )
        assert result.bindings_extracted == 0


@pytest.mark.unit
class TestLinearGates:
    async def test_close_if_done_label_stays_manual(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(
            issues={"OMN-9999": _issue(labels=("close-if-done",))}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear, run_gh_command=gh_fake
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision == EnumEvidenceAutocloseDecision.SKIPPED_LABEL
        )
        assert linear.state_updates == []
        assert linear.comments == []

    async def test_already_done_ticket_is_skipped(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue(state_type="completed")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear, run_gh_command=gh_fake
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.SKIPPED_ALREADY_DONE
        )

    async def test_linear_fetch_failure_fails_closed_without_running_dod_verify(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={})  # fetch_issue returns None
        dod_calls: list[str] = []

        async def fake_dod_verify(ticket_id, cwd, timeout):
            dod_calls.append(ticket_id)
            raise AssertionError("dod_verify must not run when Linear fetch fails")

        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=fake_dod_verify,
        )
        result = await handler.handle(_request())
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
        )
        assert dod_calls == []


@pytest.mark.unit
class TestDodVerifyFailClosed:
    async def test_nonzero_exit_fails_closed(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake({"OMN-9999": (None, 1, "boom")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.ERROR_VERIFY_NONZERO_EXIT
        )
        assert linear.state_updates == []
        assert linear.comments == []

    async def test_unparseable_output_fails_closed(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake({"OMN-9999": (None, 0, "not json")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.ERROR_VERIFY_UNPARSEABLE
        )


@pytest.mark.unit
class TestFlipPath:
    async def test_dry_run_never_mutates_linear(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake(
            {"OMN-9999": (_dod_verify_ok(total=3, verified=3, failed=0), 0, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=False))
        outcome = result.outcomes[0]
        assert outcome.decision == EnumEvidenceAutocloseDecision.FLIPPED
        assert outcome.applied is False
        assert result.dry_run is True
        assert linear.state_updates == []
        assert linear.comments == []

    async def test_apply_flips_and_posts_audit_comment(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(42, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={42: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue(issue_id="issue-abc")})
        dod_fake = _make_dod_verify_fake(
            {"OMN-9999": (_dod_verify_ok(total=3, verified=3, failed=0), 0, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))
        outcome = result.outcomes[0]
        assert outcome.decision == EnumEvidenceAutocloseDecision.FLIPPED
        assert outcome.applied is True
        assert result.tickets_flipped == 1
        assert linear.state_updates == [("issue-abc", "state-done-id")]
        assert len(linear.comments) == 1
        comment_issue_id, comment_body = linear.comments[0]
        assert comment_issue_id == "issue-abc"
        assert "pull/42" in comment_body
        assert "3/3 ACs verified" in comment_body

    async def test_apply_with_no_done_state_fails_closed(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()}, done_state_id=None)
        dod_fake = _make_dod_verify_fake(
            {"OMN-9999": (_dod_verify_ok(total=3, verified=3, failed=0), 0, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))
        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.ERROR_LINEAR_API
        )
        assert linear.state_updates == []


@pytest.mark.unit
class TestGapPath:
    async def test_dry_run_gap_never_comments(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake(
            {"OMN-9999": (_dod_verify_ok(total=3, verified=2, failed=1), 1, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=False))
        assert result.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert result.outcomes[0].applied is False
        assert linear.comments == []

    async def test_apply_posts_gap_comment_never_flips(self):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue(issue_id="issue-gap")})
        dod_fake = _make_dod_verify_fake(
            {"OMN-9999": (_dod_verify_ok(total=3, verified=2, failed=1), 1, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))
        assert result.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert result.tickets_gap_posted == 1
        assert linear.state_updates == []
        assert len(linear.comments) == 1

    async def test_zero_checks_is_treated_as_gap_not_flip(self):
        """total_checks == 0 must never satisfy 'all ACs verified'."""
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake(
            {"OMN-9999": (_dod_verify_ok(total=0, verified=0, failed=0), 0, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=False))
        assert result.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED


@pytest.mark.unit
class TestRealDodVerifyPayloadShape:
    """The reader must speak the shape `onex skill dod_verify` really prints.

    OMN-16736, second occurrence of the OMN-16736/#2925 defect class. The
    handler read ``result.total_checks``; the live CLI nests every count under
    ``result.terminal_payload``. ``result`` itself carries only the DISPATCH
    outcome. With no ``total_checks`` at that level the reader saw 0/0 on every
    run, so a fully-verified ticket could never satisfy ``all_verified`` and no
    flip was reachable even once ``node_dod_verify`` resolved.

    These tests bypass the shared double and drive a COMMITTED LIVE CAPTURE, so
    they stay honest if that helper is edited again.
    """

    @staticmethod
    def _captured() -> dict[str, object]:
        fixture = (
            Path(__file__).resolve().parents[3]
            / "fixtures"
            / "omn16736"
            / "dod-verify-omn16752.skill-result.json.captured"
        )
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload

    def test_the_live_payload_has_no_top_level_total_checks(self):
        """Pin the key that is NOT there — the whole defect in one assertion."""
        captured = self._captured()
        dispatch = captured["result"]
        assert isinstance(dispatch, dict)
        assert "total_checks" not in dispatch
        assert "terminal_payload" in dispatch
        assert dispatch["terminal_payload"]["total_checks"] == 16

    async def test_a_real_failed_verdict_at_exit_1_is_a_gap_not_an_error(self):
        """`onex skill dod_verify` exits 1 on every genuine evidence gap.

        Live capture: OMN-16752, exit 1, 9/16 verified, 6 failed, 1 skipped,
        with a complete ModelSkillResult on stdout. That is a GAP the ticket's
        owner can act on, not a verifier crash.
        """
        captured = self._captured()
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(7294, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={7294: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue(issue_id="issue-real")})
        dod_fake = _make_dod_verify_fake({"OMN-9999": (captured, 1, "")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))
        outcome = result.outcomes[0]

        assert outcome.decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert outcome.dod_verify_total_checks == 16
        assert outcome.dod_verify_verified_count == 9
        assert outcome.dod_verify_failed_count == 6
        assert linear.state_updates == []
        assert result.tickets_errored == 0

    async def test_json_without_a_terminal_payload_fails_closed(self):
        """A dispatch that emitted JSON but reached no verdict is an ERROR.

        Never a 0/0 'gap' — that would read as a ticket problem when it is a
        verifier problem.
        """
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        no_verdict = {
            "skill_name": "dod_verify",
            "node_name": "node_dod_verify",
            "status": "failed",
            "exit_code": 1,
            "result": {"workflow_result": "failed", "exit_code": 1, "error": "boom"},
        }
        dod_fake = _make_dod_verify_fake({"OMN-9999": (no_verdict, 1, "")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))

        assert (
            result.outcomes[0].decision
            == EnumEvidenceAutocloseDecision.ERROR_VERIFY_NONZERO_EXIT
        )
        assert linear.state_updates == []
        assert linear.comments == []

    async def test_a_verified_terminal_status_is_required_for_a_flip(self):
        """Counts alone do not authorize a flip; dod_verify's verdict must agree."""
        # Same clean arithmetic, but the verifier itself declined to say
        # VERIFIED — which is also what puts the receipt in the nested
        # runtime-summary arm rather than the flat success arm (OMN-16961).
        payload = _dod_verify_ok(
            total=3, verified=3, failed=0, verdict_status="skipped"
        )
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake({"OMN-9999": (payload, 0, "")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=True))

        assert result.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert linear.state_updates == []


@pytest.mark.unit
class TestAcceptanceCriteriaExtraction:
    """Pure-function coverage for the description parser (OMN-16736)."""

    def test_items_under_a_markdown_heading(self):
        description = (
            "Some context.\n\n"
            "## Acceptance Criteria\n"
            "- AC1: the thing is wired\n"
            "- AC2: the thing is proven\n\n"
            "## Notes\n"
            "- not an AC\n"
        )
        assert _acceptance_criteria_items(description) == [
            "AC1: the thing is wired",
            "AC2: the thing is proven",
        ]

    def test_bold_heading_and_numbered_items(self):
        description = "**Acceptance criteria:**\n1. first\n2) second\n"
        assert _acceptance_criteria_items(description) == ["first", "second"]

    def test_bare_ac_prefixed_lines_without_bullets(self):
        description = "### AC\nAC1: alpha\nAC2 beta\n"
        assert _acceptance_criteria_items(description) == ["AC1: alpha", "AC2 beta"]

    def test_no_acceptance_criteria_section_yields_nothing(self):
        assert _acceptance_criteria_items("Just a paragraph.\n- a bullet\n") == []

    def test_task_markers_are_stripped_from_item_text(self):
        description = "## Acceptance Criteria\n- [x] done thing\n"
        assert _acceptance_criteria_items(description) == ["done thing"]


@pytest.mark.unit
class TestAcCoverageGapFunction:
    """The guard predicate itself: conservative, and silent when it should be."""

    def test_empty_description_is_never_a_gap(self):
        assert _ac_coverage_gap("", 3) == ("", ())
        assert _ac_coverage_gap("   \n\n  ", 3) == ("", ())

    def test_unchecked_checkbox_is_a_gap_and_is_named(self):
        reason, uncovered = _ac_coverage_gap(
            "## Acceptance Criteria\n- [x] wired\n- [ ] proven on the live lane\n", 2
        )
        assert reason
        assert uncovered == ("proven on the live lane",)

    def test_unchecked_checkbox_outside_an_ac_section_still_holds(self):
        """A criterion does not have to live under a heading to be a criterion."""
        reason, uncovered = _ac_coverage_gap("Notes\n\n- [ ] follow-up gate\n", 5)
        assert reason
        assert uncovered == ("follow-up gate",)

    def test_ac_section_longer_than_total_checks_is_a_gap(self):
        reason, uncovered = _ac_coverage_gap(
            "## Acceptance Criteria\n- alpha\n- beta\n- gamma\n", 2
        )
        assert reason
        assert "3" in reason and "2" in reason
        assert uncovered == ("alpha", "beta", "gamma")

    def test_fully_covered_ac_section_is_not_a_gap(self):
        assert _ac_coverage_gap("## Acceptance Criteria\n- alpha\n- beta\n", 2) == (
            "",
            (),
        )

    def test_more_checks_than_listed_acs_is_not_a_gap(self):
        """dod_verify covering MORE than the description lists is fine."""
        assert _ac_coverage_gap("## Acceptance Criteria\n- alpha\n", 4) == ("", ())


@pytest.mark.unit
class TestAcCoverageGuard:
    """OMN-14362: an AC that lives only in the Linear description is invisible
    to dod_verify, so a clean 0-failed run is not evidence about it. The flip
    must be withheld and the uncovered criteria named."""

    def _handler(self, linear, *, total=3, verified=3, failed=0):
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(77, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={77: ["contracts/OMN-9999.yaml"]},
        )
        dod_fake = _make_dod_verify_fake(
            {
                "OMN-9999": (
                    _dod_verify_ok(total=total, verified=verified, failed=failed),
                    0,
                    "",
                )
            }
        )
        return HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )

    async def test_unchecked_linear_only_ac_blocks_an_otherwise_clean_flip(self):
        linear = FakeLinearClient(
            issues={
                "OMN-9999": _issue(
                    issue_id="issue-ac",
                    description=(
                        "## Acceptance Criteria\n"
                        "- [x] AC1: guard wired as a CI gate\n"
                        "- [ ] AC2: replayed against the real incident\n"
                    ),
                )
            }
        )
        handler = self._handler(linear)
        result = await handler.handle(_request(apply=True))
        outcome = result.outcomes[0]

        assert outcome.decision == EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
        assert result.tickets_flipped == 0
        assert linear.state_updates == []
        assert outcome.uncovered_acceptance_criteria == (
            "AC2: replayed against the real incident",
        )
        assert len(linear.comments) == 1
        _, body = linear.comments[0]
        assert "AC2: replayed against the real incident" in body
        assert "3/3 ACs verified" in body

    async def test_ac_section_longer_than_dod_verify_checks_blocks_the_flip(self):
        linear = FakeLinearClient(
            issues={
                "OMN-9999": _issue(
                    issue_id="issue-ac2",
                    description=(
                        "## Acceptance Criteria\n"
                        "- AC1: alpha\n"
                        "- AC2: beta\n"
                        "- AC3: gamma\n"
                        "- AC4: delta\n"
                    ),
                )
            }
        )
        handler = self._handler(linear, total=2, verified=2, failed=0)
        result = await handler.handle(_request(apply=True))
        outcome = result.outcomes[0]

        assert outcome.decision == EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
        assert linear.state_updates == []
        assert len(outcome.uncovered_acceptance_criteria) == 4

    async def test_fully_covered_ticket_still_flips(self):
        """The guard must not be a blanket hold: a description whose ACs are
        all checked and match the check count flips exactly as before."""
        linear = FakeLinearClient(
            issues={
                "OMN-9999": _issue(
                    issue_id="issue-ok",
                    description=(
                        "## Acceptance Criteria\n"
                        "- [x] AC1: alpha\n"
                        "- [x] AC2: beta\n"
                        "- [x] AC3: gamma\n"
                    ),
                )
            }
        )
        handler = self._handler(linear)
        result = await handler.handle(_request(apply=True))

        assert result.outcomes[0].decision == EnumEvidenceAutocloseDecision.FLIPPED
        assert result.tickets_flipped == 1
        assert linear.state_updates == [("issue-ok", "state-done-id")]

    async def test_null_description_still_flips(self):
        """Linear returns null for an empty description; that is genuinely no
        criteria, not an unreadable one, so it must not become a blanket hold."""
        linear = FakeLinearClient(
            issues={"OMN-9999": _issue(issue_id="issue-null", description=None)}
        )
        handler = self._handler(linear)
        result = await handler.handle(_request(apply=True))

        assert result.outcomes[0].decision == EnumEvidenceAutocloseDecision.FLIPPED
        assert linear.state_updates == [("issue-null", "state-done-id")]

    async def test_dry_run_ac_gap_never_comments(self):
        linear = FakeLinearClient(
            issues={
                "OMN-9999": _issue(
                    issue_id="issue-dry",
                    description="## Acceptance Criteria\n- [ ] AC1: unproven\n",
                )
            }
        )
        handler = self._handler(linear)
        result = await handler.handle(_request(apply=False))
        outcome = result.outcomes[0]

        assert outcome.decision == EnumEvidenceAutocloseDecision.GAP_AC_COVERAGE
        assert outcome.applied is False
        assert linear.state_updates == []
        assert linear.comments == []
        assert result.tickets_gap_posted == 1

    async def test_ac_gap_is_counted_as_a_gap_not_an_error(self):
        linear = FakeLinearClient(
            issues={
                "OMN-9999": _issue(
                    issue_id="issue-count",
                    description="## Acceptance Criteria\n- [ ] AC1: unproven\n",
                )
            }
        )
        handler = self._handler(linear)
        result = await handler.handle(_request(apply=True))

        assert result.tickets_gap_posted == 1
        assert result.tickets_errored == 0
        assert result.tickets_skipped == 0

    async def test_dod_verify_gap_short_circuits_before_the_ac_guard(self):
        """A failed check is already a gap; the AC guard only gates the flip
        path, so a real dod_verify failure keeps its own honest decision."""
        linear = FakeLinearClient(
            issues={
                "OMN-9999": _issue(
                    issue_id="issue-both",
                    description="## Acceptance Criteria\n- [ ] AC1: unproven\n",
                )
            }
        )
        handler = self._handler(linear, total=3, verified=2, failed=1)
        result = await handler.handle(_request(apply=True))

        assert result.outcomes[0].decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        assert linear.state_updates == []


@pytest.mark.unit
class TestSweepLevel:
    async def test_github_enumeration_failure_is_sweep_level(self):
        async def fake_gh_fail(args, timeout):
            return None, "gh: command not found"

        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=fake_gh_fail
        )
        result = await handler.handle(_request())
        assert result.success is False
        assert "gh: command not found" in result.error_message
        assert result.outcomes == ()

    async def test_max_companions_cap_respected(self):
        companions = [
            _merged_pr(i, f"evidence(OMN-{9000 + i}): x", f"OMN-{9000 + i}")
            for i in range(1, 6)
        ]
        files_by_pr = {i: [f"contracts/OMN-{9000 + i}.yaml"] for i in range(1, 6)}
        gh_fake = _make_gh_fake(companions=companions, files_by_pr=files_by_pr)
        linear = FakeLinearClient(
            issues={f"OMN-{9000 + i}": _issue() for i in range(1, 6)}
        )
        dod_fake = _make_dod_verify_fake(
            {
                f"OMN-{9000 + i}": (
                    _dod_verify_ok(total=1, verified=1, failed=0),
                    0,
                    "",
                )
                for i in range(1, 6)
            }
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(max_companions=2))
        assert result.companions_scanned == 2

    async def test_gh_timeout_seconds_defaults_to_90_when_not_overridden(self):
        # Regression coverage for OMN-16106: the CI sweep timed out at a
        # hardcoded 30.0s `gh api` timeout (duration_ms == 30048 in the live
        # failing run) that was not contract-exposed. The default must come
        # from the request model, not a hardcoded literal in the handler.
        captured_timeouts: list[float] = []

        async def fake_run_gh_command(args: list[str], timeout: float):
            captured_timeouts.append(timeout)
            return [], ""

        handler = HandlerEvidenceAutocloseSweep(
            linear_client=FakeLinearClient(), run_gh_command=fake_run_gh_command
        )
        await handler.handle(_request())
        assert captured_timeouts == [90]

    async def test_gh_timeout_seconds_is_plumbed_from_the_request(self):
        # Proves the timeout used for BOTH gh api call sites (PR-list
        # enumeration and per-PR file listing) comes from
        # request.gh_timeout_seconds, not fixed instance/constructor state.
        captured_timeouts: list[float] = []

        async def fake_run_gh_command(args: list[str], timeout: float):
            captured_timeouts.append(timeout)
            path = args[2]
            if "/files" in path:
                return [{"path": "contracts/OMN-9500.yaml"}], ""
            page = int(path.rsplit("page=", 1)[1])
            if page == 1:
                return [_merged_pr(1, "evidence(OMN-9500): x", "OMN-9500")], ""
            return [], ""

        linear = FakeLinearClient(issues={"OMN-9500": _issue()})
        dod_fake = _make_dod_verify_fake(
            {"OMN-9500": (_dod_verify_ok(total=1, verified=1, failed=0), 0, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=fake_run_gh_command,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(gh_timeout_seconds=8))
        assert result.companions_scanned == 1
        assert captured_timeouts, "gh command was never invoked"
        assert all(t == 8 for t in captured_timeouts)


@pytest.mark.unit
class TestRealSubprocessReaping:
    """Regression coverage for CodeRabbit finding: a timed-out subprocess must
    actually be killed, not just have its ``communicate()`` await cancelled
    (which leaves the child running with its pipes held open)."""

    async def test_timed_out_gh_subprocess_is_killed(self):
        handler = HandlerEvidenceAutocloseSweep(linear_client=FakeLinearClient())
        # `sleep 5` outlives the 0.1s timeout; the real runner must kill it.
        data, error = await handler._run_gh_command_real(["sleep", "5"], timeout=0.1)
        assert data is None
        assert "Timeout" in error

    async def test_reap_helper_kills_a_still_running_process(self):
        import asyncio

        from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
            _reap_timed_out_process,
        )

        proc = await asyncio.create_subprocess_exec(
            "sleep",
            "30",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.returncode is None
        await _reap_timed_out_process(proc)
        await proc.wait()  # reap the zombie so the test doesn't leak it
        assert proc.returncode is not None

    async def test_reap_helper_is_a_noop_for_an_already_exited_process(self):
        import asyncio

        from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
            _reap_timed_out_process,
        )

        proc = await asyncio.create_subprocess_exec(
            "true", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.wait()
        # Must not raise (e.g. ProcessLookupError from killing an exited proc).
        await _reap_timed_out_process(proc)


@pytest.mark.unit
class TestRealSubprocessCreationFailure:
    """Regression coverage for CodeRabbit finding: `asyncio.create_subprocess_exec`
    itself can raise OSError (e.g. FileNotFoundError for a missing `gh`/`uv`
    binary) *before* there is any process to `communicate()` with or reap.
    Both real runners must catch that and return their fail-closed error
    tuple instead of letting the OSError escape uncaught."""

    _MISSING_BINARY = "__definitely_missing_executable_for_omn_16106_tests__"

    async def test_gh_runner_returns_error_tuple_on_missing_executable(self):
        handler = HandlerEvidenceAutocloseSweep(linear_client=FakeLinearClient())
        data, error = await handler._run_gh_command_real(
            [self._MISSING_BINARY, "api", "x"], timeout=1.0
        )
        assert data is None
        assert "OS error" in error
        assert self._MISSING_BINARY in error

    async def test_dod_verify_runner_returns_error_tuple_on_missing_cwd(self):
        # A cwd that does not exist makes process creation itself raise
        # FileNotFoundError, same failure class as a missing executable.
        handler = HandlerEvidenceAutocloseSweep(linear_client=FakeLinearClient())
        result, exit_code, error = await handler._run_dod_verify_command_real(
            "OMN-9999", cwd="/definitely/does/not/exist/omn-16106", timeout=1.0
        )
        assert result is None
        assert exit_code == -1
        assert "OS error" in error


@pytest.mark.unit
class TestOmn16905OutcomeRowReportsBehaviorProving:
    """The OUTCOME row must report the verdict's real `behavior_proving_count`.

    OMN-16905. In evidence-autoclose-sweep run 33210163405 the DIAGNOSE step
    and the sweep's own OUTCOME row disagreed on exactly one counter for one
    ticket (OMN-16803) in one job: `behavior_proving=1` versus
    `dod_verify_behavior_proving_count=0`, with total/verified/failed/
    non_probative identical. That was read as a classifier regression between
    two omnimarket builds (the gate venv vs the OMN-16846 dispatch venv).

    It is not. The handler reads `behavior_proving_count` off the verdict ONLY
    inside the `all_verified` branch. Every other exit -- and OMN-16803's
    verdict was `status=skipped`, so it took the gap path -- builds its outcome
    without the field, and `ModelEvidenceAutocloseOutcome` defaults it to 0. The
    OUTCOME row therefore reported a structural 0 that had nothing to do with
    what the classifier said, and the discrepancy was an artifact of which
    branch the ticket fell down, not of which interpreter ran it.

    This matters because the OUTCOME row is the machine-readable record the
    fleet is triaged from: a counter that reads 0 whenever the ticket is not
    already flippable cannot distinguish "no behavior proof was declared" from
    "behavior proof ran, passed, and something else held the flip". That
    ambiguity is what sent OMN-16905 hunting a classifier bug that does not
    exist.

    The flip predicate itself is unaffected -- it reads the verdict directly --
    so this is a reporting-fidelity fix, not a behavior change to any gate.
    """

    async def test_gap_path_outcome_row_carries_the_verdicts_behavior_count(self):
        """A gap-path ticket must still report the behavior count it measured."""
        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake(
            {
                "OMN-9999": (
                    _dod_verify_ok(total=3, verified=2, failed=1, behavior_proving=2),
                    1,
                    "",
                )
            }
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=False))
        outcome = result.outcomes[0]
        assert outcome.decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        # The defect: this read 0 while the verdict said 2.
        assert outcome.dod_verify_behavior_proving_count == 2

    async def test_omn_16803_shape_skipped_status_still_reports_behavior_one(self):
        """The exact run-33210163405 shape: status=skipped, behavior_proving=1.

        `failed == 0` but dod_verify's own terminal status is `skipped`, so
        `all_verified` is False and the ticket takes the gap path -- the precise
        combination that produced the reported 1-vs-0 divergence.
        """
        verdict = _dod_verify_ok(
            total=6,
            verified=2,
            failed=0,
            behavior_proving=1,
            verdict_status="skipped",
        )
        payload = _verdict_of(verdict)
        payload["skipped_count"] = 3
        payload["non_probative_count"] = 1

        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake({"OMN-9999": (verdict, 0, "")})
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=False))
        outcome = result.outcomes[0]
        assert outcome.decision == EnumEvidenceAutocloseDecision.GAP_POSTED
        # Every other counter already matched the diagnose leg; only this one
        # did not, and the mismatch is what this ticket exists to close.
        assert outcome.dod_verify_total_checks == 6
        assert outcome.dod_verify_verified_count == 2
        assert outcome.dod_verify_failed_count == 0
        assert outcome.dod_verify_non_probative_count == 1
        assert outcome.dod_verify_behavior_proving_count == 1

    @pytest.mark.parametrize(
        ("status", "total", "verified", "failed", "behavior"),
        [
            # Gap by a real red.
            ("failed", 3, 2, 1, 2),
            # Gap by dod_verify's own non-verified terminal status, behaviour
            # proof present -- the OMN-16803 shape.
            ("skipped", 6, 2, 0, 1),
            # Same shape, richer behaviour count.
            ("skipped", 11, 4, 0, 3),
            # A genuine zero must still be reported as zero, not confused with
            # the structural default this ticket removed.
            ("skipped", 4, 1, 0, 0),
        ],
    )
    async def test_outcome_row_never_diverges_from_the_verdict_it_read(
        self, status, total, verified, failed, behavior
    ):
        """AC5: the sweep's row and dod_verify's own dump stay comparable.

        The regression this locks is not "the number is wrong once" but "the
        two legs can disagree at all". OMN-16905 cost a full investigation into
        a nonexistent omnimarket classifier regression precisely because the
        OUTCOME row and the diagnose leg's dump were not comparable artifacts:
        one reported what dod_verify measured, the other reported a default. A
        future divergence must fail here rather than surface as a fleet that
        silently will not flip.
        """
        verdict = _dod_verify_ok(
            total=total,
            verified=verified,
            failed=failed,
            behavior_proving=behavior,
            verdict_status=status,
        )

        gh_fake = _make_gh_fake(
            companions=[_merged_pr(1, "evidence(OMN-9999): x", "OMN-9999")],
            files_by_pr={1: ["contracts/OMN-9999.yaml"]},
        )
        linear = FakeLinearClient(issues={"OMN-9999": _issue()})
        dod_fake = _make_dod_verify_fake(
            {"OMN-9999": (verdict, 1 if failed else 0, "")}
        )
        handler = HandlerEvidenceAutocloseSweep(
            linear_client=linear,
            run_gh_command=gh_fake,
            run_dod_verify_command=dod_fake,
        )
        result = await handler.handle(_request(apply=False))
        outcome = result.outcomes[0]

        # Every counter the row publishes must equal the verdict it read.
        assert outcome.dod_verify_total_checks == total
        assert outcome.dod_verify_verified_count == verified
        assert outcome.dod_verify_failed_count == failed
        assert outcome.dod_verify_behavior_proving_count == behavior


@pytest.mark.unit
class TestOmn16905EveryVerdictBearingOutcomeReportsBehavior:
    """AC5 structural gate: report every counter you read, or none of them.

    The OMN-16905 defect was not a wrong value, it was a SILENT one. The gap
    path built its outcome with four of dod_verify's five counters and let
    ``dod_verify_behavior_proving_count`` fall through to the model's
    ``default=0``. A default is indistinguishable, in the emitted JSON, from a
    measured zero -- so the row asserted "no behavior was proven" about a
    verdict it had never asked. That is what made the diagnose leg and the
    OUTCOME row disagree 1-vs-0 on OMN-16803 in run 33210163405 with the other
    four counters identical, and it is what sent the investigation after an
    omnimarket classifier regression that does not exist.

    The per-branch unit tests above pin today's branches. This one pins the
    INVARIANT, so a branch added tomorrow cannot reintroduce the hole by
    copying the old shape: any construction that reports
    ``dod_verify_total_checks`` has a verdict in hand, and must therefore also
    state what that verdict said about behavior -- explicitly, including the
    honest ``=0`` on the unparseable branch where the key is genuinely absent.

    Construction sites that report NO dod_verify counters at all (the
    pre-verdict paths: kill switch, unresolvable binding, already-Done) are out
    of scope by design -- they never read a verdict, so they have nothing to
    under-report.
    """

    def test_no_construction_reports_counters_while_omitting_behavior(self):
        import ast
        import inspect

        from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
            handler_evidence_autoclose_sweep as _mod,
        )

        source = Path(inspect.getfile(_mod)).read_text(encoding="utf-8")
        tree = ast.parse(source)

        offenders: list[int] = []
        verdict_bearing = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                not isinstance(func, ast.Name)
                or func.id != "ModelEvidenceAutocloseOutcome"
            ):
                continue
            kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
            if "dod_verify_total_checks" not in kwargs:
                continue
            verdict_bearing += 1
            if "dod_verify_behavior_proving_count" not in kwargs:
                offenders.append(node.lineno)

        # Guard the guard: if the construction shape is ever refactored away
        # (a builder, a dict splat), this test would silently pass on zero
        # sites and stop protecting anything.
        assert verdict_bearing >= 5, (
            "expected the handler to build several verdict-bearing outcomes; "
            f"found {verdict_bearing}. If the construction shape changed, this "
            "gate must be rewritten, not deleted -- OMN-16905."
        )
        assert offenders == [], (
            "ModelEvidenceAutocloseOutcome built with dod_verify counters but "
            "WITHOUT dod_verify_behavior_proving_count at line(s) "
            f"{offenders}. That lets the model default manufacture a 0 the "
            "verifier never reported, which is the exact OMN-16905 defect: an "
            "OUTCOME row that disagrees with dod_verify's own dump. Pass the "
            "measured value, or an explicit 0 with a comment saying why."
        )
