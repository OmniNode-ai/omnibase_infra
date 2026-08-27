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

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
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


def _dod_verify_ok(
    *, total: int, verified: int, failed: int, skipped: int = 0
) -> dict[str, object]:
    return {
        "skill_name": "dod_verify",
        "node_name": "node_dod_verify",
        "status": "success",
        "correlation_id": str(uuid4()),
        "run_id": str(uuid4()),
        "exit_code": 0,
        "duration_ms": 1,
        "result": {
            "status": "verified" if failed == 0 else "failed",
            "total_checks": total,
            "verified_count": verified,
            "failed_count": failed,
            "skipped_count": skipped,
        },
        "result_model": "omnimarket.nodes.node_dod_verify.models.model_dod_verify_state.ModelDodVerifyState",
    }


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


def _issue(
    *,
    issue_id: str = "issue-uuid-1",
    state_type: str = "started",
    labels: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "id": issue_id,
        "identifier": "OMN-9999",
        "state": {"id": "state-1", "name": "In Progress", "type": state_type},
        "labels": {"nodes": [{"name": label} for label in labels]},
        "team": {"id": "team-1"},
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
            {"OMN-9999": (_dod_verify_ok(total=3, verified=2, failed=1), 0, "")}
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
            {"OMN-9999": (_dod_verify_ok(total=3, verified=2, failed=1), 0, "")}
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
