# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19566 T2 slice 1: exact-head PR lab-proof receipts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.unit

HEAD_SHA = "a" * 40
BASE_SHA = "b" * 40
MERGE_BASE_SHA = "c" * 40
CARRIED_FROM_SHA = "d" * 40
DIFF_DIGEST = f"sha256:{'e' * 64}"
OTHER_DIFF_DIGEST = f"sha256:{'f' * 64}"
REPO = "OmniNode-ai/omnibase_infra"
PR_NUMBER = 19566
PROFILE_ID = "omnibase-infra-pr-head"
PROFILE_VERSION = "v1"
STARTED = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
FINISHED = datetime(2026, 9, 26, 12, 5, tzinfo=UTC)


def _subject(
    *,
    repo: str = REPO,
    pr_number: int = PR_NUMBER,
    profile_id: str = PROFILE_ID,
    profile_version: str = PROFILE_VERSION,
    handler_kind: receipts.EnumLabProofHandlerKind | None = None,
    runner_identity: str = "github-runner:42",
    verifier_identity: str = "omnibase-pr-head-verifier",
    pr_diff_digest: str = DIFF_DIGEST,
    carried_from: str = "",
) -> receipts.ModelLabProofSubject:
    return receipts.ModelLabProofSubject(
        repo=repo,
        pr_number=pr_number,
        base_sha=BASE_SHA,
        merge_base_sha=MERGE_BASE_SHA,
        profile_id=profile_id,
        profile_version=profile_version,
        handler_kind=(
            handler_kind
            if handler_kind is not None
            else receipts.EnumLabProofHandlerKind.SCRIPT_REPLAY
        ),
        host="github-actions",
        slot="pr-head",
        runner_identity=runner_identity,
        verifier_identity=verifier_identity,
        pr_diff_digest=pr_diff_digest,
        carried_from=carried_from,
    )


def _check(name: str = "unit", *, ok: bool = True) -> receipts.ModelLabPassCheck:
    return receipts.ModelLabPassCheck(name=name, ok=ok, evidence=f"{name} evidence")


def _receipt(
    *,
    sha: str = HEAD_SHA,
    subject: receipts.ModelLabProofSubject | None = None,
    checks: tuple[receipts.ModelLabPassCheck, ...] | None = None,
) -> receipts.ModelLabPassReceipt:
    selected_checks = checks if checks is not None else (_check(),)
    return receipts.ModelLabPassReceipt(
        sha=sha,
        lane=receipts.EnumLabLane.PR_HEAD,
        started_at=STARTED,
        finished_at=FINISHED,
        result=(
            receipts.EnumLabPassResult.PASS
            if all(check.ok for check in selected_checks)
            else receipts.EnumLabPassResult.FAIL
        ),
        checks=selected_checks,
        agent_command_id=None,
        subject=subject if subject is not None else _subject(),
    )


def _verify(
    receipt: receipts.ModelLabPassReceipt,
    **overrides: Any,
) -> tuple[receipts.EnumPrHeadVerdict, str]:
    inputs: dict[str, Any] = {
        "expected_repo": REPO,
        "expected_pr_number": PR_NUMBER,
        "expected_head_sha": HEAD_SHA,
        "expected_profile_id": PROFILE_ID,
        "expected_profile_version": PROFILE_VERSION,
        "mandatory_checks": frozenset({"unit"}),
        "current_pr_diff_digest": DIFF_DIGEST,
    }
    inputs.update(overrides)
    return receipts.verify_pr_head_receipt(receipt, **inputs)


class TestPrHeadLaneIsolation:
    def test_pr_head_is_not_a_rule_24b_default_lane(self) -> None:
        assert receipts.EnumLabLane.PR_HEAD not in receipts.ANY_OF_DEFAULT_LANES

    def test_an_unqualified_gate_never_reads_the_pr_head_lane(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        observed: dict[str, object] = {}

        def fake_evaluate_gate(
            repo: str,
            sha: str,
            lanes: list[receipts.EnumLabLane],
            out: object,
            **kwargs: object,
        ) -> int:
            observed.update(repo=repo, sha=sha, lanes=lanes, out=out, kwargs=kwargs)
            return 1

        monkeypatch.setattr(receipts, "evaluate_gate", fake_evaluate_gate)

        assert receipts.main(["gate", "--sha", HEAD_SHA]) == 1
        assert receipts.EnumLabLane.PR_HEAD not in observed["lanes"]


class TestPrHeadSubjectContract:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("repo", "other/omnibase_infra"),
            ("pr_number", 0),
            ("base_sha", "ABC"),
            ("merge_base_sha", "c" * 39),
            ("profile_id", ""),
            ("profile_version", ""),
            ("host", ""),
            ("slot", ""),
            ("runner_identity", ""),
            ("verifier_identity", ""),
            ("pr_diff_digest", "e" * 64),
            ("carried_from", "D" * 40),
        ],
    )
    def test_invalid_subject_fields_are_refused(
        self, field: str, value: object
    ) -> None:
        values = {
            "repo": REPO,
            "pr_number": PR_NUMBER,
            "base_sha": BASE_SHA,
            "merge_base_sha": MERGE_BASE_SHA,
            "profile_id": PROFILE_ID,
            "profile_version": PROFILE_VERSION,
            "handler_kind": receipts.EnumLabProofHandlerKind.SCRIPT_REPLAY,
            "host": "github-actions",
            "slot": "pr-head",
            "runner_identity": "github-runner:42",
            "verifier_identity": "omnibase-pr-head-verifier",
            "pr_diff_digest": DIFF_DIGEST,
            "carried_from": "",
        }
        values[field] = value
        with pytest.raises(ValueError):
            receipts.ModelLabProofSubject(**values)

    def test_pr_head_receipt_requires_a_subject(self) -> None:
        with pytest.raises(ValueError, match="subject"):
            receipts.ModelLabPassReceipt(
                sha=HEAD_SHA,
                lane=receipts.EnumLabLane.PR_HEAD,
                started_at=STARTED,
                finished_at=FINISHED,
                result=receipts.EnumLabPassResult.PASS,
                checks=(_check(),),
                agent_command_id=None,
            )

    def test_a_post_merge_lane_refuses_a_pr_head_subject(self) -> None:
        with pytest.raises(ValueError, match="subject"):
            receipts.ModelLabPassReceipt(
                sha=HEAD_SHA,
                lane=receipts.EnumLabLane.COMPOSE_DEV,
                started_at=STARTED,
                finished_at=FINISHED,
                result=receipts.EnumLabPassResult.PASS,
                checks=(_check(),),
                agent_command_id=None,
                subject=_subject(),
            )

    def test_carried_from_must_differ_from_the_receipt_head(self) -> None:
        with pytest.raises(ValueError, match="carried_from"):
            _receipt(subject=_subject(carried_from=HEAD_SHA))

    def test_existing_shape_round_trips_byte_identically_without_subject(self) -> None:
        body = json.dumps(
            {
                "receipt_version": receipts.RECEIPT_VERSION,
                "sha": HEAD_SHA,
                "lane": receipts.EnumLabLane.COMPOSE_DEV.value,
                "started_at": STARTED.isoformat(),
                "finished_at": FINISHED.isoformat(),
                "result": receipts.EnumLabPassResult.PASS.value,
                "checks": [_check().to_dict()],
                "agent_command_id": None,
            }
        )

        parsed = receipts.parse_receipt(body)

        assert parsed.subject is None
        assert parsed.to_json() == body

    def test_pr_head_subject_is_written_and_parsed(self) -> None:
        original = _receipt(subject=_subject(carried_from=CARRIED_FROM_SHA))

        parsed = receipts.parse_receipt(original.to_json())

        assert parsed == original
        assert json.loads(original.to_json())["subject"]["repo"] == REPO


class TestPrHeadReceiptKey:
    def test_key_contains_the_full_pr_head_identity(self) -> None:
        assert receipts.pr_head_receipt_key(_receipt()) == (
            REPO,
            PR_NUMBER,
            HEAD_SHA,
            PROFILE_ID,
            PROFILE_VERSION,
        )

    def test_a_non_pr_head_receipt_has_no_pr_head_key(self) -> None:
        post_merge = receipts.build_receipt(
            sha=HEAD_SHA,
            lane=receipts.EnumLabLane.COMPOSE_DEV,
            started_at=STARTED,
            finished_at=FINISHED,
            checks=[_check()],
            agent_command_id=None,
        )
        with pytest.raises(ValueError, match="pr-head"):
            receipts.pr_head_receipt_key(post_merge)


class TestPrDiffDigest:
    def test_bytes_are_hashed_with_the_sha256_prefix(self) -> None:
        body = b"diff --git a/file.txt b/file.txt\n+proof\n"
        assert receipts.compute_pr_diff_digest(body) == (
            f"sha256:{hashlib.sha256(body).hexdigest()}"
        )

    def test_digest_is_computed_from_the_exact_git_diff(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(
            ["git", "init", "-q"], cwd=repo, check=True, env=scrub_git_location_env()
        )
        tracked = repo / "proof.txt"
        tracked.write_text("base\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "proof.txt"],
            cwd=repo,
            check=True,
            env=scrub_git_location_env(),
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Receipt Test",
                "-c",
                "user.email=receipt@example.invalid",
                "commit",
                "-qm",
                "base",
            ],
            cwd=repo,
            env=scrub_git_location_env(),
            check=True,
        )
        merge_base = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            env=scrub_git_location_env(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked.write_text("head\n", encoding="utf-8")
        subprocess.run(
            ["git", "add", "proof.txt"],
            cwd=repo,
            check=True,
            env=scrub_git_location_env(),
        )
        subprocess.run(
            [
                "git",
                "-c",
                "user.name=Receipt Test",
                "-c",
                "user.email=receipt@example.invalid",
                "commit",
                "-qm",
                "head",
            ],
            cwd=repo,
            env=scrub_git_location_env(),
            check=True,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            env=scrub_git_location_env(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--no-color", "--no-ext-diff", f"{merge_base}..{head}"],
            cwd=repo,
            env=scrub_git_location_env(),
            check=True,
            capture_output=True,
        ).stdout

        assert receipts.compute_pr_diff_digest_from_repo(repo, merge_base, head) == (
            receipts.compute_pr_diff_digest(diff)
        )


class TestPrHeadVerifier:
    def test_a_fully_valid_receipt_is_accepted(self) -> None:
        verdict, reason = _verify(_receipt())
        assert verdict is receipts.EnumPrHeadVerdict.ACCEPTED
        assert "accepted" in reason.lower()

    def test_a_non_pr_head_lane_is_refused_with_its_own_token(self) -> None:
        receipt = receipts.build_receipt(
            sha=HEAD_SHA,
            lane=receipts.EnumLabLane.COMPOSE_DEV,
            started_at=STARTED,
            finished_at=FINISHED,
            checks=[_check()],
            agent_command_id=None,
        )
        assert _verify(receipt)[0] is receipts.EnumPrHeadVerdict.NOT_PR_HEAD

    def test_a_different_head_is_refused_with_its_own_token(self) -> None:
        assert (
            _verify(_receipt(sha="9" * 40))[0]
            is receipts.EnumPrHeadVerdict.HEAD_MISMATCH
        )

    def test_a_fail_result_is_refused_with_its_own_token(self) -> None:
        receipt = _receipt(checks=(_check(ok=False),))
        assert _verify(receipt)[0] is receipts.EnumPrHeadVerdict.RESULT_NOT_PASS

    @pytest.mark.parametrize(
        "override",
        [
            {"expected_repo": "OmniNode-ai/other"},
            {"expected_pr_number": 99},
            {"expected_profile_id": "other-profile"},
            {"expected_profile_version": "v2"},
        ],
        ids=["repo", "pr-number", "profile-id", "profile-version"],
    )
    def test_a_wrong_profile_binding_is_refused_with_its_own_token(
        self, override: dict[str, object]
    ) -> None:
        assert (
            _verify(_receipt(), **override)[0]
            is receipts.EnumPrHeadVerdict.PROFILE_MISMATCH
        )

    def test_a_missing_mandatory_check_is_named_and_refused_with_its_own_token(
        self,
    ) -> None:
        verdict, reason = _verify(
            _receipt(), mandatory_checks=frozenset({"unit", "integration", "lint"})
        )
        assert verdict is receipts.EnumPrHeadVerdict.MISSING_MANDATORY_CHECK
        assert "integration" in reason
        assert "lint" in reason

    def test_an_empty_mandatory_check_set_is_refused_not_vacuously_accepted(
        self,
    ) -> None:
        verdict, reason = _verify(_receipt(), mandatory_checks=frozenset())
        assert verdict is receipts.EnumPrHeadVerdict.MISSING_MANDATORY_CHECK
        assert "no mandatory checks" in reason

    def test_the_runner_cannot_also_be_the_verifier(self) -> None:
        receipt = _receipt(
            subject=_subject(
                runner_identity="  GitHub-Runner:42 ",
                verifier_identity="github-runner:42",
            )
        )
        assert _verify(receipt)[0] is receipts.EnumPrHeadVerdict.RUNNER_IS_VERIFIER

    @pytest.mark.parametrize(
        "kind",
        [
            receipts.EnumLabProofHandlerKind.RUNTIME_IMAGE,
            receipts.EnumLabProofHandlerKind.FOUNDATION_OVERRIDE,
            receipts.EnumLabProofHandlerKind.PYPI_SIBLING_OVERRIDE,
            receipts.EnumLabProofHandlerKind.K8S_NAMESPACE,
        ],
    )
    def test_a_runtime_profile_cannot_carry_proof_from_another_head(
        self, kind: receipts.EnumLabProofHandlerKind
    ) -> None:
        receipt = _receipt(
            subject=_subject(handler_kind=kind, carried_from=CARRIED_FROM_SHA)
        )
        assert (
            _verify(receipt)[0] is receipts.EnumPrHeadVerdict.CARRY_OVER_RUNTIME_PROFILE
        )

    def test_a_changed_diff_refuses_non_runtime_carry_over(self) -> None:
        receipt = _receipt(subject=_subject(carried_from=CARRIED_FROM_SHA))
        assert (
            _verify(receipt, current_pr_diff_digest=OTHER_DIFF_DIGEST)[0]
            is receipts.EnumPrHeadVerdict.CARRY_OVER_DIGEST_MISMATCH
        )

    @pytest.mark.parametrize(
        "kind",
        [
            receipts.EnumLabProofHandlerKind.SCRIPT_REPLAY,
            receipts.EnumLabProofHandlerKind.WEB_RENDER,
            receipts.EnumLabProofHandlerKind.EXEMPT,
        ],
    )
    def test_an_unchanged_diff_accepts_non_runtime_carry_over(
        self, kind: receipts.EnumLabProofHandlerKind
    ) -> None:
        receipt = _receipt(
            subject=_subject(handler_kind=kind, carried_from=CARRIED_FROM_SHA)
        )
        assert _verify(receipt)[0] is receipts.EnumPrHeadVerdict.ACCEPTED


class TestVerifyPrHeadCli:
    def _args(self, path: Path) -> list[str]:
        return [
            "verify-pr-head",
            "--receipt",
            str(path),
            "--repo",
            REPO,
            "--pr",
            str(PR_NUMBER),
            "--head-sha",
            HEAD_SHA,
            "--profile-id",
            PROFILE_ID,
            "--profile-version",
            PROFILE_VERSION,
            "--mandatory-check",
            "unit",
            "--current-pr-diff-digest",
            DIFF_DIGEST,
        ]

    def test_cli_prints_one_json_line_and_accepts(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "receipt.json"
        path.write_text(_receipt().to_json(), encoding="utf-8")

        assert receipts.main(self._args(path)) == 0
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["token"] == receipts.EnumPrHeadVerdict.ACCEPTED

    def test_cli_returns_one_for_a_refusal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "receipt.json"
        path.write_text(_receipt(sha="9" * 40).to_json(), encoding="utf-8")

        assert receipts.main(self._args(path)) == 1
        output = json.loads(capsys.readouterr().out)
        assert output["token"] == receipts.EnumPrHeadVerdict.HEAD_MISMATCH
        assert output["reason"]

    def test_cli_returns_two_for_unreadable_input(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "missing.json"

        assert receipts.main(self._args(path)) == 2
        output = json.loads(capsys.readouterr().out)
        assert output["token"] == "UNREADABLE"
        assert output["reason"]

    def test_cli_returns_two_for_non_utf8_input(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "receipt.json"
        path.write_bytes(b"\xff\xfe")

        assert receipts.main(self._args(path)) == 2
        output = json.loads(capsys.readouterr().out)
        assert output["token"] == "UNREADABLE"
        assert output["reason"]
