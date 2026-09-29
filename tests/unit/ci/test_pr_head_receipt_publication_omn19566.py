# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19566 T2 slices 2 and 3: a pr-head receipt on the bus and on the PR head.

Slice 2 is the bus event the ``lab_proof_receipts`` projection folds. Slice 3 is
the informational ``lab-proof-receipt`` check run, posted on the PROVEN head.
The falsifiers the plan names are each held to their own outcome token here: a
PASS missing a mandatory check, a runner that is its own verifier, a carried
receipt on a runtime profile, and a head that moved after its proof.
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from scripts.ci import lab_pass_receipt as receipts

pytestmark = pytest.mark.unit

HEAD_SHA = "a" * 40
MOVED_HEAD_SHA = "9" * 40
BASE_SHA = "b" * 40
MERGE_BASE_SHA = "c" * 40
CARRIED_FROM_SHA = "d" * 40
DIFF_DIGEST = f"sha256:{'e' * 64}"
REPO = "OmniNode-ai/omnimarket"
PR_NUMBER = 3042
PROFILE_ID = "omnimarket.runtime_image"
PROFILE_VERSION = "1"
MANDATORY = frozenset({"runtime_main_healthy", "changed_path_live"})
STARTED = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)
FINISHED = datetime(2026, 9, 28, 10, 16, tzinfo=UTC)


def _subject(
    *,
    runner_identity: str = "plan-item-I7@lab-202",
    verifier_identity: str = "prepr_runtime_pool.judge@operator-mac",
    handler_kind: receipts.EnumLabProofHandlerKind = (
        receipts.EnumLabProofHandlerKind.RUNTIME_IMAGE
    ),
    carried_from: str = "",
) -> receipts.ModelLabProofSubject:
    return receipts.ModelLabProofSubject(
        repo=REPO,
        pr_number=PR_NUMBER,
        base_sha=BASE_SHA,
        merge_base_sha=MERGE_BASE_SHA,
        profile_id=PROFILE_ID,
        profile_version=PROFILE_VERSION,
        handler_kind=handler_kind,
        host="lab-202",
        slot="isolated omnibase-infra-local",
        runner_identity=runner_identity,
        verifier_identity=verifier_identity,
        pr_diff_digest=DIFF_DIGEST,
        carried_from=carried_from,
    )


def _receipt(
    *,
    checks: dict[str, bool] | None = None,
    subject: receipts.ModelLabProofSubject | None = None,
) -> receipts.ModelLabPassReceipt:
    names = checks if checks is not None else dict.fromkeys(MANDATORY, True)
    rows = tuple(
        receipts.ModelLabPassCheck(name=n, ok=ok, evidence=f"{n} read")
        for n, ok in sorted(names.items())
    )
    return receipts.ModelLabPassReceipt(
        sha=HEAD_SHA,
        lane=receipts.EnumLabLane.PR_HEAD,
        started_at=STARTED,
        finished_at=FINISHED,
        result=(
            receipts.EnumLabPassResult.PASS
            if all(names.values())
            else receipts.EnumLabPassResult.FAIL
        ),
        checks=rows,
        agent_command_id=None,
        subject=subject if subject is not None else _subject(),
    )


def _at_live(
    receipt: receipts.ModelLabPassReceipt, live: str = HEAD_SHA
) -> tuple[receipts.EnumPrHeadVerdict, str]:
    return receipts.verify_pr_head_receipt_at_live_head(
        receipt,
        live_head_sha=live,
        mandatory_checks=MANDATORY,
        current_pr_diff_digest=DIFF_DIGEST,
    )


def _check_run(
    receipt: receipts.ModelLabPassReceipt, live: str = HEAD_SHA
) -> dict[str, Any]:
    verdict, reason = _at_live(receipt, live)
    return receipts.render_pr_head_check_run(
        receipt, verdict=verdict, reason=reason, mandatory_checks=MANDATORY
    )


class TestTheFalsifiers:
    """RED first: each refusal the plan names, under its own token."""

    def test_a_pass_missing_a_mandatory_check_is_refused(self) -> None:
        receipt = _receipt(checks={"runtime_main_healthy": True})
        verdict, reason = _at_live(receipt)
        assert receipt.result is receipts.EnumLabPassResult.PASS
        assert verdict is receipts.EnumPrHeadVerdict.MISSING_MANDATORY_CHECK
        assert "changed_path_live" in reason
        body = _check_run(receipt)
        assert body["conclusion"] == "failure"
        assert "MISSING_MANDATORY_CHECK" in body["output"]["title"]
        assert "changed_path_live" in body["output"]["summary"]

    def test_a_runner_that_is_its_own_verifier_is_refused(self) -> None:
        receipt = _receipt(
            subject=_subject(
                runner_identity="lane-x@lab-202", verifier_identity=" LANE-X@lab-202 "
            )
        )
        verdict, _ = _at_live(receipt)
        assert verdict is receipts.EnumPrHeadVerdict.RUNNER_IS_VERIFIER
        assert _check_run(receipt)["conclusion"] == "failure"

    def test_a_carried_receipt_on_a_runtime_image_profile_is_refused(self) -> None:
        receipt = _receipt(subject=_subject(carried_from=CARRIED_FROM_SHA))
        verdict, _ = _at_live(receipt)
        assert verdict is receipts.EnumPrHeadVerdict.CARRY_OVER_RUNTIME_PROFILE
        body = _check_run(receipt)
        assert body["conclusion"] == "failure"
        assert CARRIED_FROM_SHA in body["output"]["summary"]

    def test_a_moved_head_is_head_mismatch_reported_on_the_proven_sha(self) -> None:
        receipt = _receipt()
        verdict, _ = _at_live(receipt, MOVED_HEAD_SHA)
        assert verdict is receipts.EnumPrHeadVerdict.HEAD_MISMATCH
        body = _check_run(receipt, MOVED_HEAD_SHA)
        # never posted on a head the proof did not run on
        assert body["head_sha"] == HEAD_SHA
        assert body["conclusion"] == "failure"
        assert body["output"]["title"].startswith("PASS HEAD_MISMATCH")


class TestTheBusEvent:
    def test_an_accepted_pass_carries_its_key_topic_and_whole_receipt(self) -> None:
        receipt = _receipt()
        verdict, reason = _at_live(receipt)
        event = receipts.build_pr_head_bus_event(
            receipt, verdict=verdict, reason=reason, mandatory_checks=MANDATORY
        )
        assert event["topic"] == receipts.LAB_PROOF_RECEIPT_EVENT_TOPIC
        assert event["topic"] == "onex.evt.omnibase-infra.lab-proof-receipt.v1"
        assert event["lane"] == "pr-head"
        assert event["receipt_key"] == (
            f"{REPO}#{PR_NUMBER}@{HEAD_SHA}:{PROFILE_ID}@{PROFILE_VERSION}"
        )
        assert event["verifier_token"] == "ACCEPTED"
        assert event["result"] == "PASS"
        assert event["missing_mandatory_checks"] == []
        assert event["handler_kind"] == "runtime_image"
        # the whole minted receipt travels, so a consumer can re-verify it
        reparsed = receipts.ModelLabPassReceipt.from_json(json.dumps(event["receipt"]))
        assert reparsed == receipt

    def test_a_fail_receipt_is_published_with_its_failing_checks(self) -> None:
        receipt = _receipt(
            checks={"runtime_main_healthy": False, "changed_path_live": True}
        )
        verdict, reason = _at_live(receipt)
        event = receipts.build_pr_head_bus_event(
            receipt, verdict=verdict, reason=reason, mandatory_checks=MANDATORY
        )
        assert event["result"] == "FAIL"
        assert event["verifier_token"] == "RESULT_NOT_PASS"
        assert event["failing_checks"] == ["runtime_main_healthy"]
        assert event["missing_mandatory_checks"] == ["runtime_main_healthy"]

    def test_a_post_merge_receipt_is_never_published_as_a_pr_head_proof(self) -> None:
        post_merge = receipts.ModelLabPassReceipt(
            sha=HEAD_SHA,
            lane=receipts.EnumLabLane.COMPOSE_DEV,
            started_at=STARTED,
            finished_at=FINISHED,
            result=receipts.EnumLabPassResult.PASS,
            checks=(receipts.ModelLabPassCheck(name="x", ok=True, evidence="e"),),
            agent_command_id=None,
        )
        with pytest.raises(ValueError, match="only a pr-head receipt"):
            receipts.build_pr_head_bus_event(
                post_merge,
                verdict=receipts.EnumPrHeadVerdict.ACCEPTED,
                reason="r",
                mandatory_checks=MANDATORY,
            )
        with pytest.raises(ValueError, match="only a pr-head receipt"):
            receipts.render_pr_head_check_run(
                post_merge,
                verdict=receipts.EnumPrHeadVerdict.ACCEPTED,
                reason="r",
                mandatory_checks=MANDATORY,
            )
        verdict, _ = receipts.verify_pr_head_receipt_at_live_head(
            post_merge,
            live_head_sha=HEAD_SHA,
            mandatory_checks=MANDATORY,
            current_pr_diff_digest=DIFF_DIGEST,
        )
        assert verdict is receipts.EnumPrHeadVerdict.NOT_PR_HEAD


class TestTheCheckRun:
    def test_only_accepted_is_success_and_the_check_is_never_the_required_name(
        self,
    ) -> None:
        body = _check_run(_receipt())
        assert body["name"] == "lab-proof-receipt"
        # the required context is the separate lab-proof job (OMN-19584)
        assert body["name"] != "lab-proof"
        assert body["conclusion"] == "success"
        assert body["status"] == "completed"
        assert body["head_sha"] == HEAD_SHA
        assert body["external_id"] == (
            f"{REPO}#{PR_NUMBER}@{HEAD_SHA}:{PROFILE_ID}@{PROFILE_VERSION}"
        )
        assert body["started_at"] == "2026-09-28T10:00:00Z"
        assert body["completed_at"] == "2026-09-28T10:16:00Z"
        assert (
            "| runtime_main_healthy | pass | runtime_main_healthy read |"
            in (body["output"]["text"])
        )

    def test_a_long_evidence_table_is_truncated_to_githubs_limit(self) -> None:
        many = {f"check_{i:04d}": True for i in range(3000)}
        many.update(dict.fromkeys(MANDATORY, True))
        body = _check_run(_receipt(checks=many))
        assert len(body["output"]["text"]) <= 65535
        assert body["output"]["text"].endswith("(truncated)")


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TestThePost:
    def test_the_post_goes_to_the_subject_repo_with_the_app_token(self) -> None:
        seen: dict[str, Any] = {}

        def urlopen(request: Any, timeout: float) -> _Response:
            seen["url"] = request.full_url
            seen["method"] = request.get_method()
            seen["auth"] = request.get_header("Authorization")
            seen["body"] = json.loads(request.data)
            return _Response(b'{"id": 7, "html_url": "https://x/runs/7"}')

        body = _check_run(_receipt())
        answer = receipts.post_check_run(
            body, repo=REPO, token="ghs_fake", urlopen=urlopen
        )
        assert answer["id"] == 7
        assert seen["url"] == f"https://api.github.com/repos/{REPO}/check-runs"
        assert seen["method"] == "POST"
        assert seen["auth"] == "Bearer ghs_fake"
        assert seen["body"]["head_sha"] == HEAD_SHA

    def test_a_refusal_is_raised_with_its_status_never_swallowed(self) -> None:
        def urlopen(request: Any, timeout: float) -> _Response:
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "Forbidden",
                {},  # type: ignore[arg-type]
                io.BytesIO(b'{"message": "You must authenticate via a GitHub App."}'),
            )

        with pytest.raises(receipts.CheckRunPostError) as caught:
            receipts.post_check_run(
                _check_run(_receipt()), repo=REPO, token="ghp_user", urlopen=urlopen
            )
        assert caught.value.status == 403
        assert "GitHub App" in str(caught.value)

    def test_no_token_and_a_foreign_repo_are_refused_before_any_request(
        self,
    ) -> None:
        def urlopen(request: Any, timeout: float) -> _Response:
            raise AssertionError("no request may be made")

        body = _check_run(_receipt())
        with pytest.raises(receipts.CheckRunPostError, match="token is required"):
            receipts.post_check_run(body, repo=REPO, token="", urlopen=urlopen)
        with pytest.raises(receipts.CheckRunPostError, match="OmniNode-ai"):
            receipts.post_check_run(
                body, repo="someone/else", token="t", urlopen=urlopen
            )


class TestTheCli:
    def _write(self, tmp_path: Path, receipt: receipts.ModelLabPassReceipt) -> Path:
        path = tmp_path / "receipt.json"
        path.write_text(receipt.to_json(), encoding="utf-8")
        return path

    def _mandatory_args(self) -> list[str]:
        args: list[str] = []
        for name in sorted(MANDATORY):
            args += ["--mandatory-check", name]
        return args

    def test_pr_head_event_writes_the_event_and_prints_the_token(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "e" / "event.json"
        code = receipts.main(
            [
                "pr-head-event",
                "--receipt",
                str(self._write(tmp_path, _receipt())),
                "--head-sha",
                HEAD_SHA,
                *self._mandatory_args(),
                "--current-pr-diff-digest",
                DIFF_DIGEST,
                "--out",
                str(out),
            ]
        )
        assert code == 0
        assert json.loads(capsys.readouterr().out)["token"] == "ACCEPTED"
        assert json.loads(out.read_text())["verifier_token"] == "ACCEPTED"

    def test_dry_run_prints_the_body_and_posts_nothing(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("a dry run must not post")

        monkeypatch.setattr(receipts, "post_check_run", refuse)
        code = receipts.main(
            [
                "post-pr-head-check",
                "--receipt",
                str(self._write(tmp_path, _receipt())),
                "--head-sha",
                MOVED_HEAD_SHA,
                *self._mandatory_args(),
                "--current-pr-diff-digest",
                DIFF_DIGEST,
                "--dry-run",
            ]
        )
        assert code == 0
        body = json.loads(capsys.readouterr().out)
        assert body["head_sha"] == HEAD_SHA
        assert body["output"]["title"].startswith("PASS HEAD_MISMATCH")

    def test_a_post_with_no_token_exits_two_and_says_why(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("LAB_PROOF_CHECK_TOKEN", raising=False)
        code = receipts.main(
            [
                "post-pr-head-check",
                "--receipt",
                str(self._write(tmp_path, _receipt())),
                "--head-sha",
                HEAD_SHA,
                *self._mandatory_args(),
                "--current-pr-diff-digest",
                DIFF_DIGEST,
            ]
        )
        assert code == 2
        said = json.loads(capsys.readouterr().out)
        assert said["posted"] is False
        assert "token is required" in said["error"]


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY = REPO_ROOT / "config" / "lab_proof_profiles.yaml"
VALIDATOR = REPO_ROOT / "scripts" / "ci" / "validate_lab_proof_profiles.py"


def _pin_json(tmp_path: Path, repo: str = REPO, kind: str = "runtime_image") -> Path:
    """The pin exactly as the workflow makes it: the validator's --pin output."""
    import subprocess
    import sys

    done = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--registry",
            str(REGISTRY),
            "--pin",
            repo,
            kind,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    path = tmp_path / "pin.json"
    path.write_text(done.stdout, encoding="utf-8")
    return path


class TestTheRegistryPin:
    """The workflow path judges a receipt by the reviewed registry, not its caller."""

    def _registry_checks(self, tmp_path: Path) -> frozenset[str]:
        pin = receipts.read_profile_pin(_pin_json(tmp_path))
        assert pin.profile_key == PROFILE_ID
        assert pin.profile_version == PROFILE_VERSION
        return pin.mandatory_checks

    def _post(
        self,
        tmp_path: Path,
        receipt: receipts.ModelLabPassReceipt,
        capsys: pytest.CaptureFixture[str],
        pin: Path | None = None,
    ) -> tuple[int, dict[str, Any]]:
        path = tmp_path / "receipt.json"
        path.write_text(receipt.to_json(), encoding="utf-8")
        code = receipts.main(
            [
                "post-pr-head-check",
                "--receipt",
                str(path),
                "--head-sha",
                HEAD_SHA,
                "--profile-pin",
                str(pin or _pin_json(tmp_path)),
                "--current-pr-diff-digest",
                DIFF_DIGEST,
                "--dry-run",
            ]
        )
        return code, json.loads(capsys.readouterr().out)

    def test_every_registry_check_passing_is_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        checks = dict.fromkeys(self._registry_checks(tmp_path), True)
        code, body = self._post(tmp_path, _receipt(checks=checks), capsys)
        assert code == 0
        assert body["conclusion"] == "success"

    def test_the_callers_shorter_list_cannot_weaken_the_registry(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # passes the two checks MANDATORY names, and nothing else the registry
        # demands: accepted under the short list, refused under the registry
        code, body = self._post(tmp_path, _receipt(), capsys)
        assert code == 0
        assert body["conclusion"] == "failure"
        assert "MISSING_MANDATORY_CHECK" in body["output"]["title"]

    def test_a_receipt_minted_under_another_profile_version_is_a_mismatch(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        checks = dict.fromkeys(self._registry_checks(tmp_path), True)
        stale = receipts.ModelLabProofSubject(
            **{**_subject().__dict__, "profile_version": "0"}
        )
        code, body = self._post(
            tmp_path, _receipt(checks=checks, subject=stale), capsys
        )
        assert code == 0
        assert "PROFILE_MISMATCH" in body["output"]["title"]

    def test_a_pin_for_another_repository_is_refused_before_any_verdict(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        other = _pin_json(tmp_path, repo="OmniNode-ai/omnibase_infra")
        code, said = self._post(tmp_path, _receipt(), capsys, pin=other)
        assert code == 1
        assert said["token"] == "PROFILE_UNRESOLVED"

    def test_the_validator_refuses_a_repository_with_no_row(
        self, tmp_path: Path
    ) -> None:
        import subprocess
        import sys

        done = subprocess.run(
            [
                sys.executable,
                str(VALIDATOR),
                "--registry",
                str(REGISTRY),
                "--pin",
                "OmniNode-ai/not-a-repo",
                "runtime_image",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 1
        assert "no profile pin" in done.stderr

    def test_no_profile_source_at_all_is_a_usage_error(self, tmp_path: Path) -> None:
        path = tmp_path / "receipt.json"
        path.write_text(_receipt().to_json(), encoding="utf-8")
        with pytest.raises(SystemExit) as exc:
            receipts.main(
                [
                    "post-pr-head-check",
                    "--receipt",
                    str(path),
                    "--head-sha",
                    HEAD_SHA,
                    "--current-pr-diff-digest",
                    DIFF_DIGEST,
                    "--dry-run",
                ]
            )
        assert exc.value.code == 2


def test_pr_diff_digest_is_the_verifiers_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import os
    import subprocess

    from omnibase_core.validators.no_unguarded_git_subprocess import (
        scrub_git_location_env,
    )

    def git(*argv: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@lab.invalid", *argv],
            cwd=tmp_path,
            env=scrub_git_location_env(os.environ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    (tmp_path / "a.txt").write_text("one\n")
    git("add", "a.txt")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "a.txt").write_text("two\n")
    git("commit", "-qam", "head")
    head = git("rev-parse", "HEAD")
    code = receipts.main(
        [
            "pr-diff-digest",
            "--repo-dir",
            str(tmp_path),
            "--merge-base",
            base,
            "--head",
            head,
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out.strip()
    assert printed == receipts.compute_pr_diff_digest_from_repo(tmp_path, base, head)
    assert printed.startswith("sha256:")
