# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the OmniGate GitHub Action verifier."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnibase_infra.gate import action_verify

pytestmark = pytest.mark.unit


def _receipt_body(receipt_json: str = "{}") -> str:
    return (
        "before\n"
        f"{action_verify._RECEIPT_START}\n"
        f"{receipt_json}\n"
        f"{action_verify._RECEIPT_END}\n"
        "after\n"
    )


def _config(
    *,
    signing: str = "sigstore",
    allow_unsigned: bool = False,
    advisory_blocks: bool = False,
    exempt_users: tuple[str, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        gate=SimpleNamespace(exempt_users=exempt_users),
        receipt=SimpleNamespace(
            max_receipt_bytes=4096,
            max_age_minutes=120,
            advisory_blocks=advisory_blocks,
            signing=signing,
            allow_unsigned=allow_unsigned,
            identity=SimpleNamespace(
                expected_issuer="https://token.actions.githubusercontent.com",
                allowed_identities=(
                    "https://github.com/org/repo/.github/workflows/omnigate.yml@refs/heads/main",
                ),
                allowed_identity_regexes=(),
            ),
        ),
    )


def _receipt(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "repository_id": "123",
        "project_url": "https://github.com/org/repo",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "commit_sha": "b" * 40,
        "diff_hash": "sha256:" + "c" * 64,
        "config_hash": "sha256:" + "d" * 64,
        "timestamp": datetime(2026, 5, 17, 12, 0, tzinfo=UTC),
        "checks": (SimpleNamespace(name="lint", status="pass"),),
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _patch_verifier_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: SimpleNamespace | None = None,
    receipt: SimpleNamespace | None = None,
) -> dict[str, object]:
    calls: dict[str, object] = {"cat_file": []}

    def verify_commit(repo_path: Path, sha: str) -> None:
        calls["cat_file"].append((repo_path, sha))

    class Signer:
        def verify(
            self,
            signed_receipt: object,
            *,
            expected_identity: str,
            expected_issuer: str,
        ) -> bool:
            calls["identity"] = expected_identity
            calls["issuer"] = expected_issuer
            calls["signed_receipt"] = signed_receipt
            return True

    monkeypatch.setattr(action_verify, "_verify_commit_object", verify_commit)
    monkeypatch.setattr(
        action_verify,
        "_load_omnigate_config",
        lambda config_path: config or _config(),
    )
    monkeypatch.setattr(
        action_verify,
        "_compute_pr_diff_hash",
        lambda repo_path, *, base_sha, head_sha: "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(
        action_verify,
        "_compute_config_hash",
        lambda config_path: "sha256:" + "d" * 64,
    )
    monkeypatch.setattr(
        action_verify,
        "_model_validate_receipt_json",
        lambda receipt_json: receipt or _receipt(),
    )
    monkeypatch.setattr(action_verify, "_signer", Signer)
    monkeypatch.setattr(
        action_verify,
        "datetime",
        SimpleNamespace(
            now=lambda tz: datetime(2026, 5, 17, 12, 30, tzinfo=UTC),
        ),
    )
    return calls


class TestReceiptExtraction:
    def test_extracts_exactly_one_framed_receipt(self) -> None:
        assert (
            action_verify._extract_receipt_from_pr_body(
                _receipt_body('{"ok": true}'),
                max_bytes=100,
            )
            == '{"ok": true}'
        )

    def test_rejects_missing_duplicate_or_oversized_receipts(self) -> None:
        assert (
            action_verify._extract_receipt_from_pr_body("no receipt", max_bytes=100)
            is None
        )
        assert (
            action_verify._extract_receipt_from_pr_body(
                _receipt_body("{}") + _receipt_body("{}"),
                max_bytes=100,
            )
            is None
        )
        assert (
            action_verify._extract_receipt_from_pr_body(
                _receipt_body('{"large": true}'),
                max_bytes=4,
            )
            is None
        )


class TestVerifyPrReceipt:
    def test_success_verifies_trusted_refs_and_sigstore_policy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls = _patch_verifier_dependencies(monkeypatch)

        decision = action_verify.verify_pr_receipt(
            _receipt_body(),
            tmp_path,
            tmp_path / ".omnigate.yaml",
            repository_id="123",
            repository_url="https://github.com/org/repo",
            base_sha="a" * 40,
            head_sha="b" * 40,
        )

        assert decision["ok"] is True
        assert decision["action"] == "pass"
        assert decision["receipt_diff_hash"] == "sha256:" + "c" * 64
        assert calls["cat_file"] == [
            (tmp_path, "a" * 40),
            (tmp_path, "b" * 40),
        ]
        assert (
            calls["identity"]
            == "https://github.com/org/repo/.github/workflows/omnigate.yml@refs/heads/main"
        )
        assert calls["issuer"] == "https://token.actions.githubusercontent.com"

    def test_missing_receipt_fails_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        _patch_verifier_dependencies(monkeypatch)

        decision = action_verify.verify_pr_receipt(
            "no receipt",
            tmp_path,
            tmp_path / ".omnigate.yaml",
            repository_id="123",
            repository_url="https://github.com/org/repo",
            base_sha="a" * 40,
            head_sha="b" * 40,
        )

        assert decision["ok"] is False
        assert decision["action"] == "fail"
        assert decision["receipt_diff_hash"] is None
        assert "No valid OmniGate receipt" in str(decision["reason"])
        assert set(decision) == {
            "ok",
            "action",
            "reason",
            "receipt_diff_hash",
            "checked_at",
        }

    def test_diff_mismatch_reports_receipt_diff_hash(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        receipt = _receipt(diff_hash="sha256:" + "e" * 64)
        _patch_verifier_dependencies(monkeypatch, receipt=receipt)

        decision = action_verify.verify_pr_receipt(
            _receipt_body(),
            tmp_path,
            tmp_path / ".omnigate.yaml",
            repository_id="123",
            repository_url="https://github.com/org/repo",
            base_sha="a" * 40,
            head_sha="b" * 40,
        )

        assert decision["ok"] is False
        assert decision["action"] == "fail"
        assert decision["receipt_diff_hash"] == "sha256:" + "e" * 64
        assert "Diff hash mismatch" in str(decision["reason"])

    def test_failed_checks_fail_closed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        receipt = _receipt(checks=(SimpleNamespace(name="test", status="fail"),))
        _patch_verifier_dependencies(monkeypatch, receipt=receipt)

        decision = action_verify.verify_pr_receipt(
            _receipt_body(),
            tmp_path,
            tmp_path / ".omnigate.yaml",
            repository_id="123",
            repository_url="https://github.com/org/repo",
            base_sha="a" * 40,
            head_sha="b" * 40,
        )

        assert decision["ok"] is False
        assert decision["action"] == "fail"
        assert decision["reason"] == "Checks failed: test"

    def test_exempt_actor_skips_receipt_verification(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls = _patch_verifier_dependencies(
            monkeypatch,
            config=_config(exempt_users=("dependabot[bot]",)),
        )

        decision = action_verify.verify_pr_receipt(
            "no receipt",
            tmp_path,
            tmp_path / ".omnigate.yaml",
            repository_id="123",
            repository_url="https://github.com/org/repo",
            base_sha="a" * 40,
            head_sha="b" * 40,
            actor="dependabot[bot]",
        )

        assert decision["ok"] is True
        assert decision["action"] == "pass"
        assert decision["receipt_diff_hash"] is None
        assert calls["cat_file"] == [(tmp_path, "a" * 40), (tmp_path, "b" * 40)]


def test_main_writes_decision_and_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    event_path = tmp_path / "event.json"
    decision_path = tmp_path / "decision.json"
    event_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        action_verify,
        "_decision_from_event",
        lambda **kwargs: action_verify._decision(
            ok=False,
            action="fail",
            reason="missing receipt",
        ),
    )

    result = action_verify.main(
        [
            "--event-path",
            str(event_path),
            "--repo-path",
            str(tmp_path),
            "--config",
            str(tmp_path / ".omnigate.yaml"),
            "--decision-out",
            str(decision_path),
        ],
    )

    assert result == 1
    assert json.loads(decision_path.read_text(encoding="utf-8"))["action"] == "fail"


def test_workflow_is_read_only_pull_request_verifier() -> None:
    workflow = Path(".github/workflows/omnigate.yml").read_text(encoding="utf-8")

    assert "pull_request:" in workflow
    assert "pull_request_target" not in workflow
    assert "contents: read" in workflow
    assert "pull-requests: read" in workflow
    assert "github.event.pull_request.head.repo.fork == true" in workflow
    assert "git cat-file -e" in workflow
    assert (
        'git show "${{ github.event.pull_request.base.sha }}:.omnigate.yaml"'
        in workflow
    )
    assert "OMNIGATE_INSTALL_MODE:-repo-local" in workflow
    assert "issues: write" not in workflow
