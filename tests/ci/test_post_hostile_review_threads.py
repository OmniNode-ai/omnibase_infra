# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/ci/post_hostile_review_threads.py (OMN-17492).

Covers the pure assembly logic: finding flattening, the nit-suppression
noise policy, dedupe fingerprints, unified-diff RIGHT-side line parsing,
anchor resolution, and single-review payload assembly (posture, cap,
truncation banner, marker/fingerprint embedding). Network calls are not
exercised here — the script's HTTP layer is a thin stdlib wrapper and the
posting path is proven live by the OMN-17492 canary run.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "post_hostile_review_threads.py"
)
_spec = importlib.util.spec_from_file_location("post_hostile_review_threads", _SCRIPT)
assert _spec is not None and _spec.loader is not None
poster = importlib.util.module_from_spec(_spec)
sys.modules["post_hostile_review_threads"] = poster
_spec.loader.exec_module(poster)


def _finding(
    severity: str = "warning",
    file_path: str = "src/foo.py:12",
    model: str = "glm-review",
    rule_id: str = "ai-reviewer:glm-review:correctness",
    message: str = "Hand-rolled retry loop swallows the terminal error",
) -> dict[str, Any]:
    return {
        "severity": severity,
        "file_path": file_path,
        "model": model,
        "rule_id": rule_id,
        "normalized_message": message,
        "raw_message": message,
    }


class TestCollectFindings:
    def test_flattens_successful_models_and_attaches_model(self) -> None:
        result = {
            "results": [
                {
                    "model": "glm-review",
                    "success": True,
                    "findings": [{"severity": "error", "file_path": "a.py"}],
                },
                {
                    "model": "qwen3-review",
                    "success": False,
                    "findings": [{"severity": "error", "file_path": "b.py"}],
                },
            ]
        }
        findings = poster.collect_findings(result)
        assert len(findings) == 1
        assert findings[0]["model"] == "glm-review"

    def test_empty_result(self) -> None:
        assert poster.collect_findings({}) == []


class TestNoisePolicy:
    def test_drops_hints_and_sorts_most_severe_first(self) -> None:
        findings = [
            _finding(severity="hint"),
            _finding(severity="info"),
            _finding(severity="error"),
            _finding(severity="warning"),
        ]
        postable, suppressed = poster.split_by_noise_policy(findings)
        assert suppressed == 1
        assert [f["severity"] for f in postable] == ["error", "warning", "info"]


class TestFingerprint:
    def test_stable_across_line_shifts(self) -> None:
        a = _finding(file_path="src/foo.py:12")
        b = _finding(file_path="src/foo.py:12")
        assert poster.finding_fingerprint(a) == poster.finding_fingerprint(b)

    def test_differs_per_model(self) -> None:
        a = _finding(model="glm-review")
        b = _finding(model="qwen3-review")
        assert poster.finding_fingerprint(a) != poster.finding_fingerprint(b)

    def test_shape_is_12_hex(self) -> None:
        fp = poster.finding_fingerprint(_finding())
        assert len(fp) == 12
        int(fp, 16)


class TestParseRightSideLines:
    def test_added_and_context_lines_counted_removed_skipped(self) -> None:
        patch = (
            "@@ -10,3 +20,4 @@ def f():\n"
            " context\n"
            "-removed\n"
            "+added one\n"
            "+added two\n"
            " tail\n"
        )
        lines = poster.parse_right_side_lines(patch)
        assert lines == {20, 21, 22, 23}

    def test_no_newline_marker_consumes_nothing(self) -> None:
        patch = "@@ -1 +1 @@\n+only\n\\ No newline at end of file\n"
        assert poster.parse_right_side_lines(patch) == {1}

    def test_none_patch(self) -> None:
        assert poster.parse_right_side_lines(None) == set()


class TestResolveAnchor:
    CHANGED = {"src/foo.py": {10, 11, 12}, "src/bar.py": set()}

    def test_line_anchor_when_line_in_diff(self) -> None:
        kind, path, line = poster.resolve_anchor(
            _finding(file_path="src/foo.py:12"), self.CHANGED
        )
        assert (kind, path, line) == ("line", "src/foo.py", 12)

    def test_file_anchor_when_line_not_in_diff(self) -> None:
        kind, path, line = poster.resolve_anchor(
            _finding(file_path="src/foo.py:999"), self.CHANGED
        )
        assert (kind, path, line) == ("file", "src/foo.py", None)

    def test_file_anchor_for_bare_changed_path(self) -> None:
        kind, path, line = poster.resolve_anchor(
            _finding(file_path="src/bar.py"), self.CHANGED
        )
        assert (kind, path, line) == ("file", "src/bar.py", None)

    def test_body_for_unknown_path(self) -> None:
        kind, _path, _line = poster.resolve_anchor(
            _finding(file_path="not/in/diff.py:3"), self.CHANGED
        )
        assert kind == "body"

    def test_body_for_non_path_location(self) -> None:
        kind, _, _ = poster.resolve_anchor(_finding(file_path="plan"), self.CHANGED)
        assert kind == "body"


class TestBuildReview:
    CHANGED = {"src/foo.py": {10, 11, 12}}

    def test_request_changes_posture_on_major(self) -> None:
        payload, stats = poster.build_review(
            [_finding(severity="warning")],
            self.CHANGED,
            set(),
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is not None
        assert payload["event"] == "REQUEST_CHANGES"
        assert stats["posted_threads"] == 1

    def test_comment_posture_on_minor_only(self) -> None:
        payload, _ = poster.build_review(
            [_finding(severity="info")],
            self.CHANGED,
            set(),
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is not None
        assert payload["event"] == "COMMENT"

    def test_dedupe_skips_already_posted_fingerprints(self) -> None:
        finding = _finding()
        existing = {poster.finding_fingerprint(finding)}
        payload, stats = poster.build_review(
            [finding],
            self.CHANGED,
            existing,
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is None
        assert stats["deduped"] == 1

    def test_cap_moves_overflow_to_truncation_banner(self) -> None:
        findings = [_finding(message=f"distinct finding number {i}") for i in range(30)]
        payload, stats = poster.build_review(
            findings,
            self.CHANGED,
            set(),
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
            max_thread_comments=25,
        )
        assert payload is not None
        assert len(payload["comments"]) == 25
        assert stats["truncated"] == 5
        assert "TRUNCATED" in payload["body"]

    def test_marker_and_fp_in_every_comment(self) -> None:
        payload, _ = poster.build_review(
            [_finding()],
            self.CHANGED,
            set(),
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is not None
        body = payload["comments"][0]["body"]
        assert poster.MARKER in body
        assert "fp=" in body

    def test_unanchored_findings_land_in_body(self) -> None:
        payload, stats = poster.build_review(
            [_finding(file_path="plan")],
            self.CHANGED,
            set(),
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is not None
        assert payload["comments"] == []
        assert stats["body_findings"] == 1
        assert "not anchored" in payload["body"]

    def test_nothing_to_post_returns_none(self) -> None:
        payload, _ = poster.build_review(
            [],
            self.CHANGED,
            set(),
            suppressed_hints=3,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is None

    def test_line_anchored_comment_shape(self) -> None:
        payload, _ = poster.build_review(
            [_finding(file_path="src/foo.py:11")],
            self.CHANGED,
            set(),
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is not None
        comment = payload["comments"][0]
        assert comment["path"] == "src/foo.py"
        assert comment["line"] == 11
        assert comment["side"] == "RIGHT"
        assert "subject_type" not in comment

    def test_file_anchored_comment_shape(self) -> None:
        payload, _ = poster.build_review(
            [_finding(file_path="src/foo.py:999")],
            self.CHANGED,
            set(),
            suppressed_hints=0,
            models_succeeded=["glm-review"],
            models_failed=[],
        )
        assert payload is not None
        comment = payload["comments"][0]
        assert comment["subject_type"] == "file"
        assert "line" not in comment


class TestQuorumSplit:
    """OMN-18479: only findings more than one model raised get threads.

    The thread gate blocks while hostile-reviewer threads are unresolved,
    so a thread IS the blocking surface on this repo. A finding raised by
    one model of three must therefore be reported without one.
    """

    @staticmethod
    def _result_with_quorum(
        *, agreement_count: int, finding_id: str = "f-1", threshold: int = 2
    ) -> dict[str, Any]:
        return {
            "quorum": {
                "verdict": "passed",
                "quorum_threshold": threshold,
                "blocking_findings": [],
                "warning_findings": [
                    {
                        "agreement_count": agreement_count,
                        "finding_ids": [finding_id],
                    }
                ],
            }
        }

    def test_single_model_finding_is_held_below_quorum(self) -> None:
        finding = _finding()
        finding["finding_id"] = "f-1"
        agreed, below = poster.split_by_quorum(
            self._result_with_quorum(agreement_count=1), [finding]
        )
        assert agreed == []
        assert len(below) == 1

    def test_agreed_finding_is_postable(self) -> None:
        """Positive control for the zero above, same fixture, two models."""
        finding = _finding()
        finding["finding_id"] = "f-1"
        agreed, below = poster.split_by_quorum(
            self._result_with_quorum(agreement_count=2), [finding]
        )
        assert len(agreed) == 1
        assert below == []

    def test_absent_quorum_block_posts_everything(self) -> None:
        """An older reviewer with no quorum block must not silently drop findings."""
        finding = _finding()
        finding["finding_id"] = "f-1"
        agreed, below = poster.split_by_quorum({}, [finding])
        assert len(agreed) == 1
        assert below == []

    def test_below_quorum_findings_never_request_changes(self) -> None:
        agreed_finding = _finding(severity="error", file_path="src/foo.py:12")
        below_finding = _finding(severity="error", file_path="src/bar.py:44")
        payload, stats = poster.build_review(
            [],
            {"src/foo.py": {12}},
            set(),
            suppressed_hints=0,
            models_succeeded=["qwen3-review", "qwen3-review-b"],
            models_failed=[],
            below_quorum=[below_finding],
        )
        assert payload is not None
        assert payload["event"] == "COMMENT"
        assert stats["below_quorum"] == 1
        assert "below quorum" in payload["body"].lower()
        assert below_finding["normalized_message"][:40] in payload["body"]
        # Positive control: the same severity, above quorum, still requests changes.
        payload_agreed, _ = poster.build_review(
            [agreed_finding],
            {"src/foo.py": {12}},
            set(),
            suppressed_hints=0,
            models_succeeded=["qwen3-review", "qwen3-review-b"],
            models_failed=[],
        )
        assert payload_agreed is not None
        assert payload_agreed["event"] == "REQUEST_CHANGES"


class TestPublishOmn17427:
    """File-level findings must not ride in the create-review comments array.

    Live: every reviewed omnibase_infra dev PR merged 2026-10-19..21 logged
    ``review POST failed (422): ... 0.subjectType (Field is not defined on
    DraftPullRequestReviewComment)`` and demoted all findings to the review
    body, so the thread gate never had a thread to read.
    """

    BASE = "https://api.example.invalid/repos/o/r"

    def _payload(self) -> dict[str, Any]:
        return {
            "event": "REQUEST_CHANGES",
            "body": "summary",
            "comments": [
                {"path": "a.py", "body": "line finding", "line": 3, "side": "RIGHT"},
                {"path": "b.py", "body": "file finding", "subject_type": "file"},
            ],
        }

    def _http_error(self) -> Exception:
        import email.message
        import io
        import urllib.error

        return urllib.error.HTTPError(
            "u", 422, "Unprocessable", email.message.Message(), io.BytesIO(b"{}")
        )

    def test_review_carries_line_comments_only_and_file_findings_post_alone(
        self,
    ) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def api(method: str, url: str, token: str, body: dict[str, Any]) -> None:
            calls.append((url, body))

        code = poster.publish(
            self._payload(),
            base=self.BASE,
            pr_number=7,
            token="t",
            head_sha="h" * 40,
            api=api,
        )
        assert code == 0
        review_url, review = calls[0]
        assert review_url.endswith("/pulls/7/reviews")
        assert all("subject_type" not in c for c in review["comments"])
        assert [c["path"] for c in review["comments"]] == ["a.py"]
        file_url, file_comment = calls[1]
        assert file_url.endswith("/pulls/7/comments")
        assert file_comment == {
            "body": "file finding",
            "path": "b.py",
            "subject_type": "file",
            "commit_id": "h" * 40,
        }

    def test_a_rejected_file_comment_is_demoted_not_lost(self) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []

        def api(method: str, url: str, token: str, body: dict[str, Any]) -> None:
            calls.append((url, body))
            if url.endswith("/comments"):
                raise self._http_error()

        code = poster.publish(
            self._payload(),
            base=self.BASE,
            pr_number=7,
            token="t",
            head_sha="h" * 40,
            api=api,
        )
        assert code == 0
        demoted_url, demoted = calls[-1]
        assert demoted_url.endswith("/pulls/7/reviews")
        assert "file finding" in demoted["body"]
