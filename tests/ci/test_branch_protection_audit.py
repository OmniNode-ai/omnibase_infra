# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/audit_branch_protection_lib.py (OMN-9034).

Tests validate Check A / Check B / fix-payload logic by importing the lib
directly and injecting a fake `gh` callable that returns canned JSON.
No subprocess/bash/gh invocations happen at unit-test time — that's the
whole point of the lib extraction (OMN-9034 thread 8).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Import the lib from scripts/ (not on sys.path by default)
# ---------------------------------------------------------------------------

_LIB_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "audit_branch_protection_lib.py"
)
_spec = importlib.util.spec_from_file_location("audit_branch_protection_lib", _LIB_PATH)
assert _spec is not None and _spec.loader is not None, f"cannot load {_LIB_PATH}"
lib = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lib)

OWNER = "OmniNode-ai"
REPO = "omnibase_infra"


# ---------------------------------------------------------------------------
# Helpers to build synthetic gh api responses
# ---------------------------------------------------------------------------


def _protection_payload(rac: int = 0, contexts: list[str] | None = None) -> str:
    return json.dumps(
        {
            "required_pull_request_reviews": {"required_approving_review_count": rac},
            "required_status_checks": {
                "strict": True,
                "contexts": contexts or [],
            },
            "enforce_admins": {"enabled": False},
            "restrictions": None,
        }
    )


def _check_runs_payload(names: list[str]) -> str:
    return json.dumps(
        {"check_runs": [{"name": n, "status": "completed"} for n in names]}
    )


def _fake_gh(
    protection_json: str,
    commits: list[str],
    check_runs_by_sha: dict[str, list[str]],
) -> tuple[lib.GhCaller, list[list[str]]]:
    """Build a fake `gh` callable and a call-log list for assertion."""
    calls: list[list[str]] = []

    def gh(args: list[str]) -> tuple[int, str]:
        calls.append(args)
        url = next((a for a in args if "/" in a and not a.startswith("-")), "")
        if url.endswith("/branches/main/protection"):
            return 0, protection_json
        if "/commits?per_page=" in url:
            # Return one sha per line (jq .[].sha format)
            return 0, "\n".join(commits) + ("\n" if commits else "")
        if "/check-runs" in url:
            sha = url.split("/commits/")[1].split("/check-runs")[0]
            names = check_runs_by_sha.get(sha, [])
            return 0, _check_runs_payload(names)
        return 1, ""

    return gh, calls


# ---------------------------------------------------------------------------
# parse_required_approving_review_count
# ---------------------------------------------------------------------------


class TestParseRac:
    def test_zero_when_field_absent(self) -> None:
        assert lib.parse_required_approving_review_count("{}") == 0

    def test_zero_when_value_zero(self) -> None:
        assert (
            lib.parse_required_approving_review_count(_protection_payload(rac=0)) == 0
        )

    def test_extracts_positive_value(self) -> None:
        assert (
            lib.parse_required_approving_review_count(_protection_payload(rac=3)) == 3
        )

    def test_malformed_json_returns_zero(self) -> None:
        assert lib.parse_required_approving_review_count("not json") == 0


# ---------------------------------------------------------------------------
# parse_required_contexts
# ---------------------------------------------------------------------------


class TestParseContexts:
    def test_empty_when_absent(self) -> None:
        assert lib.parse_required_contexts("{}") == []

    def test_extracts_list(self) -> None:
        ctxs = lib.parse_required_contexts(
            _protection_payload(contexts=["ci / build", "ci / lint"])
        )
        assert ctxs == ["ci / build", "ci / lint"]


# ---------------------------------------------------------------------------
# build_fix_payload
# ---------------------------------------------------------------------------


class TestBuildFixPayload:
    def test_preserves_status_checks_drops_reviews(self) -> None:
        original = _protection_payload(rac=3, contexts=["ci / build", "ci / test"])
        payload = lib.build_fix_payload(original)

        assert payload["required_pull_request_reviews"] is None
        assert payload["restrictions"] is None
        assert payload["enforce_admins"] is False
        assert payload["required_status_checks"]["strict"] is True
        assert set(payload["required_status_checks"]["contexts"]) == {
            "ci / build",
            "ci / test",
        }

    def test_handles_missing_status_checks(self) -> None:
        source = json.dumps(
            {
                "required_pull_request_reviews": {"required_approving_review_count": 1},
                "enforce_admins": {"enabled": True},
                "restrictions": None,
            }
        )
        payload = lib.build_fix_payload(source)

        assert "required_status_checks" not in payload
        assert payload["enforce_admins"] is True
        assert payload["required_pull_request_reviews"] is None


# ---------------------------------------------------------------------------
# find_orphan_contexts
# ---------------------------------------------------------------------------


class TestFindOrphans:
    def test_detects_orphan(self) -> None:
        orphans = lib.find_orphan_contexts(
            ["ci / old-job", "ci / build"], {"ci / build", "ci / lint"}
        )
        assert orphans == ["ci / old-job"]

    def test_all_matching_returns_empty(self) -> None:
        orphans = lib.find_orphan_contexts(["ci / build"], {"ci / build", "ci / lint"})
        assert orphans == []

    def test_empty_required_returns_empty(self) -> None:
        assert lib.find_orphan_contexts([], {"ci / build"}) == []


# ---------------------------------------------------------------------------
# collect_seen_check_run_names — exercises pagination (OMN-9034 thread 7)
# ---------------------------------------------------------------------------


class TestCollectSeen:
    def test_paginates_when_page_full(self) -> None:
        # A single commit with 2 pages of 100 check-runs each, then a short 3rd page.
        page1 = [f"ci / job-{i}" for i in range(100)]
        page2 = [f"ci / job-{i}" for i in range(100, 200)]
        page3 = [f"ci / job-{i}" for i in range(200, 250)]

        call_count = {"n": 0}

        def gh(args: list[str]) -> tuple[int, str]:
            call_count["n"] += 1
            url = next((a for a in args if "/" in a and not a.startswith("-")), "")
            if "/commits?per_page=" in url:
                return 0, "sha-one\n"
            if "/check-runs" in url:
                if url.endswith("&page=1"):
                    return 0, _check_runs_payload(page1)
                if url.endswith("&page=2"):
                    return 0, _check_runs_payload(page2)
                if url.endswith("&page=3"):
                    return 0, _check_runs_payload(page3)
                return 0, _check_runs_payload([])
            return 1, ""

        seen = lib.collect_seen_check_run_names(OWNER, REPO, 5, gh)

        assert len(seen) == 250
        assert "ci / job-0" in seen
        assert "ci / job-249" in seen

    def test_stops_on_empty_page_after_full(self) -> None:
        """Page 1 returns exactly PAGE_SIZE runs → must fetch page 2.
        Page 2 returns empty → loop must terminate. Regression guard for
        pagination that would otherwise loop forever on a busy repo.
        """
        page1 = [f"ci / j{i}" for i in range(100)]

        def gh(args: list[str]) -> tuple[int, str]:
            url = next((a for a in args if "/" in a and not a.startswith("-")), "")
            if "/commits?per_page=" in url:
                return 0, "sha-one\n"
            if "/check-runs" in url and url.endswith("&page=1"):
                return 0, _check_runs_payload(page1)
            if "/check-runs" in url and url.endswith("&page=2"):
                return 0, _check_runs_payload([])
            return 1, ""

        seen = lib.collect_seen_check_run_names(OWNER, REPO, 5, gh)
        assert len(seen) == 100
        assert "ci / j0" in seen


# ---------------------------------------------------------------------------
# audit_repo — end-to-end (still injected, no real gh)
# ---------------------------------------------------------------------------


class TestAuditRepo:
    def test_clean_repo_returns_ok(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(rac=0, contexts=["ci / build"]),
            ["sha1"],
            {"sha1": ["ci / build", "ci / lint"]},
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "ok"
        assert result["rac"] == 0
        assert result["orphan_contexts"] == []
        # Explicit behavioral assertion — not returncode-in-(0,1) smoke test
        assert REPO in result["message"]

    def test_rac_violation_is_flagged(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(rac=1, contexts=[]),
            ["sha1"],
            {"sha1": []},
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "violation"
        assert result["rac"] == 1
        assert "Check A" in result["message"]
        assert "must be 0" in result["message"]

    def test_orphan_context_is_flagged(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(rac=0, contexts=["ci / old-job"]),
            ["sha1", "sha2"],
            {"sha1": ["ci / new-job"], "sha2": ["ci / lint"]},
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "violation"
        assert result["orphan_contexts"] == ["ci / old-job"]
        assert "Check B" in result["message"]
        assert "ci / old-job" in result["message"]

    def test_inaccessible_protection_returns_skip(self) -> None:
        def gh(args: list[str]) -> tuple[int, str]:
            return 1, ""  # gh api call fails

        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "skip"
        assert REPO in result["message"]

    def test_both_checks_fail_accumulates_messages(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(rac=2, contexts=["ci / missing"]),
            ["sha1"],
            {"sha1": ["ci / present"]},
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "violation"
        assert result["rac"] == 2
        assert result["orphan_contexts"] == ["ci / missing"]
        assert "Check A" in result["message"]
        assert "Check B" in result["message"]


# ---------------------------------------------------------------------------
# Page size constant — guards against regression to per_page=50 (thread 7)
# ---------------------------------------------------------------------------


def test_page_size_is_github_max() -> None:
    """GitHub REST API caps per_page at 100. Regression guard for thread 7."""
    assert lib.PAGE_SIZE == 100
