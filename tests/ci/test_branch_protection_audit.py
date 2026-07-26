# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for scripts/audit_branch_protection_lib.py (OMN-9034, OMN-14696).

Tests validate Check A (reviews, via GraphQL) / Check B (orphans) / fix-payload
logic by importing the lib directly and injecting a fake `gh` callable that
returns canned JSON. No subprocess/bash/gh invocations happen at unit-test time
— that's the whole point of the lib extraction (OMN-9034 thread 8).

OMN-14696 extends the audit to the `dev` branch (per-branch attribution) and
adds the load-bearing SAFETY invariant that a non-`main` audit is strictly
READ-ONLY: it never returns a fix-eligible `protection_json`, never reads
default-branch check-runs (Check B), and therefore can never drive a PUT to a
dev branch's protection.
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


def _graphql_payload(rules: list[tuple[str, bool]]) -> str:
    """Build a GraphQL branchProtectionRules response.

    `rules` is a list of `(pattern, requires_approving_reviews)` tuples.
    """
    return json.dumps(
        {
            "data": {
                "repository": {
                    "branchProtectionRules": {
                        "nodes": [
                            {"pattern": p, "requiresApprovingReviews": r}
                            for p, r in rules
                        ]
                    }
                }
            }
        }
    )


def _fake_gh(
    protection_json: str,
    commits: list[str],
    check_runs_by_sha: dict[str, list[str]],
    graphql_json: str | None = None,
) -> tuple[lib.GhCaller, list[list[str]]]:
    """Build a fake `gh` callable and a call-log list for assertion.

    `graphql_json` is returned for the `gh api graphql ...` review-enforcement
    query. When None, the GraphQL call fails (rc=1) to simulate an unreachable
    repo (e.g. no CROSS_REPO_PAT).
    """
    calls: list[list[str]] = []

    def gh(args: list[str]) -> tuple[int, str]:
        calls.append(args)
        # Review-enforcement GraphQL query (checked first — has no URL path).
        if "graphql" in args:
            return (0, graphql_json) if graphql_json is not None else (1, "")
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
# parse_requires_approving_reviews — GraphQL review-enforcement signal (Check A)
# ---------------------------------------------------------------------------


class TestParseRequiresApprovingReviews:
    def test_true_when_enforced(self) -> None:
        payload = _graphql_payload([("main", True)])
        assert lib.parse_requires_approving_reviews(payload, "main") is True

    def test_false_when_not_enforced(self) -> None:
        payload = _graphql_payload([("main", False)])
        assert lib.parse_requires_approving_reviews(payload, "main") is False

    def test_selects_the_requested_branch(self) -> None:
        """The dev rule (False) must be returned for `dev`, not the main rule."""
        payload = _graphql_payload([("main", True), ("dev", False)])
        assert lib.parse_requires_approving_reviews(payload, "dev") is False
        assert lib.parse_requires_approving_reviews(payload, "main") is True

    def test_none_when_no_matching_branch(self) -> None:
        """No rule for `dev` → None (unknown), NOT an inferred enforcement."""
        payload = _graphql_payload([("main", True)])
        assert lib.parse_requires_approving_reviews(payload, "dev") is None

    def test_none_on_malformed_json(self) -> None:
        assert lib.parse_requires_approving_reviews("not json", "main") is None

    def test_none_when_nodes_absent(self) -> None:
        assert lib.parse_requires_approving_reviews("{}", "main") is None

    def test_does_not_use_rest_count(self) -> None:
        """Guard against regression to the phantom REST count: a REST-shaped
        payload (with required_approving_review_count) carries no GraphQL nodes,
        so the parser returns None rather than reading the phantom count.
        """
        rest_like = _protection_payload(rac=1)
        assert lib.parse_requires_approving_reviews(rest_like, "main") is None


class TestFetchRequiresApprovingReviews:
    def test_reachable_and_enforced(self) -> None:
        gh, _ = _fake_gh("", [], {}, graphql_json=_graphql_payload([("main", True)]))
        reachable, enforced = lib.fetch_requires_approving_reviews(
            OWNER, REPO, "main", gh
        )
        assert reachable is True
        assert enforced is True

    def test_unreachable_when_gh_fails(self) -> None:
        gh, _ = _fake_gh("", [], {}, graphql_json=None)  # graphql rc=1
        reachable, enforced = lib.fetch_requires_approving_reviews(
            OWNER, REPO, "dev", gh
        )
        assert reachable is False
        assert enforced is None

    def test_reachable_but_no_rule_for_branch(self) -> None:
        gh, _ = _fake_gh("", [], {}, graphql_json=_graphql_payload([("main", False)]))
        reachable, enforced = lib.fetch_requires_approving_reviews(
            OWNER, REPO, "dev", gh
        )
        assert reachable is True
        assert enforced is None


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

    def test_pr_only_context_not_flagged_when_unseen(self) -> None:
        """`main-target-guard` binds its check-run to the PR head SHA, never to
        a `main` commit, so it can never appear in `seen`. It must NOT be
        flagged as an orphan even with an empty `seen` set (OMN-13517).
        """
        assert lib.find_orphan_contexts(["main-target-guard"], set()) == []

    def test_genuine_stale_context_still_flagged_when_unseen(self) -> None:
        """A genuinely-stale (renamed/removed) CI context like `CI Summary`
        must STILL be flagged when absent from `seen` — the PR-only allowlist
        must not mask real drift (e.g. omninode_infra's stale contexts).
        """
        assert lib.find_orphan_contexts(["CI Summary"], set()) == ["CI Summary"]

    def test_pr_only_excluded_but_other_orphans_kept(self) -> None:
        """In a mixed list, the PR-only context is suppressed while a real
        orphan in the same call is preserved — order-stable.
        """
        orphans = lib.find_orphan_contexts(
            ["main-target-guard", "CI Summary", "ci / build"],
            {"ci / build"},
        )
        assert orphans == ["CI Summary"]

    def test_pr_only_context_present_in_seen_also_not_flagged(self) -> None:
        """Even if `main-target-guard` somehow appears in `seen`, it is not an
        orphan — covers both filter branches.
        """
        assert (
            lib.find_orphan_contexts(["main-target-guard"], {"main-target-guard"}) == []
        )

    def test_pr_only_contexts_constant_contains_main_target_guard(self) -> None:
        """Regression guard: the allowlist must contain `main-target-guard`
        (OMN-12243 / OMN-13517) and be an immutable frozenset.
        """
        assert "main-target-guard" in lib.PR_ONLY_CONTEXTS
        assert isinstance(lib.PR_ONLY_CONTEXTS, frozenset)


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
# audit_repo — MAIN branch, end-to-end (still injected, no real gh)
# ---------------------------------------------------------------------------


class TestAuditRepoMain:
    def test_clean_repo_returns_ok(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(contexts=["ci / build"]),
            ["sha1"],
            {"sha1": ["ci / build", "ci / lint"]},
            graphql_json=_graphql_payload([("main", False)]),
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "ok"
        assert result["branch"] == "main"
        assert result["review_enforced"] is False
        assert result["orphan_contexts"] == []
        # Explicit behavioral assertion — not a returncode-in-(0,1) smoke test
        assert REPO in result["message"]
        assert "main" in result["message"]

    def test_reviews_enforced_is_flagged(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(contexts=[]),
            ["sha1"],
            {"sha1": []},
            graphql_json=_graphql_payload([("main", True)]),
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "violation"
        assert result["review_enforced"] is True
        assert "Check A" in result["message"]
        assert "requiresApprovingReviews=true" in result["message"]

    def test_orphan_context_is_flagged(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(contexts=["ci / old-job"]),
            ["sha1", "sha2"],
            {"sha1": ["ci / new-job"], "sha2": ["ci / lint"]},
            graphql_json=_graphql_payload([("main", False)]),
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "violation"
        assert result["orphan_contexts"] == ["ci / old-job"]
        assert "Check B" in result["message"]
        assert "ci / old-job" in result["message"]

    def test_inaccessible_protection_returns_skip(self) -> None:
        def gh(args: list[str]) -> tuple[int, str]:
            return 1, ""  # every gh call (incl. graphql) fails

        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "skip"
        assert result["protection_json"] == ""
        assert REPO in result["message"]

    def test_both_checks_fail_accumulates_messages(self) -> None:
        gh, _ = _fake_gh(
            _protection_payload(contexts=["ci / missing"]),
            ["sha1"],
            {"sha1": ["ci / present"]},
            graphql_json=_graphql_payload([("main", True)]),
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "violation"
        assert result["review_enforced"] is True
        assert result["orphan_contexts"] == ["ci / missing"]
        assert "Check A" in result["message"]
        assert "Check B" in result["message"]

    def test_main_violation_returns_fix_eligible_payload(self) -> None:
        """On a MAIN violation the raw protection payload IS returned so the
        main-only --fix path can build a PUT body. (Contrast with dev, below.)
        """
        gh, _ = _fake_gh(
            _protection_payload(contexts=[]),
            ["sha1"],
            {"sha1": []},
            graphql_json=_graphql_payload([("main", True)]),
        )
        result = lib.audit_repo(OWNER, REPO, gh)
        assert result["status"] == "violation"
        assert result["protection_json"] != ""
        # And the fix payload is buildable from it.
        payload = lib.build_fix_payload(result["protection_json"])
        assert payload["required_pull_request_reviews"] is None


# ---------------------------------------------------------------------------
# audit_repo — DEV branch: READ-ONLY, GraphQL-only, never fix-eligible (OMN-14696)
# ---------------------------------------------------------------------------


class TestAuditRepoDev:
    def test_dev_reviews_not_enforced_is_ok(self) -> None:
        gh, _ = _fake_gh(
            "", [], {}, graphql_json=_graphql_payload([("main", False), ("dev", False)])
        )
        result = lib.audit_repo(OWNER, REPO, gh, branch="dev")
        assert result["status"] == "ok"
        assert result["branch"] == "dev"
        assert result["review_enforced"] is False
        assert "dev" in result["message"]

    def test_dev_reviews_enforced_is_violation(self) -> None:
        gh, _ = _fake_gh(
            "", [], {}, graphql_json=_graphql_payload([("main", False), ("dev", True)])
        )
        result = lib.audit_repo(OWNER, REPO, gh, branch="dev")
        assert result["status"] == "violation"
        assert result["review_enforced"] is True
        assert "Check A" in result["message"]
        assert "dev" in result["message"]

    def test_dev_graphql_unreachable_returns_skip(self) -> None:
        gh, _ = _fake_gh("", [], {}, graphql_json=None)  # graphql rc=1
        result = lib.audit_repo(OWNER, REPO, gh, branch="dev")
        assert result["status"] == "skip"
        assert "dev" in result["message"]

    def test_dev_no_matching_rule_is_ok_not_violation(self) -> None:
        """A protected repo whose GraphQL has no `dev` rule → unknown → NON-failing
        (never inferred as enforced)."""
        gh, _ = _fake_gh("", [], {}, graphql_json=_graphql_payload([("main", False)]))
        result = lib.audit_repo(OWNER, REPO, gh, branch="dev")
        assert result["status"] == "ok"
        assert result["review_enforced"] is None

    def test_dev_audit_never_returns_fix_eligible_payload(self) -> None:
        """SAFETY INVARIANT: a non-main audit ALWAYS returns an empty
        protection_json — on ok and on violation — so the --fix path can never
        build/PUT a payload for a dev branch's protection.
        """
        for rule, expected_status in [
            (("dev", False), "ok"),
            (("dev", True), "violation"),
        ]:
            gh, _ = _fake_gh("", [], {}, graphql_json=_graphql_payload([rule]))
            result = lib.audit_repo(OWNER, REPO, gh, branch="dev")
            assert result["status"] == expected_status
            assert result["protection_json"] == "", (
                f"dev audit ({expected_status}) leaked a fix-eligible payload"
            )

    def test_dev_audit_never_reads_main_protection_or_check_runs(self) -> None:
        """SAFETY INVARIANT: a dev audit performs ONLY the GraphQL review query.
        It must NOT fetch REST `branches/main/protection` (Check B's default-branch
        read + the fix payload source) nor any `check-runs` — those are main-scoped.
        """
        gh, calls = _fake_gh(
            _protection_payload(contexts=["ci / would-be-orphan"]),
            ["sha1"],
            {"sha1": []},
            graphql_json=_graphql_payload([("dev", False)]),
        )
        result = lib.audit_repo(OWNER, REPO, gh, branch="dev")
        assert result["status"] == "ok"
        assert result["orphan_contexts"] == []  # Check B did not run
        flat = [" ".join(c) for c in calls]
        assert all("graphql" in c for c in calls), (
            f"dev audit issued a non-GraphQL gh call: {flat}"
        )
        assert not any("/branches/main/protection" in c for c in flat)
        assert not any("/check-runs" in c for c in flat)


# ---------------------------------------------------------------------------
# DEV_EXEMPT_REPOS — must match onex_change_control#4281 / OMN-14683
# ---------------------------------------------------------------------------


def test_dev_exempt_repos_matches_sibling_guard() -> None:
    """The dev-exempt set must match onex_change_control#4281's DEV_EXEMPT_REPOS
    so dev-unprotected repos don't false-fail (OMN-14683 / OMN-14696)."""
    assert frozenset({"omnistream", "omniweb"}) == lib.DEV_EXEMPT_REPOS
    assert isinstance(lib.DEV_EXEMPT_REPOS, frozenset)


# ---------------------------------------------------------------------------
# Page size constant — guards against regression to per_page=50 (thread 7)
# ---------------------------------------------------------------------------


def test_page_size_is_github_max() -> None:
    """GitHub REST API caps per_page at 100. Regression guard for thread 7."""
    assert lib.PAGE_SIZE == 100
