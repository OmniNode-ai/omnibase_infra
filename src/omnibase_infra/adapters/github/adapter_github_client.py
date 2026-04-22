# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Canonical GitHub HTTP client for all OmniNode nodes.

Single adapter in omnibase_infra. Every node that needs GitHub API access
imports this — no node should shell out to ``gh`` or build its own urllib
client.

Reads ``GH_PAT`` from environment (fail-fast, no fallback).

Capabilities:
  - GraphQL queries (paginated)
  - REST GET / POST / PUT / DELETE
  - Fetch open PRs with rich status fields
  - Fetch branch protection rules
  - Resolve PR GraphQL node IDs
  - Resolve PR refs (head/base)
  - Fetch failing CI run IDs and job names
  - Fetch open review thread comment IDs
  - Fetch conflict files
  - Enable auto-merge (GraphQL mutation)
  - Rerun CI checks
  - Post review thread replies

OMN-MERGE-SWEEP.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

_log = logging.getLogger(__name__)

_GITHUB_GRAPHQL = "https://api.github.com/graphql"
_GITHUB_REST = "https://api.github.com"
_DEFAULT_TIMEOUT = 30


class GitHubTransport:
    """Low-level HTTP transport for GitHub API (private)."""

    def __init__(self, token: str | None = None) -> None:
        self._token = token or os.environ.get("GH_PAT", "")
        if not self._token:
            raise RuntimeError(
                "GH_PAT environment variable is not set. "
                "Export it before using GitHubHttpClient."
            )

    def _graphql(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        """Execute a GraphQL query. Returns the ``data`` dict. Never raises."""
        payload = json.dumps({"query": query, "variables": variables}).encode()
        req = urllib.request.Request(  # noqa: S310
            _GITHUB_GRAPHQL,
            data=payload,
            headers={
                "Authorization": f"bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:  # noqa: S310
                body = json.loads(resp.read())
            if "errors" in body:
                _log.warning("GraphQL errors: %s", body["errors"])
                return {}
            data = body.get("data")
            return data if isinstance(data, dict) else {}
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            _log.warning("GraphQL request failed: %s", exc)
            return {}

    def _rest_get(
        self, path: str, *, timeout: int = _DEFAULT_TIMEOUT
    ) -> dict[str, object] | None:
        """REST GET. Returns parsed JSON or None. Never raises."""
        return self._rest_request("GET", path, timeout=timeout)

    def _rest_post(
        self,
        path: str,
        body: dict[str, object] | None = None,
        *,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> dict[str, object] | None:
        """REST POST. Returns parsed JSON or None. Never raises."""
        return self._rest_request("POST", path, json_body=body, timeout=timeout)

    def _rest_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, object] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> dict[str, object] | None:
        """Generic REST request. Returns parsed JSON or None. Never raises."""
        url = f"{_GITHUB_REST}{path}"
        data = json.dumps(json_body).encode() if json_body else None
        req = urllib.request.Request(  # noqa: S310
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                raw = resp.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            _log.warning("REST %s %s error: %s", method, path, exc)
            return None
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
            _log.warning("REST %s %s failed: %s", method, path, exc)
            return None


class GitHubPrApi(GitHubTransport):
    """PR query and CI check methods (private). Inherits GitHubTransport."""

    _PR_GRAPHQL_QUERY = """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(states: [OPEN], first: 100, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title isDraft mergeable mergeStateStatus reviewDecision
        headRefName baseRefName headRefOid
        labels(first: 20) { nodes { name } }
      }
    }
  }
}
"""

    def fetch_open_prs(self, repo: str) -> list[dict[str, object]]:
        """Fetch all open PRs with rich status fields via GraphQL."""
        owner, name = _split_repo(repo)
        all_prs: list[dict[str, object]] = []
        cursor: str | None = None

        while True:
            variables: dict[str, object] = {"owner": owner, "name": name}
            if cursor:
                variables["after"] = cursor

            data = self._graphql(self._PR_GRAPHQL_QUERY, variables)
            repo_data = data.get("repository")
            if not isinstance(repo_data, dict):
                break

            pr_conn = repo_data.get("pullRequests") or {}
            if not isinstance(pr_conn, dict):
                break
            nodes = pr_conn.get("nodes") or []
            if not isinstance(nodes, list):
                break
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                labels_raw = node.get("labels")
                label_nodes = (labels_raw if isinstance(labels_raw, dict) else {}).get(
                    "nodes"
                ) or []
                node["labels"] = [
                    {"name": ln["name"]} for ln in label_nodes if isinstance(ln, dict)
                ]
                node["statusCheckRollup"] = self._fetch_pr_checks_rest(
                    repo, int(node["number"])
                )
                all_prs.append(node)

            page_info = pr_conn.get("pageInfo") or {}
            if not isinstance(page_info, dict) or not page_info.get("hasNextPage"):
                break
            cursor_val = page_info.get("endCursor")
            cursor = str(cursor_val) if cursor_val else None

        return all_prs

    def _fetch_pr_checks_rest(
        self, repo: str, pr_number: int
    ) -> list[dict[str, object]]:
        """Fetch CI check conclusions for a PR via REST API."""
        results: list[dict[str, object]] = []
        detail = self.fetch_pr_detail(repo, pr_number, "headRefOid")
        if detail:
            head_oid = detail.get("headRefOid", "")
            if head_oid:
                data = self._rest_get(
                    f"/repos/{repo}/commits/{head_oid}/check-runs", timeout=15
                )
                if isinstance(data, dict):
                    check_runs = data.get("check_runs") or []
                    if isinstance(check_runs, list):
                        for run in check_runs:
                            if not isinstance(run, dict):
                                continue
                            conclusion = str(run.get("conclusion") or "").upper()
                            results.append(
                                {
                                    "name": str(run.get("name") or ""),
                                    "conclusion": conclusion,
                                    "status": str(run.get("status") or ""),
                                    "isRequired": True,
                                }
                            )
        return results

    def fetch_pr_detail(
        self, repo: str, pr_number: int, fields: str = "id,headRefName"
    ) -> dict[str, object] | None:
        """Fetch specific fields for a single PR via GraphQL."""
        query = (
            """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) { """
            + fields
            + """ }
  }
}
"""
        )
        owner, name = _split_repo(repo)
        data = self._graphql(query, {"owner": owner, "name": name, "number": pr_number})
        repo_data = data.get("repository")
        if not isinstance(repo_data, dict):
            return None
        pr = repo_data.get("pullRequest")
        return pr if isinstance(pr, dict) else None

    def resolve_pr_graphql_id(
        self, repo: str, pr_number: int
    ) -> tuple[str | None, str | None]:
        """Resolve (node_id, head_ref_name) for a PR. Returns (None, None) on failure."""
        detail = self.fetch_pr_detail(repo, pr_number, "id,headRefName")
        if not detail:
            return None, None
        node_id = detail.get("id")
        head_ref = detail.get("headRefName")
        return (str(node_id) if node_id else None), (
            str(head_ref) if head_ref else None
        )

    def resolve_pr_refs(self, repo: str, pr_number: int) -> tuple[str, str, str] | None:
        """Resolve (head_ref, base_ref, head_oid) for a PR. Returns None on failure."""
        detail = self.fetch_pr_detail(
            repo, pr_number, "headRefName,baseRefName,headRefOid"
        )
        if not detail:
            return None
        head_raw = detail.get("headRefName")
        base_raw = detail.get("baseRefName")
        oid_raw = detail.get("headRefOid")
        head = str(head_raw) if head_raw else ""
        base = str(base_raw) if base_raw else ""
        oid = str(oid_raw) if oid_raw else ""
        if not head or not base or not oid:
            _log.error("Missing ref fields for %s#%d: %r", repo, pr_number, detail)
            return None
        return head, base, oid

    def fetch_failing_run_id(self, repo: str, pr_number: int) -> str | None:
        """Find the most recent failing GitHub Actions run ID for a PR."""
        checks = self._fetch_pr_checks_rest(repo, pr_number)
        for ctx in checks:
            if ctx.get("conclusion") == "FAILURE":
                detail = self.fetch_pr_detail(repo, pr_number, "headRefOid")
                if detail:
                    head_oid = detail.get("headRefOid", "")
                    if head_oid:
                        data = self._rest_get(
                            f"/repos/{repo}/commits/{head_oid}/check-runs"
                        )
                        if isinstance(data, dict):
                            check_runs = data.get("check_runs") or []
                            if isinstance(check_runs, list):
                                for run in check_runs:
                                    if not isinstance(run, dict):
                                        continue
                                    if (
                                        str(run.get("conclusion") or "").upper()
                                        == "FAILURE"
                                    ):
                                        url = str(
                                            run.get("details_url")
                                            or run.get("html_url")
                                            or ""
                                        )
                                        run_id = (
                                            url.rstrip("/").split("/")[-1]
                                            if url
                                            else None
                                        )
                                        if run_id:
                                            return run_id
        return None

    def fetch_failing_job_name(self, repo: str, pr_number: int) -> str | None:
        """Find the name of the first failing CI job for a PR."""
        checks = self._fetch_pr_checks_rest(repo, pr_number)
        for ctx in checks:
            if ctx.get("conclusion") == "FAILURE":
                name = ctx.get("name")
                return str(name) if name is not None else None
        return None

    def fetch_open_thread_comment_ids(self, repo: str, pr_number: int) -> list[str]:
        """Resolve open review thread comment node IDs for a PR."""
        query = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50) {
        nodes {
          isResolved
          comments(first: 1) { nodes { id } }
        }
      }
    }
  }
}
"""
        owner, name = _split_repo(repo)
        data = self._graphql(query, {"owner": owner, "name": name, "number": pr_number})
        repo_data = data.get("repository")
        pr_data = repo_data.get("pullRequest") if isinstance(repo_data, dict) else None
        pr_dict = pr_data if isinstance(pr_data, dict) else {}
        threads_raw = pr_dict.get("reviewThreads")
        threads_raw = threads_raw if isinstance(threads_raw, dict) else {}
        threads_list = threads_raw.get("nodes") or []
        threads: list[dict[str, object]] = [
            t for t in threads_list if isinstance(t, dict)
        ]
        ids: list[str] = []
        for thread in threads:
            if thread.get("isResolved"):
                continue
            comments_raw = thread.get("comments")
            comments_dict = comments_raw if isinstance(comments_raw, dict) else {}
            comments_nodes = comments_dict.get("nodes") or []
            for comment in comments_nodes:
                if not isinstance(comment, dict):
                    continue
                node_id = comment.get("id")
                if node_id:
                    ids.append(str(node_id))
                    break
        return ids


class GitHubHttpClient(GitHubPrApi):
    """Single GitHub HTTP client. Reads GH_PAT (fail-fast).

    Usage::

        from omnibase_infra.adapters.github.adapter_github_client import GitHubHttpClient

        client = GitHubHttpClient()
        prs = client.fetch_open_prs("OmniNode-ai/omnimarket")
        protection = client.fetch_branch_protection("OmniNode-ai/omnimarket")
    """

    def fetch_pr_files(self, repo: str, pr_number: int) -> list[str]:
        """Fetch file paths changed in a PR."""
        path = f"/repos/{repo}/pulls/{pr_number}/files"
        data = self._rest_get(path)
        if not isinstance(data, list):
            return []
        return [f["path"] for f in data if isinstance(f, dict) and f.get("path")]

    def fetch_branch_protection(self, repo: str, branch: str = "main") -> int | None:
        """Fetch required_approving_review_count for a branch. None = no protection."""
        data = self._rest_get(f"/repos/{repo}/branches/{branch}/protection")
        if data is None:
            return None
        reviews = data.get("required_pull_request_reviews")
        if not isinstance(reviews, dict):
            return None
        raw = reviews.get("required_approving_review_count")
        return raw if isinstance(raw, int) else None

    def enable_auto_merge(self, pr_node_id: str, merge_method: str = "SQUASH") -> bool:
        """Enable auto-merge on a PR via GraphQL mutation. Returns True on success."""
        mutation = """
mutation($pullRequestId: ID!, $mergeMethod: PullRequestMergeMethod) {
  enablePullRequestAutoMerge(input: {pullRequestId: $pullRequestId, mergeMethod: $mergeMethod}) {
    pullRequest { autoMergeRequest { enabledAt } }
  }
}
"""
        data = self._graphql(
            mutation, {"pullRequestId": pr_node_id, "mergeMethod": merge_method}
        )
        return "enablePullRequestAutoMerge" in data

    def rerun_check_suite(self, repo: str, check_suite_id: str) -> bool:
        """Rerun a check suite via REST API. Returns True on success."""
        result = self._rest_post(
            f"/repos/{repo}/check-suites/{check_suite_id}/rerequest"
        )
        return result is not None

    def post_review_thread_reply(self, thread_comment_id: str, body: str) -> bool:
        """Reply to a review thread comment via GraphQL mutation."""
        mutation = """
mutation($pullRequestReviewThreadId: ID!, $body: String!) {
  addPullRequestReviewThreadReply(input: {pullRequestReviewThreadId: $pullRequestReviewThreadId, body: $body}) {
    comment { id }
  }
}
"""
        data = self._graphql(
            mutation, {"pullRequestReviewThreadId": thread_comment_id, "body": body}
        )
        return "addPullRequestReviewThreadReply" in data


def _split_repo(repo: str) -> tuple[str, str]:
    """Split 'OmniNode-ai/omnimarket' into ('OmniNode-ai', 'omnimarket')."""
    parts = repo.split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid repo format: {repo!r} — expected 'org/name'")
    return parts[0], parts[1]


__all__: list[str] = ["GitHubHttpClient"]
