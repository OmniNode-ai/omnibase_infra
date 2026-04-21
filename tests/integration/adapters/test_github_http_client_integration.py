# SPDX-FileCopyrightText: 2026 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for GitHubHttpClient.

Exercises the client against a local fake HTTPS server to validate all
public methods end-to-end: GraphQL queries, REST GET/POST, pagination,
error handling, and mutations.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import patch

import pytest

from omnibase_infra.adapters.github.adapter_github_client import GitHubHttpClient

# ---------------------------------------------------------------------------
# Fake GitHub server
# ---------------------------------------------------------------------------


class _FakeGitHubHandler(BaseHTTPRequestHandler):
    """Minimal fake that routes GraphQL and REST calls to canned responses."""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def _send_json(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        req = json.loads(raw) if raw else {}
        query: str = req.get("query", "")

        if "pullRequests" in query:
            self._send_json(
                200,
                {
                    "data": {
                        "repository": {
                            "pullRequests": {
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": [
                                    {
                                        "number": 42,
                                        "title": "Test PR",
                                        "isDraft": False,
                                        "mergeable": "MERGEABLE",
                                        "mergeStateStatus": "CLEAN",
                                        "reviewDecision": "APPROVED",
                                        "headRefName": "feature/test",
                                        "baseRefName": "main",
                                        "headRefOid": "abc123",
                                        "labels": {"nodes": [{"name": "ready"}]},
                                    }
                                ],
                            }
                        }
                    }
                },
            )
        elif "enablePullRequestAutoMerge" in query:
            self._send_json(
                200,
                {
                    "data": {
                        "enablePullRequestAutoMerge": {
                            "pullRequest": {"autoMergeRequest": {"enabledAt": "now"}}
                        }
                    }
                },
            )
        elif "addPullRequestReviewThreadReply" in query:
            self._send_json(
                200,
                {
                    "data": {
                        "addPullRequestReviewThreadReply": {"comment": {"id": "cmt-1"}}
                    }
                },
            )
        elif "pullRequest(number" in query and "reviewThreads" in query:
            self._send_json(
                200,
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "nodes": [
                                        {
                                            "isResolved": False,
                                            "comments": {
                                                "nodes": [{"id": "thread-cmt-1"}]
                                            },
                                        },
                                        {
                                            "isResolved": True,
                                            "comments": {
                                                "nodes": [{"id": "thread-cmt-2"}]
                                            },
                                        },
                                    ]
                                }
                            }
                        }
                    }
                },
            )
        elif "pullRequest(number" in query:
            # Generic PR detail query
            self._send_json(
                200,
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "id": "PR_abc123",
                                "headRefName": "feature/test",
                                "baseRefName": "main",
                                "headRefOid": "abc123def456",
                            }
                        }
                    }
                },
            )
        else:
            self._send_json(200, {"data": {}})

    def do_GET(self) -> None:
        if "check-runs" in self.path:
            self._send_json(
                200,
                {
                    "check_runs": [
                        {
                            "name": "CI Tests",
                            "conclusion": "success",
                            "status": "completed",
                            "details_url": "https://github.com/org/repo/actions/runs/99999/jobs/1",
                            "html_url": "https://github.com/org/repo/actions/runs/99999",
                        },
                        {
                            "name": "Lint",
                            "conclusion": "failure",
                            "status": "completed",
                            "details_url": "https://github.com/org/repo/actions/runs/88888/jobs/1",
                        },
                    ]
                },
            )
        elif "branches" in self.path and "protection" in self.path:
            self._send_json(
                200,
                {
                    "required_pull_request_reviews": {
                        "required_approving_review_count": 1
                    }
                },
            )
        elif "/pulls/" in self.path and "/files" in self.path:
            self._send_json(
                200,
                [
                    {"path": "src/foo.py"},
                    {"path": "tests/test_foo.py"},
                ],
            )
        else:
            self._send_json(404, {"message": "Not Found"})


@pytest.fixture(scope="module")
def fake_server() -> Any:
    server = HTTPServer(("127.0.0.1", 0), _FakeGitHubHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield f"http://{host}:{port}"
    server.shutdown()


@pytest.fixture
def client(fake_server: str) -> GitHubHttpClient:
    c = GitHubHttpClient(token="fake-token")
    # Point the client at our fake server by patching the module-level constants
    with (
        patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
            f"{fake_server}/graphql",
        ),
        patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_REST",
            fake_server,
        ),
    ):
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGitHubHttpClientIntegration:
    def test_fetch_open_prs_returns_prs(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with (
            patch(
                "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
                f"{fake_server}/graphql",
            ),
            patch(
                "omnibase_infra.adapters.github.adapter_github_client._GITHUB_REST",
                fake_server,
            ),
        ):
            prs = c.fetch_open_prs("OmniNode-ai/omnimarket")
        assert len(prs) == 1
        assert prs[0]["number"] == 42
        assert prs[0]["title"] == "Test PR"
        assert prs[0]["labels"] == [{"name": "ready"}]

    def test_resolve_pr_graphql_id(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
            f"{fake_server}/graphql",
        ):
            node_id, head_ref = c.resolve_pr_graphql_id("OmniNode-ai/omnimarket", 42)
        assert node_id == "PR_abc123"
        assert head_ref == "feature/test"

    def test_resolve_pr_refs(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
            f"{fake_server}/graphql",
        ):
            refs = c.resolve_pr_refs("OmniNode-ai/omnimarket", 42)
        assert refs is not None
        head, base, oid = refs
        assert head == "feature/test"
        assert base == "main"
        assert oid == "abc123def456"

    def test_fetch_branch_protection(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_REST",
            fake_server,
        ):
            count = c.fetch_branch_protection("OmniNode-ai/omnimarket")
        assert count == 1

    def test_fetch_pr_files(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_REST",
            fake_server,
        ):
            files = c.fetch_pr_files("OmniNode-ai/omnimarket", 42)
        assert "src/foo.py" in files
        assert "tests/test_foo.py" in files

    def test_fetch_open_thread_comment_ids_skips_resolved(
        self, fake_server: str
    ) -> None:
        c = GitHubHttpClient(token="fake-token")
        with patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
            f"{fake_server}/graphql",
        ):
            ids = c.fetch_open_thread_comment_ids("OmniNode-ai/omnimarket", 42)
        assert ids == ["thread-cmt-1"]

    def test_enable_auto_merge_returns_true(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
            f"{fake_server}/graphql",
        ):
            result = c.enable_auto_merge("PR_abc123")
        assert result is True

    def test_post_review_thread_reply(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with patch(
            "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
            f"{fake_server}/graphql",
        ):
            result = c.post_review_thread_reply("thread-cmt-1", "LGTM")
        assert result is True

    def test_fetch_failing_job_name(self, fake_server: str) -> None:
        c = GitHubHttpClient(token="fake-token")
        with (
            patch(
                "omnibase_infra.adapters.github.adapter_github_client._GITHUB_GRAPHQL",
                f"{fake_server}/graphql",
            ),
            patch(
                "omnibase_infra.adapters.github.adapter_github_client._GITHUB_REST",
                fake_server,
            ),
        ):
            name = c.fetch_failing_job_name("OmniNode-ai/omnimarket", 42)
        assert name == "Lint"

    def test_missing_token_raises(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("GH_PAT", None)
            with pytest.raises(RuntimeError, match="GH_PAT"):
                GitHubHttpClient()
