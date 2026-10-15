# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Demotion through the sweep, its Linear client and the shared HTTP adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers.handler_evidence_autoclose_sweep import (
    HandlerEvidenceAutocloseSweep,
    _LinearClient,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
    ModelEvidenceAutocloseSweepRequest,
)
from omnibase_infra.protocols.protocol_linear_graphql_transport import (
    ProtocolLinearGraphQLTransport,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("apply", "confirm_readback", "fail_second_page", "expected"),
    [
        (True, True, False, "demoted"),
        (False, True, False, "would_demote"),
        (True, False, False, "error_readback_unconfirmed"),
        (True, True, True, None),
    ],
    ids=["apply-readback-comment", "dry-run", "readback-refused", "partial-census"],
)
async def test_sweep_deploy_demotion_over_linear_http_transport(
    monkeypatch: pytest.MonkeyPatch,
    apply: bool,
    confirm_readback: bool,
    fail_second_page: bool,
    expected: str | None,
) -> None:
    operations: list[str] = []
    mutations: list[dict[str, str]] = []
    comments: list[str] = []
    state = {"id": "started-id", "name": "In Progress", "type": "started"}
    backlog = {"id": "backlog-id", "name": "Backlog", "type": "backlog"}
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal state
        assert request.headers["Authorization"] == "test-key-omn18489"
        payload = json.loads(request.content)
        query, variables = payload["query"], payload["variables"]
        operation = query.split("(", 1)[0].strip().split()[-1]
        operations.append(operation)
        if operation == "InProgressIssues":
            assert 'name: { eq: "In Progress" }' in query
            assert "createdAt: { gte: $since }" in query
            since = datetime.fromisoformat(variables["since"].replace("Z", "+00:00"))
            age_hours = (datetime.now(UTC) - since).total_seconds() / 3600
            assert 168 <= age_hours < 169
            if variables["after"] is not None:
                assert variables["after"] == "page-2"
                return httpx.Response(
                    200, json={"errors": [{"message": "unreadable page"}]}
                )
            data = {
                "issues": {
                    "pageInfo": {
                        "hasNextPage": fail_second_page,
                        "endCursor": "page-2",
                    },
                    "nodes": [
                        {
                            "id": "issue-id",
                            "identifier": "OMN-9001",
                            "startedAt": old,
                            "state": state,
                            "team": {"id": "team-id"},
                            "comments": {
                                "pageInfo": {"hasNextPage": False},
                                "nodes": [],
                            },
                        }
                    ],
                }
            }
        elif operation == "TeamStates":
            assert variables == {"teamId": "team-id"}
            data = {"team": {"states": {"nodes": [state, backlog]}}}
        elif operation == "UpdateIssueState":
            assert variables == {"issueId": "issue-id", "stateId": "backlog-id"}
            mutations.append(variables)
            if confirm_readback:
                state = backlog
            data = {"issueUpdate": {"success": True}}
        elif operation == "GetIssue":
            assert variables == {"id": "OMN-9001"}
            data = {"issue": {"id": "issue-id", "state": state}}
        elif operation == "CreateComment":
            assert variables["issueId"] == "issue-id"
            comments.append(variables["body"])
            data = {"commentCreate": {"success": True}}
        else:
            raise AssertionError(f"Unexpected GraphQL operation: {operation}")
        return httpx.Response(200, json={"data": data})

    gh_searches: list[list[str]] = []

    async def run_gh(args: list[str], timeout: float) -> tuple[object, str]:
        # Only the external GitHub boundary is replaced; no command is executed.
        if "search/issues" in " ".join(args):
            gh_searches.append(args)
            return {"total_count": 0, "incomplete_results": False, "items": []}, ""
        assert "/pulls?" in " ".join(args)
        return [], ""

    original_transport = AdapterLinearGraphQLProjectTracker.graphql_transport
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:

        @asynccontextmanager
        async def transport(
            api_key: str, timeout_seconds: float
        ) -> AsyncIterator[ProtocolLinearGraphQLTransport]:
            async with original_transport(
                api_key=api_key, timeout_seconds=timeout_seconds, client=http_client
            ) as adapter:
                yield adapter

        monkeypatch.setattr(
            AdapterLinearGraphQLProjectTracker, "graphql_transport", transport
        )
        result = await HandlerEvidenceAutocloseSweep(
            linear_client=_LinearClient(api_key="test-key-omn18489", max_attempts=1),
            run_gh_command=run_gh,
        ).handle(
            ModelEvidenceAutocloseSweepRequest(
                correlation_id=uuid4(),
                apply=apply,
                demotion_window_hours=168,
                linear_retry_max_attempts=1,
            )
        )

    demotion = result.in_progress_demotion
    assert demotion is not None
    if fail_second_page:
        assert demotion.success is False
        assert demotion.outcomes == ()
        assert "enumeration failed" in demotion.error_message
        assert operations == ["InProgressIssues", "InProgressIssues"]
        assert not mutations and not comments and not gh_searches
    else:
        assert [o.decision.value for o in demotion.outcomes] == [expected]
        assert len(gh_searches) == 1
        assert len(mutations) == int(apply)
        assert demotion.tickets_demoted == int(apply and confirm_readback)
        if apply:
            assert operations[:4] == [
                "InProgressIssues",
                "TeamStates",
                "UpdateIssueState",
                "GetIssue",
            ]
        else:
            assert operations == ["InProgressIssues"]
        if apply and confirm_readback:
            assert state == backlog
            assert operations[-1] == "CreateComment"
            assert len(comments) == 1
            assert "168 hours" in comments[0]
            for resolution in (
                "In Progress",
                "Comment",
                "Pull request",
                "Claim",
                "Fence",
            ):
                assert resolution in comments[0]
        else:
            assert comments == []
