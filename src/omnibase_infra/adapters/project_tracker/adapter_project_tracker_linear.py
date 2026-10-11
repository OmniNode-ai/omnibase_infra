# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration target for omnibase_compat.adapters.adapter_project_tracker_linear.

Provides list_teams, list_issue_labels, and list_issue_statuses via
the shared AdapterLinearGraphQLProjectTracker transport.

Migrated from omnibase_compat (compat removal date: 2026-09-01, OMN-12193).
The compat version used synchronous urllib; this version uses async httpx and
the standard infra error hierarchy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, cast

from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_issue_status import (
    ModelProjectTrackerIssueStatus,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_label import (
    ModelProjectTrackerLabel,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_team import (
    ModelProjectTrackerTeam,
)

JsonObject = dict[str, object]

_QUERY_LIST_TEAMS: Final[str] = """
query {
    teams { nodes { id name key } }
}
"""

_QUERY_LIST_LABELS: Final[str] = """
query ($filter: IssueLabelFilter!) {
    issueLabels(filter: $filter) {
        nodes { id name color team { id } }
    }
}
"""

_QUERY_LIST_STATUSES: Final[str] = """
query ($filter: WorkflowStateFilter!) {
    workflowStates(filter: $filter) {
        nodes { id name type team { id } }
    }
}
"""


def _nested_id(raw: Mapping[str, object], key: str) -> str | None:
    val = raw.get(key)
    if isinstance(val, Mapping):
        inner = val.get("id")
        return inner if isinstance(inner, str) else None
    return None


def _required_str(raw: Mapping[str, object], key: str) -> str:
    val = raw.get(key)
    if isinstance(val, str):
        return val
    raise ValueError(f"Missing string field '{key}' in Linear response node: {raw}")


def _optional_str(raw: Mapping[str, object], key: str) -> str | None:
    val = raw.get(key)
    if val is None or isinstance(val, str):
        return val
    raise ValueError(f"Expected optional string field '{key}' in Linear response node")


def _extract_nodes(data: Mapping[str, object], root_key: str) -> list[JsonObject]:
    # _execute already unwraps the top-level "data" envelope; receive it directly.
    root = data.get(root_key)
    if not isinstance(root, Mapping):
        raise ValueError(f"Missing '{root_key}' in Linear response: {data}")
    nodes = root.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError(f"Missing 'nodes' under '{root_key}': {root}")
    return [cast("JsonObject", n) for n in nodes if isinstance(n, dict)]


class AdapterProjectTrackerLinear(AdapterLinearGraphQLProjectTracker):
    """Async Linear GraphQL adapter for team/label/status discovery.

    Uses the shared GraphQL transport, error mapping and lifecycle.
    Exposes list_teams, list_issue_labels, and list_issue_statuses.

    Auth via constructor api_key arg or LINEAR_API_KEY / LINEAR_TOKEN env vars.
    Resilience via MixinAsyncCircuitBreaker (threshold=5, reset=60s).
    """

    async def list_teams(self) -> list[ModelProjectTrackerTeam]:
        data = await self._execute(_QUERY_LIST_TEAMS, operation="list_teams")
        nodes = _extract_nodes(data, "teams")
        return [
            ModelProjectTrackerTeam(
                id=_required_str(n, "id"),
                name=_required_str(n, "name"),
                key=_required_str(n, "key"),
            )
            for n in nodes
        ]

    async def list_issue_labels(self, team: str) -> list[ModelProjectTrackerLabel]:
        data = await self._execute(
            _QUERY_LIST_LABELS,
            operation="list_issue_labels",
            variables={"filter": {"team": {"key": {"eq": team}}}},
        )
        nodes = _extract_nodes(data, "issueLabels")
        return [
            ModelProjectTrackerLabel(
                id=_required_str(n, "id"),
                name=_required_str(n, "name"),
                color=_optional_str(n, "color"),
                team_id=_nested_id(n, "team"),
            )
            for n in nodes
        ]

    async def list_issue_statuses(
        self, team: str
    ) -> list[ModelProjectTrackerIssueStatus]:
        data = await self._execute(
            _QUERY_LIST_STATUSES,
            operation="list_issue_statuses",
            variables={"filter": {"team": {"key": {"eq": team}}}},
        )
        nodes = _extract_nodes(data, "workflowStates")
        return [
            ModelProjectTrackerIssueStatus(
                id=_required_str(n, "id"),
                name=_required_str(n, "name"),
                type=_required_str(n, "type"),
                team_id=_nested_id(n, "team"),
            )
            for n in nodes
        ]


__all__: list[str] = [
    "AdapterProjectTrackerLinear",
]
