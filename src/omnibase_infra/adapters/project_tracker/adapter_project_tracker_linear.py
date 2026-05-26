# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration target for omnibase_compat.adapters.adapter_project_tracker_linear.

Provides list_teams, list_issue_labels, and list_issue_statuses via
the Linear GraphQL API using httpx.AsyncClient + MixinAsyncCircuitBreaker.

Migrated from omnibase_compat (compat removal date: 2026-09-01, OMN-12193).
The compat version used synchronous urllib; this version uses async httpx and
the standard infra error hierarchy.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final, cast

import httpx

from omnibase_infra.adapters.project_tracker.model_project_tracker_issue_status import (
    ModelProjectTrackerIssueStatus,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_label import (
    ModelProjectTrackerLabel,
)
from omnibase_infra.adapters.project_tracker.model_project_tracker_team import (
    ModelProjectTrackerTeam,
)
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import (
    InfraAuthenticationError,
    InfraConnectionError,
    InfraTimeoutError,
    ModelInfraErrorContext,
    ModelTimeoutErrorContext,
)
from omnibase_infra.mixins import MixinAsyncCircuitBreaker
from omnibase_infra.utils.util_error_sanitization import sanitize_error_string

DEFAULT_LINEAR_GRAPHQL_ENDPOINT: Final[str] = "https://api.linear.app/graphql"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
SERVICE_NAME: Final[str] = "linear-project-tracker"
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


class AdapterProjectTrackerLinear(MixinAsyncCircuitBreaker):
    """Async Linear GraphQL adapter for team/label/status discovery.

    Migration target for omnibase_compat.adapters.adapter_project_tracker_linear.
    Exposes list_teams, list_issue_labels, and list_issue_statuses.

    Auth via constructor api_key arg or LINEAR_API_KEY / LINEAR_TOKEN env vars.
    Resilience via MixinAsyncCircuitBreaker (threshold=5, reset=60s).
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = DEFAULT_LINEAR_GRAPHQL_ENDPOINT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("LINEAR_API_KEY")
            or os.environ.get("LINEAR_TOKEN")
        )
        if resolved_key is not None:
            resolved_key = resolved_key.strip()
        if not resolved_key:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation="construct",
                target_name=SERVICE_NAME,
            )
            raise InfraAuthenticationError(
                "AdapterProjectTrackerLinear requires LINEAR_API_KEY or LINEAR_TOKEN",
                context=context,
                auth_method="api_key",
            )

        self._api_key: str = resolved_key
        self._endpoint: str = endpoint
        self._timeout_seconds: float = timeout_seconds
        self._request_headers: dict[str, str] = {
            "Authorization": resolved_key,
            "Content-Type": "application/json",
        }
        self._owns_client: bool = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers=self._request_headers,
        )

        self._init_circuit_breaker(
            threshold=5,
            reset_timeout=60.0,
            service_name=SERVICE_NAME,
            transport_type=EnumInfraTransportType.HTTP,
            half_open_successes=1,
        )

    async def close(self) -> None:
        await self.cancel_active_recovery()
        if self._owns_client:
            await self._client.aclose()

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

    async def _execute(
        self,
        query: str,
        operation: str,
        variables: dict[str, object] | None = None,
    ) -> JsonObject:
        async with self._circuit_breaker_lock:
            await self._check_circuit_breaker(operation=operation)

        payload: dict[str, object] = {"query": query}
        if variables is not None:
            payload["variables"] = variables

        try:
            response = await self._client.post(
                self._endpoint,
                json=payload,
                headers=self._request_headers,
                timeout=self._timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            timeout_ctx = ModelTimeoutErrorContext(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
                timeout_seconds=self._timeout_seconds,
            )
            raise InfraTimeoutError(
                f"Linear GraphQL request timed out: {operation}",
                context=timeout_ctx,
            ) from exc
        except httpx.HTTPError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                f"Linear GraphQL transport error: {operation}",
                context=context,
            ) from exc

        status = response.status_code
        if status in (401, 403):
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraAuthenticationError(
                f"Linear API rejected credential (HTTP {status})",
                context=context,
                auth_method="api_key",
            )
        if status >= 500:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                f"Linear GraphQL HTTP {status}",
                context=context,
            )
        if status >= 400:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                f"Linear GraphQL HTTP {status}",
                context=context,
            )

        try:
            body = response.json()
        except ValueError as exc:
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                "Linear GraphQL response was not valid JSON",
                context=context,
            ) from exc

        if not isinstance(body, dict):
            async with self._circuit_breaker_lock:
                await self._record_circuit_failure(operation=operation)
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                "Linear GraphQL response was not a JSON object",
                context=context,
            )

        errors = body.get("errors")
        if errors:
            first_msg = "unknown"
            if isinstance(errors, list) and errors and isinstance(errors[0], dict):
                first_msg = sanitize_error_string(
                    str(errors[0].get("message", "unknown"))
                )
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.HTTP,
                operation=operation,
                target_name=SERVICE_NAME,
            )
            raise InfraConnectionError(
                f"Linear GraphQL error: {first_msg}",
                context=context,
            )

        async with self._circuit_breaker_lock:
            await self._reset_circuit_breaker()

        data = body.get("data")
        if not isinstance(data, dict):
            return {}
        return cast("JsonObject", data)


__all__: list[str] = [
    "AdapterProjectTrackerLinear",
]
