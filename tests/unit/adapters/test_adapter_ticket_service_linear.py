# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for AdapterTicketLinear.

Covers:
    - get_ticket (UUID and human-readable identifier)
    - list_tickets (with and without filters)
    - get_ticket_status
    - health_check (success and failure)
    - write methods raise NotImplementedError
    - constructor validation
    - close() lifecycle
    - error handling (HTTP errors, GraphQL errors, not-found)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from omnibase_infra.adapters.ticket.adapter_ticket_service_linear import (
    AdapterTicketLinear,
    _build_issue_filter,
    _normalise_issue,
)
from omnibase_infra.errors import InfraConnectionError, InfraUnavailableError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SAMPLE_ISSUE_RAW: dict = {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "identifier": "OMN-1234",
    "title": "Fix auth middleware",
    "description": "Session tokens non-compliant",
    "url": "https://linear.app/omninode/issue/OMN-1234",
    "priority": 2,
    "state": {"name": "In Progress"},
    "assignee": {"name": "Jonah", "email": "jonah@example.com"},
    "labels": {"nodes": [{"name": "bug"}, {"name": "security"}]},
    "createdAt": "2026-04-01T00:00:00Z",
    "updatedAt": "2026-04-07T00:00:00Z",
}

_EXPECTED_NORMALISED: dict = {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "identifier": "OMN-1234",
    "title": "Fix auth middleware",
    "description": "Session tokens non-compliant",
    "url": "https://linear.app/omninode/issue/OMN-1234",
    "priority": 2,
    "status": "In Progress",
    "assignee_name": "Jonah",
    "assignee_email": "jonah@example.com",
    "labels": ["bug", "security"],
    "created_at": "2026-04-01T00:00:00Z",
    "updated_at": "2026-04-07T00:00:00Z",
}


@pytest.fixture()
def adapter() -> AdapterTicketLinear:
    return AdapterTicketLinear(linear_api_key="lin_test_key_123")


def _mock_response(json_data: dict, status_code: int = 200) -> httpx.Response:
    """Build a mock httpx.Response."""
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = json_data
    response.raise_for_status = MagicMock()
    if status_code >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error",
            request=MagicMock(spec=httpx.Request),
            response=response,
        )
    return response


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


class TestNormaliseIssue:
    def test_full_issue(self) -> None:
        result = _normalise_issue(_SAMPLE_ISSUE_RAW)
        assert result == _EXPECTED_NORMALISED

    def test_missing_nested_fields(self) -> None:
        raw = {"id": "abc", "identifier": "X-1", "state": None, "assignee": None}
        result = _normalise_issue(raw)
        assert result["status"] == ""
        assert result["assignee_name"] is None
        assert result["labels"] == []


class TestBuildIssueFilter:
    def test_empty_filters(self) -> None:
        assert _build_issue_filter({}) == {}

    def test_status_filter(self) -> None:
        result = _build_issue_filter({"status": "In Progress"})
        assert result == {"state": {"name": {"eq": "In Progress"}}}

    def test_multiple_filters(self) -> None:
        result = _build_issue_filter({"status": "Done", "team": "OMN"})
        assert result == {
            "state": {"name": {"eq": "Done"}},
            "team": {"key": {"eq": "OMN"}},
        }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_empty_api_key_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty linear_api_key"):
            AdapterTicketLinear(linear_api_key="")

    def test_valid_construction(self, adapter: AdapterTicketLinear) -> None:
        assert adapter._api_key == "lin_test_key_123"


# ---------------------------------------------------------------------------
# get_ticket
# ---------------------------------------------------------------------------


class TestGetTicket:
    @pytest.mark.asyncio()
    async def test_get_ticket_by_uuid(self, adapter: AdapterTicketLinear) -> None:
        mock_resp = _mock_response({"data": {"issue": _SAMPLE_ISSUE_RAW}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            result = await adapter.get_ticket("a1b2c3d4-e5f6-7890-abcd-ef1234567890")
            assert result["identifier"] == "OMN-1234"
            assert result["status"] == "In Progress"

    @pytest.mark.asyncio()
    async def test_get_ticket_by_identifier(self, adapter: AdapterTicketLinear) -> None:
        mock_resp = _mock_response(
            {"data": {"issueSearch": {"nodes": [_SAMPLE_ISSUE_RAW]}}}
        )
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            result = await adapter.get_ticket("OMN-1234")
            assert result["identifier"] == "OMN-1234"

    @pytest.mark.asyncio()
    async def test_get_ticket_not_found_raises_key_error(
        self, adapter: AdapterTicketLinear
    ) -> None:
        mock_resp = _mock_response({"data": {"issue": None}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            with pytest.raises(KeyError, match="Ticket not found"):
                await adapter.get_ticket("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    @pytest.mark.asyncio()
    async def test_get_ticket_identifier_not_found(
        self, adapter: AdapterTicketLinear
    ) -> None:
        mock_resp = _mock_response({"data": {"issueSearch": {"nodes": []}}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            with pytest.raises(KeyError, match="Ticket not found"):
                await adapter.get_ticket("OMN-99999")


# ---------------------------------------------------------------------------
# list_tickets
# ---------------------------------------------------------------------------


class TestListTickets:
    @pytest.mark.asyncio()
    async def test_list_tickets_no_filters(self, adapter: AdapterTicketLinear) -> None:
        mock_resp = _mock_response({"data": {"issues": {"nodes": [_SAMPLE_ISSUE_RAW]}}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            result = await adapter.list_tickets()
            assert len(result) == 1
            assert result[0]["identifier"] == "OMN-1234"

    @pytest.mark.asyncio()
    async def test_list_tickets_with_filters(
        self, adapter: AdapterTicketLinear
    ) -> None:
        mock_resp = _mock_response({"data": {"issues": {"nodes": [_SAMPLE_ISSUE_RAW]}}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            result = await adapter.list_tickets(
                filters={"status": "In Progress"}, limit=10
            )
            assert len(result) == 1

    @pytest.mark.asyncio()
    async def test_list_tickets_empty(self, adapter: AdapterTicketLinear) -> None:
        mock_resp = _mock_response({"data": {"issues": {"nodes": []}}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            result = await adapter.list_tickets()
            assert result == []


# ---------------------------------------------------------------------------
# get_ticket_status
# ---------------------------------------------------------------------------


class TestGetTicketStatus:
    @pytest.mark.asyncio()
    async def test_get_ticket_status(self, adapter: AdapterTicketLinear) -> None:
        mock_resp = _mock_response({"data": {"issue": _SAMPLE_ISSUE_RAW}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            status = await adapter.get_ticket_status(
                "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            )
            assert status == "In Progress"


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio()
    async def test_health_check_success(self, adapter: AdapterTicketLinear) -> None:
        mock_resp = _mock_response({"data": {"viewer": {"id": "123"}}})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            assert await adapter.health_check() is True

    @pytest.mark.asyncio()
    async def test_health_check_failure(self, adapter: AdapterTicketLinear) -> None:
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("unreachable")
            mock_get_client.return_value = mock_client

            assert await adapter.health_check() is False


# ---------------------------------------------------------------------------
# Write methods — NotImplementedError
# ---------------------------------------------------------------------------


class TestWriteMethodsNotImplemented:
    @pytest.mark.asyncio()
    async def test_create_ticket_raises(self, adapter: AdapterTicketLinear) -> None:
        with pytest.raises(NotImplementedError, match="create_ticket"):
            await adapter.create_ticket(title="t", description="d")

    @pytest.mark.asyncio()
    async def test_update_ticket_status_raises(
        self, adapter: AdapterTicketLinear
    ) -> None:
        with pytest.raises(NotImplementedError, match="update_ticket_status"):
            await adapter.update_ticket_status("OMN-1", "done")

    @pytest.mark.asyncio()
    async def test_add_comment_raises(self, adapter: AdapterTicketLinear) -> None:
        with pytest.raises(NotImplementedError, match="add_comment"):
            await adapter.add_comment("OMN-1", "hello")


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    @pytest.mark.asyncio()
    async def test_http_error_raises_infra_connection_error(
        self, adapter: AdapterTicketLinear
    ) -> None:
        mock_resp = _mock_response({}, status_code=500)
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            with pytest.raises(InfraConnectionError, match="HTTP 500"):
                await adapter.get_ticket("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    @pytest.mark.asyncio()
    async def test_graphql_errors_raise_infra_unavailable(
        self, adapter: AdapterTicketLinear
    ) -> None:
        mock_resp = _mock_response({"errors": [{"message": "Rate limited"}]})
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_resp
            mock_get_client.return_value = mock_client

            with pytest.raises(InfraUnavailableError, match="GraphQL errors"):
                await adapter.get_ticket("a1b2c3d4-e5f6-7890-abcd-ef1234567890")

    @pytest.mark.asyncio()
    async def test_network_error_raises_infra_connection_error(
        self, adapter: AdapterTicketLinear
    ) -> None:
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.ConnectError("refused")
            mock_get_client.return_value = mock_client

            with pytest.raises(InfraConnectionError, match="connection failed"):
                await adapter.get_ticket("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio()
    async def test_close_with_client(self, adapter: AdapterTicketLinear) -> None:
        mock_client = AsyncMock()
        mock_client.is_closed = False
        adapter._client = mock_client

        await adapter.close()
        mock_client.aclose.assert_awaited_once()
        assert adapter._client is None

    @pytest.mark.asyncio()
    async def test_close_without_client(self, adapter: AdapterTicketLinear) -> None:
        # Should not raise
        await adapter.close()
