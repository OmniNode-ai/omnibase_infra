# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Gmail API Handler - Shared OAuth2 + REST transport layer.

This handler provides the shared transport layer for all Gmail nodes.
It manages OAuth2 token refresh internally and exposes a clean async API
for listing, reading, searching, and modifying Gmail messages.

Architecture:
    Follows the ONEX operation handler pattern (mirrors HandlerSlackWebhook):
    - Credentials from env vars: GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET,
      GMAIL_REFRESH_TOKEN
    - Manages OAuth2 access token refresh internally (100s buffer before
      Gmail's guaranteed 3600s lifetime)
    - Label resolution: config uses label names, API uses IDs; results
      are cached with 5-minute TTL
    - Failure semantics:
        * list_messages / search_messages fail → returns [] (hard_failed)
        * get_message / modify_labels / delete_message fail → returns
          empty dict / False (caller skip-and-continue)

Configuration:
    - GMAIL_CLIENT_ID: OAuth2 client ID
    - GMAIL_CLIENT_SECRET: OAuth2 client secret
    - GMAIL_REFRESH_TOKEN: OAuth2 refresh token (long-lived)

Infisical Paths:
    - /services/omnibase_infra/gmail/GMAIL_CLIENT_ID
    - /services/omnibase_infra/gmail/GMAIL_CLIENT_SECRET
    - /services/omnibase_infra/gmail/GMAIL_REFRESH_TOKEN

Related Tickets:
    - OMN-2729: Add HandlerGmailApi shared OAuth2 + REST client
    - OMN-2728: Gmail Integration epic (omnibase_infra)
"""

from __future__ import annotations

import logging
import os
import time
from typing import cast

import httpx

from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Gmail OAuth2 token endpoint
_GMAIL_TOKEN_URL: str = "https://oauth2.googleapis.com/token"

# Gmail REST API base URL
_GMAIL_API_BASE: str = "https://gmail.googleapis.com/gmail/v1/users/me"

# Refresh token when expiry is within this many seconds
_TOKEN_REFRESH_BUFFER_SECONDS: int = 100

# Label cache TTL in seconds (5 minutes)
_LABEL_CACHE_TTL_SECONDS: int = 300

# Type alias for a raw JSON API response dict
_ApiDict = dict[str, JsonType]


class HandlerGmailApi:
    """Shared OAuth2 + REST transport layer for Gmail API access.

    Manages token refresh internally and provides a clean async API
    for all Gmail operations needed by ONEX Gmail nodes.

    Token Management:
        - Access tokens are obtained via OAuth2 refresh token grant.
        - Token is proactively refreshed when within 100s of expiry.
        - Refresh failure raises RuntimeError (sanitized); nothing
          works without a valid token.

    Label Resolution:
        - Gmail API uses label IDs internally; human-readable label
          names are used in ONEX config.
        - ``resolve_label_ids()`` translates name → ID with a 5-minute
          TTL cache (refreshed lazily on next call after expiry).

    Failure Semantics:
        - ``list_messages`` / ``search_messages``: returns ``[]`` on
          failure (caller should mark ``hard_failed=True``).
        - ``get_message``: returns ``{}`` on failure (caller skip).
        - ``modify_labels`` / ``delete_message``: returns ``False`` on
          failure (caller skip-and-continue).
        - ``list_labels``: returns ``[]`` on failure.

    Attributes:
        _client_id: OAuth2 client ID from env.
        _client_secret: OAuth2 client secret from env.
        _refresh_token: OAuth2 refresh token from env.
        _access_token: Current access token (empty until first refresh).
        _token_expiry: Unix timestamp when current token expires.
        _label_cache: Cached {name: id} mapping.
        _label_cache_expires_at: Unix timestamp when label cache expires.
        _http_client: Optional shared httpx.AsyncClient.

    Example:
        >>> handler = HandlerGmailApi()
        >>> messages = await handler.list_messages(["INBOX"], max_results=10)
        >>> msg = await handler.get_message(messages[0]["id"])
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        refresh_token: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize handler with OAuth2 credentials.

        Args:
            client_id: OAuth2 client ID. If not provided, reads from
                GMAIL_CLIENT_ID environment variable.
            client_secret: OAuth2 client secret. If not provided, reads
                from GMAIL_CLIENT_SECRET environment variable.
            refresh_token: OAuth2 refresh token. If not provided, reads
                from GMAIL_REFRESH_TOKEN environment variable.
            http_client: Optional shared httpx.AsyncClient. If not
                provided, a new client is created per request batch.
        """
        self._client_id: str = (
            client_id if client_id is not None else os.getenv("GMAIL_CLIENT_ID", "")
        )
        self._client_secret: str = (
            client_secret
            if client_secret is not None
            else os.getenv("GMAIL_CLIENT_SECRET", "")
        )
        self._refresh_token: str = (
            refresh_token
            if refresh_token is not None
            else os.getenv("GMAIL_REFRESH_TOKEN", "")
        )
        self._access_token: str = ""
        self._token_expiry: float = 0.0
        self._label_cache: dict[str, str] = {}
        self._label_cache_expires_at: float = 0.0
        self._http_client = http_client

    def __repr__(self) -> str:
        """Mask credentials to prevent accidental exposure in logs."""
        has_token = bool(self._access_token)
        return f"<{type(self).__name__} token_active={has_token}>"

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed.

        Proactively refreshes the token when within
        ``_TOKEN_REFRESH_BUFFER_SECONDS`` of expiry.

        Returns:
            Valid access token string.

        Raises:
            RuntimeError: If token refresh fails. The error message is
                sanitized to avoid leaking credentials.
        """
        now = time.monotonic()
        if (
            self._access_token
            and now < self._token_expiry - _TOKEN_REFRESH_BUFFER_SECONDS
        ):
            return self._access_token

        # Refresh needed
        if not self._client_id or not self._client_secret or not self._refresh_token:
            raise RuntimeError(
                "Gmail OAuth2 credentials not configured. "
                "Set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, GMAIL_REFRESH_TOKEN."
            )

        logger.debug("Refreshing Gmail OAuth2 access token")

        try:
            client_created = False
            client = self._http_client
            if client is None:
                client = httpx.AsyncClient()
                client_created = True

            try:
                response = await client.post(
                    _GMAIL_TOKEN_URL,
                    data={
                        "client_id": self._client_id,
                        "client_secret": self._client_secret,
                        "refresh_token": self._refresh_token,
                        "grant_type": "refresh_token",
                    },
                    timeout=10.0,
                )
            finally:
                if client_created:
                    await client.aclose()

            if response.status_code != 200:
                raise RuntimeError(
                    f"Gmail token refresh failed with HTTP {response.status_code}"
                )

            body: _ApiDict = response.json()
            access_token = str(body.get("access_token", ""))
            raw_expires = body.get("expires_in", 3600)
            if isinstance(raw_expires, int):
                expires_in: int = raw_expires
            elif isinstance(raw_expires, (float, str)):
                expires_in = int(raw_expires)
            else:
                expires_in = 3600

            if not access_token:
                raise RuntimeError("Gmail token refresh returned empty access_token")

            self._access_token = access_token
            self._token_expiry = time.monotonic() + expires_in
            logger.debug(
                "Gmail OAuth2 token refreshed",
                extra={"expires_in": expires_in},
            )
            return self._access_token

        except RuntimeError:
            raise
        except Exception as exc:
            # Sanitize: do not include credentials in the error message
            raise RuntimeError(
                f"Gmail token refresh encountered an unexpected error: {type(exc).__name__}"
            ) from None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_messages(
        self,
        label_ids: list[str],
        max_results: int = 50,
    ) -> list[_ApiDict]:
        """List Gmail messages matching the given label IDs.

        Args:
            label_ids: List of Gmail label ID strings to filter by.
            max_results: Maximum number of message stubs to return.
                Default is 50. Gmail API maximum per page is 500.

        Returns:
            List of message stub dicts (each has ``id`` and ``threadId``).
            Returns empty list on any error (caller marks hard_failed).
        """
        try:
            token = await self._ensure_token()
            # httpx accepts Sequence values for multi-value params
            params: dict[str, object] = {
                "maxResults": max_results,
                "labelIds": label_ids,
            }
            result = await self._get(
                f"{_GMAIL_API_BASE}/messages",
                params=params,
                token=token,
            )
            raw_messages = result.get("messages", [])
            messages: list[_ApiDict] = _extract_list_of_dicts(raw_messages)
            return messages
        except Exception as exc:
            logger.warning(
                "HandlerGmailApi.list_messages failed",
                extra={"error": str(exc), "label_ids": label_ids},
            )
            return []

    async def get_message(
        self,
        message_id: str,
        message_format: str = "full",
    ) -> _ApiDict:
        """Fetch a single Gmail message by ID.

        Args:
            message_id: Gmail message ID string.
            message_format: Gmail API format parameter. Default is "full".
                Other values: "minimal", "raw", "metadata".

        Returns:
            Raw message dict from Gmail API, or empty dict on error
            (caller should skip-and-continue).
        """
        try:
            token = await self._ensure_token()
            result = await self._get(
                f"{_GMAIL_API_BASE}/messages/{message_id}",
                params={"format": message_format},
                token=token,
            )
            return result
        except Exception as exc:
            logger.warning(
                "HandlerGmailApi.get_message failed",
                extra={"error": str(exc), "message_id": message_id},
            )
            return {}

    async def modify_labels(
        self,
        message_id: str,
        add_label_ids: list[str],
        remove_label_ids: list[str],
    ) -> bool:
        """Add and/or remove labels on a Gmail message.

        Args:
            message_id: Gmail message ID string.
            add_label_ids: Label IDs to add.
            remove_label_ids: Label IDs to remove.

        Returns:
            True if operation succeeded, False on error (caller skip).
        """
        try:
            token = await self._ensure_token()
            # Build body dict with explicit list[str] values
            body: _ApiDict = {
                "addLabelIds": list(add_label_ids),
                "removeLabelIds": list(remove_label_ids),
            }
            await self._post(
                f"{_GMAIL_API_BASE}/messages/{message_id}/modify",
                body=body,
                token=token,
            )
            return True
        except Exception as exc:
            logger.warning(
                "HandlerGmailApi.modify_labels failed",
                extra={"error": str(exc), "message_id": message_id},
            )
            return False

    async def delete_message(self, message_id: str) -> bool:
        """Permanently delete a Gmail message.

        Note: This bypasses the trash. Use with caution.

        Args:
            message_id: Gmail message ID string.

        Returns:
            True if deletion succeeded, False on error (caller skip).
        """
        try:
            token = await self._ensure_token()
            await self._delete(
                f"{_GMAIL_API_BASE}/messages/{message_id}",
                token=token,
            )
            return True
        except Exception as exc:
            logger.warning(
                "HandlerGmailApi.delete_message failed",
                extra={"error": str(exc), "message_id": message_id},
            )
            return False

    async def search_messages(
        self,
        query: str,
        max_results: int = 500,
    ) -> list[_ApiDict]:
        """Search Gmail messages using Gmail query syntax.

        Handles pagination automatically up to max_results.

        Args:
            query: Gmail search query string (e.g., "from:noreply@example.com").
            max_results: Maximum total messages to return across all pages.
                Default is 500.

        Returns:
            List of message stub dicts (each has ``id`` and ``threadId``).
            Returns empty list on any error (caller marks hard_failed).
        """
        try:
            token = await self._ensure_token()
            messages: list[_ApiDict] = []
            page_token: str | None = None
            page_size = min(max_results, 500)

            while len(messages) < max_results:
                remaining = max_results - len(messages)
                params: dict[str, object] = {
                    "q": query,
                    "maxResults": min(page_size, remaining),
                }
                if page_token:
                    params["pageToken"] = page_token

                result = await self._get(
                    f"{_GMAIL_API_BASE}/messages",
                    params=params,
                    token=token,
                )
                raw_page = result.get("messages", [])
                page_messages: list[_ApiDict] = _extract_list_of_dicts(raw_page)
                messages.extend(page_messages)

                next_token = result.get("nextPageToken")
                page_token = str(next_token) if next_token is not None else None
                if not page_token or not page_messages:
                    break

            return messages[:max_results]

        except Exception as exc:
            logger.warning(
                "HandlerGmailApi.search_messages failed",
                extra={"error": str(exc), "query": query},
            )
            return []

    async def list_labels(self) -> list[_ApiDict]:
        """List all Gmail labels for the authenticated user.

        Returns:
            List of label dicts, each with at minimum ``id`` and ``name``
            keys. Returns empty list on error.
        """
        try:
            token = await self._ensure_token()
            result = await self._get(
                f"{_GMAIL_API_BASE}/labels",
                params={},
                token=token,
            )
            raw_labels = result.get("labels", [])
            labels: list[_ApiDict] = _extract_list_of_dicts(raw_labels)
            return labels
        except Exception as exc:
            logger.warning(
                "HandlerGmailApi.list_labels failed",
                extra={"error": str(exc)},
            )
            return []

    async def resolve_label_ids(
        self,
        label_names: list[str],
    ) -> dict[str, str]:
        """Resolve label names to Gmail label IDs, with 5-minute TTL cache.

        Translates human-readable label names (used in ONEX config) to
        Gmail label ID strings (required by the API). Cache is refreshed
        lazily after the TTL expires.

        Args:
            label_names: List of label name strings to resolve
                (e.g., ["INBOX", "MyCustomLabel"]).

        Returns:
            Dict mapping name → ID for all names that could be resolved.
            Names that don't match any label are omitted from the result.
        """
        now = time.monotonic()
        if not self._label_cache or now >= self._label_cache_expires_at:
            # Refresh cache
            raw_labels = await self.list_labels()
            cache: dict[str, str] = {}
            for label in raw_labels:
                name = str(label.get("name", ""))
                label_id = str(label.get("id", ""))
                if name and label_id:
                    cache[name] = label_id
            self._label_cache = cache
            self._label_cache_expires_at = now + _LABEL_CACHE_TTL_SECONDS
            logger.debug(
                "Gmail label cache refreshed",
                extra={"label_count": len(cache)},
            )

        return {
            name: self._label_cache[name]
            for name in label_names
            if name in self._label_cache
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get(
        self,
        url: str,
        params: dict[str, object],
        token: str,
    ) -> _ApiDict:
        """Execute an authenticated GET request.

        Args:
            url: Full URL to request.
            params: Query parameters (str, int, or list[str] values).
            token: OAuth2 access token.

        Returns:
            Parsed JSON response dict.

        Raises:
            RuntimeError: On non-2xx HTTP status or network error.
        """
        headers = {"Authorization": f"Bearer {token}"}
        client_created = False
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient()
            client_created = True

        # Build httpx-compatible param list, expanding list values for multi-value params.
        # Coerce to list[tuple[str, str | None]] which httpx accepts; cast for mypy.
        param_list: list[tuple[str, str | None]] = []
        for key, val in params.items():
            if isinstance(val, list):
                for item in val:
                    param_list.append((key, str(item)))
            else:
                param_list.append((key, str(val) if val is not None else None))
        # cast: list[tuple[str, str | None]] is a valid subtype of httpx QueryParams
        httpx_params = cast(
            "list[tuple[str, str | int | float | bool | None]]",
            param_list,
        )

        try:
            response = await client.get(
                url, params=httpx_params, headers=headers, timeout=30.0
            )
            if response.status_code not in (200, 204):
                raise RuntimeError(
                    f"Gmail API GET {url} returned HTTP {response.status_code}"
                )
            if response.status_code == 204:
                return {}
            result: _ApiDict = response.json()
            return result
        finally:
            if client_created:
                await client.aclose()

    async def _post(
        self,
        url: str,
        body: _ApiDict,
        token: str,
    ) -> _ApiDict:
        """Execute an authenticated POST request.

        Args:
            url: Full URL to request.
            body: JSON body dict.
            token: OAuth2 access token.

        Returns:
            Parsed JSON response dict.

        Raises:
            RuntimeError: On non-2xx HTTP status or network error.
        """
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        client_created = False
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient()
            client_created = True

        try:
            response = await client.post(url, json=body, headers=headers, timeout=30.0)
            if response.status_code not in (200, 204):
                raise RuntimeError(
                    f"Gmail API POST {url} returned HTTP {response.status_code}"
                )
            if response.status_code == 204:
                return {}
            result: _ApiDict = response.json()
            return result
        finally:
            if client_created:
                await client.aclose()

    async def _delete(
        self,
        url: str,
        token: str,
    ) -> None:
        """Execute an authenticated DELETE request.

        Args:
            url: Full URL to request.
            token: OAuth2 access token.

        Raises:
            RuntimeError: On non-2xx HTTP status or network error.
        """
        headers = {"Authorization": f"Bearer {token}"}
        client_created = False
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient()
            client_created = True

        try:
            response = await client.delete(url, headers=headers, timeout=30.0)
            if response.status_code not in (200, 204):
                raise RuntimeError(
                    f"Gmail API DELETE {url} returned HTTP {response.status_code}"
                )
        finally:
            if client_created:
                await client.aclose()


def _extract_list_of_dicts(value: JsonType) -> list[_ApiDict]:
    """Safely extract a list of dicts from a JsonType value.

    Args:
        value: A JSON value that may be a list of dict objects.

    Returns:
        List of dict[str, JsonType] items, or empty list if value is not
        a list or contains non-dict items.
    """
    if not isinstance(value, list):
        return []
    result: list[_ApiDict] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
    return result


__all__: list[str] = ["HandlerGmailApi"]
