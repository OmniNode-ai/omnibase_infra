# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17664: the closer writes as an application, not as a person.

Measured 2026-09-05T20:33Z against OMN-17957 and recorded in the handler's own
comment at ``_prior_revert_reason``: every entry in that ticket's history,
including the sweep's OWN 19:36:02.430Z flip, carries
``actorId 7a850ce1-f95e-431f-b4e3-62f7449f04c0``. ``LINEAR_API_KEY`` is a
PERSONAL key and Linear attributes its writes to the person who minted it, so a
closer flip is indistinguishable from a human flip on the only surface that
records who did it. Re-confirmed live 2026-09-13: that uuid is the ``viewer.id``
the workspace key resolves to.

These tests pin the three states the credential resolution can be in and the
one it must refuse:

* both application secrets present  -> exchange them for an app actor token and
  send it as a bearer, so the write attributes to the application;
* exactly one present               -> REFUSE, and do not silently fall back to
  the personal key, because a half-configured app identity is a deployment
  error and falling through would write as a person while the operator believes
  the app path is live;
* neither present, personal key set -> the documented fallback, taken loudly so
  the run says in its own log that its writes will carry a person's name;
* nothing set                       -> no HTTP at all.

The secret value never reaches a log record or ``last_error`` on any path.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
    handler_evidence_autoclose_sweep as sweep_mod,
)
from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.enum_linear_identity_path import (
    EnumLinearIdentityPath,
)

_LinearClient = sweep_mod._LinearClient

_QUERY = "query Q { ok }"
_GRAPHQL_REQUEST = httpx.Request("POST", "https://api.linear.app/graphql")

# Deliberately not a plausible credential: these tests assert that whatever is
# configured never escapes into a log line, so the fixture value has to be
# searchable and obviously synthetic.
_FAKE_CLIENT_SECRET = "synthetic-not-a-real-secret-0000"
_FAKE_ACCESS_TOKEN = "synthetic-access-token-1111"
_FAKE_PERSONAL_KEY = "synthetic-personal-key-2222"


def _response(status: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status_code=status, json=payload, request=_GRAPHQL_REQUEST)


def _token_ok() -> httpx.Response:
    return _response(
        200,
        {
            "access_token": _FAKE_ACCESS_TOKEN,
            "token_type": "Bearer",
            "expires_in": 2591999,
            "scope": "read write",
        },
    )


def _graphql_ok() -> httpx.Response:
    return _response(200, {"data": {"ok": True}})


class _RecordingAsyncClient:
    """Records every POST (url, headers, form data, json body) in order."""

    script: list[object] = []
    posts: list[dict[str, Any]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _RecordingAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, url: str, **kwargs: object) -> httpx.Response:
        index = len(type(self).posts)
        type(self).posts.append(
            {
                "url": url,
                "headers": dict(kwargs.get("headers") or {}),
                "data": dict(kwargs.get("data") or {}),
                "json": kwargs.get("json"),
            }
        )
        item = type(self).script[min(index, len(type(self).script) - 1)]
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, httpx.Response)
        return item


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingAsyncClient]:
    _RecordingAsyncClient.script = []
    _RecordingAsyncClient.posts = []
    monkeypatch.setattr(sweep_mod.httpx, "AsyncClient", _RecordingAsyncClient)
    return _RecordingAsyncClient


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """No Linear credential of any kind is inherited from the running shell."""
    for name in (
        sweep_mod._LINEAR_API_KEY_ENV,
        sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV,
        sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.unit
@pytest.mark.asyncio
async def test_application_secrets_are_exchanged_for_a_bearer_token(
    recording: type[_RecordingAsyncClient], clean_env: pytest.MonkeyPatch
) -> None:
    """The app path: one token POST, then the query carries the bearer."""
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV, "client-id-abc")
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV, _FAKE_CLIENT_SECRET)
    recording.script = [_token_ok(), _graphql_ok()]

    client = _LinearClient(max_attempts=1, base_delay_seconds=0.0)
    assert await client._query(_QUERY, {}) == {"ok": True}

    assert len(recording.posts) == 2
    token_post, graphql_post = recording.posts
    assert token_post["url"] == sweep_mod._LINEAR_OAUTH_TOKEN_URL
    # Linear's documented client-credentials parameters, form-encoded. `scope`
    # is REQUIRED by that endpoint and is comma separated, not space separated.
    assert token_post["data"]["grant_type"] == "client_credentials"
    assert token_post["data"]["client_id"] == "client-id-abc"
    assert token_post["data"]["client_secret"] == _FAKE_CLIENT_SECRET
    assert token_post["data"]["scope"] == sweep_mod._LINEAR_CLOSER_TOKEN_SCOPES
    assert "," in sweep_mod._LINEAR_CLOSER_TOKEN_SCOPES

    assert graphql_post["url"] == sweep_mod._LINEAR_API_URL
    assert graphql_post["headers"]["Authorization"] == f"Bearer {_FAKE_ACCESS_TOKEN}"
    assert client.identity_path is EnumLinearIdentityPath.OAUTH_APPLICATION
    assert client.last_error == ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_token_is_exchanged_once_and_reused_for_the_whole_run(
    recording: type[_RecordingAsyncClient], clean_env: pytest.MonkeyPatch
) -> None:
    """Linear's docs ask for one token per run, not one per request."""
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV, "client-id-abc")
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV, _FAKE_CLIENT_SECRET)
    recording.script = [_token_ok(), _graphql_ok(), _graphql_ok()]

    client = _LinearClient(max_attempts=1, base_delay_seconds=0.0)
    assert await client._query(_QUERY, {}) == {"ok": True}
    assert await client._query(_QUERY, {}) == {"ok": True}

    token_posts = [
        post
        for post in recording.posts
        if post["url"] == sweep_mod._LINEAR_OAUTH_TOKEN_URL
    ]
    assert len(token_posts) == 1
    assert len(recording.posts) == 3


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("present", ["id", "secret"])
async def test_partial_application_config_refuses_and_makes_no_call(
    recording: type[_RecordingAsyncClient],
    clean_env: pytest.MonkeyPatch,
    present: str,
) -> None:
    """Half a configured app identity is an error, never a quiet fallback.

    The personal key IS set here. Falling through to it would produce a run
    that writes as a person while the operator who set one of the two secrets
    believes the application path is live — the exact silent-degradation class
    OMN-16832 removed from this workflow's GitHub credential.
    """
    if present == "id":
        clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV, "client-id-abc")
    else:
        clean_env.setenv(
            sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV, _FAKE_CLIENT_SECRET
        )
    clean_env.setenv(sweep_mod._LINEAR_API_KEY_ENV, _FAKE_PERSONAL_KEY)
    recording.script = [_graphql_ok()]

    client = _LinearClient(max_attempts=1, base_delay_seconds=0.0)
    assert await client._query(_QUERY, {}) is None

    assert recording.posts == []
    assert client.identity_path is EnumLinearIdentityPath.MISCONFIGURED
    assert sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV in client.last_error
    assert sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV in client.last_error
    assert _FAKE_CLIENT_SECRET not in client.last_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_personal_key_fallback_is_taken_and_is_logged_as_such(
    recording: type[_RecordingAsyncClient],
    clean_env: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The fallback is explicit: the run says whose name its writes will carry."""
    clean_env.setenv(sweep_mod._LINEAR_API_KEY_ENV, _FAKE_PERSONAL_KEY)
    recording.script = [_graphql_ok()]

    with caplog.at_level(logging.INFO, logger=sweep_mod.logger.name):
        client = _LinearClient(max_attempts=1, base_delay_seconds=0.0)
        assert await client._query(_QUERY, {}) == {"ok": True}

    assert len(recording.posts) == 1
    assert recording.posts[0]["url"] == sweep_mod._LINEAR_API_URL
    assert recording.posts[0]["headers"]["Authorization"] == _FAKE_PERSONAL_KEY
    assert client.identity_path is EnumLinearIdentityPath.PERSONAL_API_KEY

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert EnumLinearIdentityPath.PERSONAL_API_KEY.value in rendered
    assert _FAKE_PERSONAL_KEY not in rendered


@pytest.mark.unit
@pytest.mark.asyncio
async def test_application_path_is_named_in_the_log_when_taken(
    recording: type[_RecordingAsyncClient],
    clean_env: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run log names the identity path — never the token."""
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV, "client-id-abc")
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV, _FAKE_CLIENT_SECRET)
    recording.script = [_token_ok(), _graphql_ok()]

    with caplog.at_level(logging.INFO, logger=sweep_mod.logger.name):
        client = _LinearClient(max_attempts=1, base_delay_seconds=0.0)
        assert await client._query(_QUERY, {}) == {"ok": True}

    rendered = "\n".join(record.getMessage() for record in caplog.records)
    assert EnumLinearIdentityPath.OAUTH_APPLICATION.value in rendered
    assert _FAKE_ACCESS_TOKEN not in rendered
    assert _FAKE_CLIENT_SECRET not in rendered


@pytest.mark.unit
@pytest.mark.asyncio
async def test_token_exchange_failure_fails_closed_without_a_graphql_call(
    recording: type[_RecordingAsyncClient], clean_env: pytest.MonkeyPatch
) -> None:
    """A rejected exchange must not fall through to the personal key."""
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV, "client-id-abc")
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV, _FAKE_CLIENT_SECRET)
    clean_env.setenv(sweep_mod._LINEAR_API_KEY_ENV, _FAKE_PERSONAL_KEY)
    recording.script = [_response(401, {"error": "invalid_client"}), _graphql_ok()]

    client = _LinearClient(max_attempts=1, base_delay_seconds=0.0)
    assert await client._query(_QUERY, {}) is None

    assert len(recording.posts) == 1
    assert recording.posts[0]["url"] == sweep_mod._LINEAR_OAUTH_TOKEN_URL
    assert client.last_error
    assert _FAKE_CLIENT_SECRET not in client.last_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_credential_at_all_makes_no_call(
    recording: type[_RecordingAsyncClient], clean_env: pytest.MonkeyPatch
) -> None:
    recording.script = [_graphql_ok()]
    client = _LinearClient(max_attempts=1, base_delay_seconds=0.0)
    assert await client._query(_QUERY, {}) is None
    assert recording.posts == []
    assert client.last_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_explicit_injected_key_bypasses_env_resolution_entirely(
    recording: type[_RecordingAsyncClient], clean_env: pytest.MonkeyPatch
) -> None:
    """An injected key is a test seam, so it must not trigger an exchange.

    Pins the pre-existing constructor contract the retry tests rely on: an
    explicitly passed ``api_key`` is used verbatim, and an explicitly passed
    empty string still means "no credential" rather than "read the env".
    """
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV, "client-id-abc")
    clean_env.setenv(sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV, _FAKE_CLIENT_SECRET)
    recording.script = [_graphql_ok()]

    client = _LinearClient(
        api_key=_FAKE_PERSONAL_KEY, max_attempts=1, base_delay_seconds=0.0
    )
    assert await client._query(_QUERY, {}) == {"ok": True}
    assert len(recording.posts) == 1
    assert recording.posts[0]["url"] == sweep_mod._LINEAR_API_URL

    empty = _LinearClient(api_key="", max_attempts=1, base_delay_seconds=0.0)
    assert await empty._query(_QUERY, {}) is None
    assert len(recording.posts) == 1


@pytest.mark.unit
def test_every_credential_env_name_is_self_declared() -> None:
    """OMN-14951 gap 2: this boundary file declares the names it reads."""
    declared = set(_LinearClient.required_secrets)
    assert sweep_mod._LINEAR_API_KEY_ENV in declared
    assert sweep_mod._LINEAR_CLOSER_CLIENT_ID_ENV in declared
    assert sweep_mod._LINEAR_CLOSER_CLIENT_SECRET_ENV in declared
