# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-16106: the closer's Linear reads survive a transient fault.

Every Linear read in ``handler_evidence_autoclose_sweep`` is consumed by a
fail-closed caller, so a single lost HTTP call ends the candidate's run as
``ERROR_LINEAR_API`` with no verdict reached about it. Measured on the live
30-minute schedule 2026-09-05, runs 33970676719 (14:03Z) and 33972096907
(14:32Z): 8-of-18 then 5-of-13 bound candidates were dropped that way, and
OMN-17160 / OMN-17934 errored on the first tick and reached a real verdict on
the second with nothing about them changed — the failure is a property of the
attempt, not of the ticket.

These tests pin the split the fix turns on: a fault whose cause lives in the
ATTEMPT is retried, and a fault whose cause is deterministic (credential,
binding, malformed query) still fails on the first answer, because retrying it
reproduces it exactly and only burns the run's budget.
"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.handlers import (
    handler_evidence_autoclose_sweep as sweep_mod,
)

_LinearClient = sweep_mod._LinearClient

_QUERY = "query Q { ok }"
_REQUEST = httpx.Request("POST", "https://api.linear.app/graphql")


def _response(
    status: int,
    payload: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=payload if payload is not None else {},
        headers=headers or {},
        request=_REQUEST,
    )


class _ScriptedAsyncClient:
    """Replays a queued script of responses/exceptions, counting attempts."""

    script: list[object] = []
    calls: int = 0

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> _ScriptedAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def post(self, *args: object, **kwargs: object) -> httpx.Response:
        index = type(self).calls
        type(self).calls += 1
        item = type(self).script[min(index, len(type(self).script) - 1)]
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, httpx.Response)
        return item


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch) -> type[_ScriptedAsyncClient]:
    """Replace the httpx client the sweep constructs inside ``_query``."""
    _ScriptedAsyncClient.script = []
    _ScriptedAsyncClient.calls = 0
    monkeypatch.setattr(sweep_mod.httpx, "AsyncClient", _ScriptedAsyncClient)
    return _ScriptedAsyncClient


def _client(max_attempts: int = 4) -> object:
    # base_delay 0.0 keeps every retry path exercisable without waiting out a
    # production backoff — the same reason `readback_delay_seconds` allows 0.
    return _LinearClient(
        api_key="lin_test", max_attempts=max_attempts, base_delay_seconds=0.0
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transport_error_then_success_returns_data(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    """The live failure mode: one lost call, then the same query succeeds."""
    scripted.script = [
        httpx.ConnectTimeout("timed out", request=_REQUEST),
        _response(200, {"data": {"issue": {"id": "abc"}}}),
    ]
    result = await _client()._query(_QUERY, {})
    assert result == {"issue": {"id": "abc"}}
    assert scripted.calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_success_clears_last_error(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    """A recovered call must not leave a stale cause for the next reason line."""
    scripted.script = [
        httpx.ReadTimeout("timed out", request=_REQUEST),
        _response(200, {"data": {"issue": {"id": "abc"}}}),
    ]
    client = _client()
    await client._query(_QUERY, {})
    assert client.last_error == ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_429_is_retried(scripted: type[_ScriptedAsyncClient]) -> None:
    scripted.script = [
        _response(429, {}, {"Retry-After": "0"}),
        _response(200, {"data": {"team": {}}}),
    ]
    assert await _client()._query(_QUERY, {}) == {"team": {}}
    assert scripted.calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_500_retries_to_the_cap_then_fails_closed(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    """Exhausting the budget still returns None — the fail-closed contract."""
    scripted.script = [_response(503, {})]
    client = _client(max_attempts=3)
    assert await client._query(_QUERY, {}) is None
    assert scripted.calls == 3
    assert "503" in client.last_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_http_401_is_not_retried(scripted: type[_ScriptedAsyncClient]) -> None:
    """A credential defect reproduces exactly; a retry only burns budget."""
    scripted.script = [_response(401, {})]
    client = _client()
    assert await client._query(_QUERY, {}) is None
    assert scripted.calls == 1
    assert client.last_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graphql_rate_limit_error_is_retried(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    """Linear answers an over-budget request 200 + a GraphQL error payload.

    A status-code-only classifier calls that a malformed query and refuses to
    retry it, which is the exact shape this fix must not reintroduce.
    """
    scripted.script = [
        _response(
            200,
            {
                "errors": [
                    {
                        "message": "Rate limit exceeded",
                        "extensions": {"code": "RATELIMITED"},
                    }
                ]
            },
        ),
        _response(200, {"data": {"issue": {"id": "abc"}}}),
    ]
    assert await _client()._query(_QUERY, {}) == {"issue": {"id": "abc"}}
    assert scripted.calls == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_graphql_query_error_is_not_retried(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    scripted.script = [
        _response(200, {"errors": [{"message": "Cannot query field 'nope'"}]})
    ]
    client = _client()
    assert await client._query(_QUERY, {}) is None
    assert scripted.calls == 1
    assert client.last_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_max_attempts_one_restores_single_shot_behaviour(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    """The policy must be able to express the pre-fix behaviour exactly."""
    scripted.script = [httpx.ConnectError("refused", request=_REQUEST)]
    assert await _client(max_attempts=1)._query(_QUERY, {}) is None
    assert scripted.calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_api_key_makes_no_call_at_all(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    scripted.script = [_response(200, {"data": {}})]
    client = _LinearClient(api_key="", max_attempts=4, base_delay_seconds=0.0)
    assert await client._query(_QUERY, {}) is None
    assert scripted.calls == 0
    assert "LINEAR_API_KEY" in client.last_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_hundred_without_data_object_is_not_retried(
    scripted: type[_ScriptedAsyncClient],
) -> None:
    """An uninterpretable shape is deterministic, so it fails on first answer."""
    scripted.script = [_response(200, {"unexpected": True})]
    client = _client()
    assert await client._query(_QUERY, {}) is None
    assert scripted.calls == 1
    assert client.last_error


@pytest.mark.unit
@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ({"Retry-After": "2"}, 2.0),
        ({"Retry-After": "0"}, 0.0),
        ({}, None),
        ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"Retry-After": "-5"}, None),
        ({"Retry-After": "3600"}, None),
    ],
)
def test_retry_after_parsing(header: dict[str, str], expected: float | None) -> None:
    """Only a sane delta-seconds value is honoured.

    An hour-long instruction inside a 60-minute job budget spends the run
    rather than saving it, so it is ignored in favour of the computed backoff.
    """
    assert _LinearClient._retry_after_seconds(_response(429, {}, header)) == expected


@pytest.mark.unit
def test_backoff_grows_and_is_capped() -> None:
    client = _LinearClient(api_key="k", max_attempts=8, base_delay_seconds=1.0)
    first = client._backoff_seconds(0)
    assert 0.5 <= first <= 1.0
    later = client._backoff_seconds(3)
    assert 4.0 <= later <= 8.0
    assert client._backoff_seconds(20) <= sweep_mod._LINEAR_RETRY_MAX_HONOURED_DELAY_S


@pytest.mark.unit
def test_zero_base_delay_never_sleeps() -> None:
    client = _LinearClient(api_key="k", max_attempts=4, base_delay_seconds=0.0)
    assert client._backoff_seconds(0) == 0.0
    assert client._backoff_seconds(5) == 0.0


@pytest.mark.unit
def test_apply_retry_policy_is_what_the_contract_drives() -> None:
    client = _LinearClient(api_key="k")
    client.apply_retry_policy(max_attempts=7, base_delay_seconds=2.5)
    assert client._max_attempts == 7
    assert client._base_delay_seconds == 2.5
    # Floors, so a caller cannot express "zero attempts" (which would be a
    # silent no-op read that reads as a clean failure) or a negative sleep.
    client.apply_retry_policy(max_attempts=0, base_delay_seconds=-1.0)
    assert client._max_attempts == 1
    assert client._base_delay_seconds == 0.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handle_binds_the_contract_declared_policy_to_the_real_client() -> None:
    """The retry policy is the CONTRACT's, not a constant in the client.

    The client is built in ``__init__``, before any request exists, so a
    policy that never reached it would be a contract field that reads as
    configuration and configures nothing (the OMN-17935 defect one field
    over). This pins the binding at the one point the request is in hand.
    """
    from omnibase_infra.nodes.node_evidence_autoclose_sweep_effect.models.model_evidence_autoclose_sweep_request import (
        ModelEvidenceAutocloseSweepRequest,
    )

    async def _no_companions(
        args: list[str], timeout: float
    ) -> tuple[object | None, str]:
        return [], ""

    client = _LinearClient(api_key="lin_test")
    handler = sweep_mod.HandlerEvidenceAutocloseSweep(
        linear_client=client, run_gh_command=_no_companions
    )
    await handler.handle(
        ModelEvidenceAutocloseSweepRequest(
            correlation_id=uuid4(),
            occ_repo="OmniNode-ai/onex_change_control",
            linear_retry_max_attempts=6,
            linear_retry_base_delay_seconds=0.0,
        )
    )
    assert client._max_attempts == 6
    assert client._base_delay_seconds == 0.0


@pytest.mark.unit
def test_error_detail_is_empty_for_an_injected_double() -> None:
    """A test double has no HTTP layer, so the reason reads as it always did."""

    class _Double:
        last_error = "should never be read"

    assert sweep_mod._linear_error_detail(_Double()) == ""
    client = _LinearClient(api_key="k")
    client.last_error = "Linear API returned HTTP 429."
    assert sweep_mod._linear_error_detail(client) == " (Linear API returned HTTP 429.)"
