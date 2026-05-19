# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for ``FakeValkeyClient`` (OMN-9265).

Verifies interface compliance, happy-path get/set/ping, error injection,
closed-state behaviour, and counter tracking.
"""

from __future__ import annotations

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from omnibase_infra.test_utils.fake_valkey_client import FakeValkeyClient

# ---------------------------------------------------------------------------
# ping
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ping_returns_true_when_open() -> None:
    client = FakeValkeyClient()
    assert await client.ping() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ping_raises_when_closed() -> None:
    client = FakeValkeyClient(closed=True)
    with pytest.raises(RedisConnectionError):
        await client.ping()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ping_raises_injected_error() -> None:
    err = RedisConnectionError("forced ping failure")
    client = FakeValkeyClient(ping_error=err)
    with pytest.raises(RedisConnectionError, match="forced ping failure"):
        await client.ping()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ping_raises_redis_timeout_error() -> None:
    err = RedisTimeoutError("ping timeout")
    client = FakeValkeyClient(ping_error=err)
    with pytest.raises(RedisTimeoutError):
        await client.ping()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ping_raises_generic_redis_error() -> None:
    err = RedisError("unexpected")
    client = FakeValkeyClient(ping_error=err)
    with pytest.raises(RedisError):
        await client.ping()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ping_increments_call_count() -> None:
    client = FakeValkeyClient()
    await client.ping()
    await client.ping()
    assert client.ping_call_count == 2


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_none_for_missing_key() -> None:
    client = FakeValkeyClient()
    assert await client.get("missing") is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_preloaded_value() -> None:
    client = FakeValkeyClient(initial={"seq": "42"})
    assert await client.get("seq") == "42"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_returns_value_written_by_set() -> None:
    client = FakeValkeyClient()
    await client.set("key", "value")
    assert await client.get("key") == "value"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_raises_injected_error() -> None:
    err = RedisConnectionError("get failed")
    client = FakeValkeyClient(get_error=err)
    with pytest.raises(RedisConnectionError, match="get failed"):
        await client.get("any")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_raises_asyncio_timeout() -> None:
    client = FakeValkeyClient(get_error=TimeoutError("timed out"))
    with pytest.raises(TimeoutError):
        await client.get("any")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_increments_call_count() -> None:
    client = FakeValkeyClient()
    await client.get("a")
    await client.get("b")
    assert client.get_call_count == 2


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_returns_true_on_success() -> None:
    client = FakeValkeyClient()
    assert await client.set("k", "v") is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_stores_value() -> None:
    client = FakeValkeyClient()
    await client.set("seq", "100")
    assert client.store["seq"] == "100"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_overwrites_existing_value() -> None:
    client = FakeValkeyClient(initial={"seq": "0"})
    await client.set("seq", "99")
    assert client.store["seq"] == "99"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_raises_injected_error() -> None:
    err = RedisConnectionError("set failed")
    client = FakeValkeyClient(set_error=err)
    with pytest.raises(RedisConnectionError, match="set failed"):
        await client.set("k", "v")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_raises_redis_timeout_error() -> None:
    client = FakeValkeyClient(set_error=RedisTimeoutError("timeout"))
    with pytest.raises(RedisTimeoutError):
        await client.set("k", "v")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_raises_generic_redis_error() -> None:
    client = FakeValkeyClient(set_error=RedisError("oops"))
    with pytest.raises(RedisError):
        await client.set("k", "v")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_set_increments_call_count() -> None:
    client = FakeValkeyClient()
    await client.set("a", "1")
    await client.set("b", "2")
    assert client.set_call_count == 2


# ---------------------------------------------------------------------------
# aclose
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aclose_marks_client_closed() -> None:
    client = FakeValkeyClient()
    await client.aclose()
    assert client._closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aclose_idempotent() -> None:
    client = FakeValkeyClient()
    await client.aclose()
    await client.aclose()
    assert client.aclose_call_count == 2
    assert client._closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ping_raises_after_aclose() -> None:
    client = FakeValkeyClient()
    await client.aclose()
    with pytest.raises(RedisConnectionError):
        await client.ping()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aclose_increments_call_count() -> None:
    client = FakeValkeyClient()
    await client.aclose()
    assert client.aclose_call_count == 1


# ---------------------------------------------------------------------------
# initial data isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_initial_data_is_copied_not_referenced() -> None:
    source = {"key": "original"}
    client = FakeValkeyClient(initial=source)
    source["key"] = "mutated"
    assert await client.get("key") == "original"


# ---------------------------------------------------------------------------
# RuntimeScheduler integration smoke
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_scheduler_sequence_round_trip() -> None:
    """Simulate the sequence number persist/load cycle used by RuntimeScheduler."""
    client = FakeValkeyClient()
    seq_key = "scheduler:test:seq"

    # Simulate RuntimeScheduler persisting sequence number
    assert client.store.get(seq_key) is None
    await client.set(seq_key, "42")

    # Simulate RuntimeScheduler loading sequence number on restart
    raw = await client.get(seq_key)
    assert raw == "42"
    assert int(raw) == 42
