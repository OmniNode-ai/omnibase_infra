# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""HandlerRegistrationStoragePostgres takes its settings from the container (OMN-22131).

The handler's constructor accepts only the container and resolves
ModelRegistrationStoragePostgresConfig with ``get_service``. Two container
profiles bind the config: the in-memory profile binds the declared file's
values with a host, and the Redpanda-overlay profile binds the same file with a
DSN and an overlay of pool size, timeout and circuit breaker values. The pool
the handler opens and the waits it applies follow the bound config in each.
"""

from __future__ import annotations

import inspect
from typing import Any
from uuid import uuid4

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_core.enums import EnumInjectionScope
from omnibase_core.errors.error_service_resolution import ServiceResolutionError
from omnibase_infra.handlers.registration_storage import (
    handler_registration_storage_postgres as module,
)
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)
from omnibase_infra.models.registration_storage import (
    DECLARED_SETTINGS_PATH,
    ModelRegistrationStoragePostgresConfig,
)
from omnibase_infra.models.resilience import ModelCircuitBreakerConfig


def _inmemory_profile() -> ModelRegistrationStoragePostgresConfig:
    return ModelRegistrationStoragePostgresConfig.from_file(
        DECLARED_SETTINGS_PATH,
        dsn="",
        host="postgres.inmemory.test",
        port=5436,
        password="",
        auto_create_schema=False,
    )


def _redpanda_overlay_profile() -> ModelRegistrationStoragePostgresConfig:
    base = ModelRegistrationStoragePostgresConfig.from_file(
        DECLARED_SETTINGS_PATH,
        dsn="postgresql://overlay@postgres.redpanda.test:5436/omnibase_infra",
        host="",
        port=5436,
        password="overlay-secret",
        auto_create_schema=True,
    )
    return base.model_copy(
        update={
            "pool_size": 3,
            "timeout_seconds": 7.5,
            "circuit_breaker": ModelCircuitBreakerConfig(
                threshold=2,
                reset_timeout_seconds=11.0,
                service_name="postgres.storage.overlay",
                transport_type=base.circuit_breaker.transport_type,
            ),
        }
    )


PROFILES = {
    "inmemory": _inmemory_profile,
    "redpanda_overlay": _redpanda_overlay_profile,
}


async def _container_with(
    config: ModelRegistrationStoragePostgresConfig,
) -> ModelONEXContainer:
    container = ModelONEXContainer()
    assert container.service_registry is not None
    await container.service_registry.register_instance(
        interface=ModelRegistrationStoragePostgresConfig,
        instance=config,
        scope=EnumInjectionScope.GLOBAL,
        metadata={"description": "registration storage settings (test)"},
    )
    return container


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[str] = []

    async def execute(self, query: str, *_params: Any) -> str:
        self.executed.append(query)
        return "OK"


class _FakeAcquire:
    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _FakePool:
    def __init__(self) -> None:
        self.conn = _FakeConnection()

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)

    async def close(self) -> None:
        return None


@pytest.mark.unit
def test_constructor_accepts_only_the_container() -> None:
    params = list(
        inspect.signature(HandlerRegistrationStoragePostgres.__init__).parameters
    )
    assert params == ["self", "container"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("profile", sorted(PROFILES))
async def test_handler_opens_the_pool_the_bound_config_names(
    profile: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = PROFILES[profile]()
    container = await _container_with(config)
    calls: list[dict[str, Any]] = []
    pool = _FakePool()

    async def _fake_create_pool(**kwargs: Any) -> _FakePool:
        calls.append(kwargs)
        return pool

    monkeypatch.setattr(module.asyncpg, "create_pool", _fake_create_pool)

    handler = HandlerRegistrationStoragePostgres(container)
    opened = await handler._ensure_pool(correlation_id=uuid4())

    assert opened is pool
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["max_size"] == config.pool_size
    if config.dsn.get_secret_value():
        assert kwargs["dsn"] == config.dsn.get_secret_value()
        assert "host" not in kwargs
    else:
        assert kwargs["host"] == config.host
        assert kwargs["port"] == config.port
        assert kwargs["database"] == config.database
        assert kwargs["user"] == config.user
        assert kwargs["password"] is None
    assert bool(pool.conn.executed) is config.auto_create_schema
    assert handler._timeout_seconds == config.timeout_seconds
    assert handler.circuit_breaker_threshold == config.circuit_breaker.threshold
    assert (
        handler.circuit_breaker_reset_timeout
        == config.circuit_breaker.reset_timeout_seconds
    )
    assert handler.service_name == config.circuit_breaker.service_name

    # A second call reuses the pool.
    assert await handler._ensure_pool() is pool
    assert len(calls) == 1


@pytest.mark.unit
def test_declared_file_holds_the_previous_defaults() -> None:
    config = _inmemory_profile()
    assert config.database == "omnibase_infra"
    assert config.user == "postgres"
    assert config.pool_size == 10
    assert config.timeout_seconds == 30.0
    assert config.circuit_breaker.threshold == 5
    assert config.circuit_breaker.reset_timeout_seconds == 60.0
    assert config.circuit_breaker.service_name == "postgres.storage"


@pytest.mark.unit
def test_config_without_dsn_or_host_fails_to_load() -> None:
    with pytest.raises(ValueError, match="dsn or a host"):
        ModelRegistrationStoragePostgresConfig.from_file(
            DECLARED_SETTINGS_PATH,
            dsn="",
            host="",
            port=5436,
            password="",
            auto_create_schema=False,
        )


@pytest.mark.unit
def test_handler_refuses_a_container_without_the_config() -> None:
    with pytest.raises(
        ServiceResolutionError, match="ModelRegistrationStoragePostgresConfig"
    ):
        HandlerRegistrationStoragePostgres(ModelONEXContainer())
