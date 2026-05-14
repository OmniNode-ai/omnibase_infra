# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression guard: dsn-based handler must not require POSTGRES_HOST env var.

Context:
    Runtime boot crashed with ``KeyError('POSTGRES_HOST')`` inside
    ``HandlerRegistrationStoragePostgres.__init__`` even when constructed
    with ``dsn=...`` set from ``OMNIBASE_INFRA_DB_URL``. The dsn-path of
    ``asyncpg.create_pool`` does not consult ``host``, so requiring it
    from the environment is a wiring bug.

    Fix: skip the ``os.environ["POSTGRES_HOST"]`` lookup when ``dsn`` is
    provided; keep the env requirement only for the host/port construction
    path that asyncpg actually consumes.
"""

from __future__ import annotations

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)


@pytest.mark.unit
def test_dsn_construction_does_not_require_postgres_host_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor with dsn must succeed even when POSTGRES_HOST is unset."""
    monkeypatch.delenv("POSTGRES_HOST", raising=False)

    handler = HandlerRegistrationStoragePostgres(
        container=ModelONEXContainer(),
        dsn="postgresql://postgres:secret@postgres:5432/omnibase_infra",
    )

    assert handler._dsn == "postgresql://postgres:secret@postgres:5432/omnibase_infra"


@pytest.mark.unit
def test_explicit_host_overrides_postgres_host_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit host argument must take precedence over env var."""
    monkeypatch.setenv("POSTGRES_HOST", "env-host")

    handler = HandlerRegistrationStoragePostgres(
        container=ModelONEXContainer(),
        host="explicit-host",
    )

    assert handler._host == "explicit-host"


@pytest.mark.unit
def test_no_dsn_no_host_falls_back_to_postgres_host_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When neither dsn nor host is provided, env var is the source of truth."""
    monkeypatch.setenv("POSTGRES_HOST", "env-host")

    handler = HandlerRegistrationStoragePostgres(container=ModelONEXContainer())

    assert handler._host == "env-host"


@pytest.mark.unit
def test_no_dsn_no_host_no_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without dsn, host, or env var the constructor must fail loudly."""
    monkeypatch.delenv("POSTGRES_HOST", raising=False)

    with pytest.raises(KeyError):
        HandlerRegistrationStoragePostgres(container=ModelONEXContainer())
