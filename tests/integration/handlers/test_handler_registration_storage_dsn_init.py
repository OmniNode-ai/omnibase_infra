# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration regression: dsn-constructed handler does not require POSTGRES_HOST.

Context (OMN-10954):
    Runtime boot on .201 crashed with ``KeyError('POSTGRES_HOST')`` inside
    ``HandlerRegistrationStoragePostgres.__init__`` even when the kernel's
    ``node_registration_orchestrator`` plugin constructed the handler with
    ``dsn=OMNIBASE_INFRA_DB_URL``. The dsn-path of ``asyncpg.create_pool``
    does not consult ``host``, so the env requirement was dead weight that
    bricked container startup whenever the compose file shipped only
    ``OMNIBASE_INFRA_DB_URL`` (the canonical seam) and omitted the legacy
    ``POSTGRES_HOST`` var.

    This integration test reproduces the boot wiring: it imports the same
    constructor the plugin uses, calls it with the exact kwargs the plugin
    passes, and asserts the call returns a usable handler with ``POSTGRES_HOST``
    intentionally absent from the environment. No real Postgres pool is
    created — the pool is lazy and only materialized on first request.
"""

from __future__ import annotations

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.handlers.registration_storage.handler_registration_storage_postgres import (
    HandlerRegistrationStoragePostgres,
)


@pytest.mark.integration
def test_plugin_style_dsn_construction_without_postgres_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the boot-time call site that crashed runtime kernel on .201.

    Mirrors ``node_registration_orchestrator.plugin._wire_registration_storage``:
    constructs the handler with only ``container``, ``dsn``, and
    ``auto_create_schema=True``. Asserts that the constructor succeeds with
    ``POSTGRES_HOST`` cleared from the environment.
    """
    monkeypatch.delenv("POSTGRES_HOST", raising=False)

    handler = HandlerRegistrationStoragePostgres(
        container=ModelONEXContainer(),
        dsn="postgresql://postgres:secret@postgres:5432/omnibase_infra",
        auto_create_schema=True,
    )

    assert handler.handler_type == "postgresql"
    assert handler._dsn == "postgresql://postgres:secret@postgres:5432/omnibase_infra"
