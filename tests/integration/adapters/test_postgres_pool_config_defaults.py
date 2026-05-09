# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for ModelPostgresPoolConfig env-var behaviour (OMN-10732).

Verifies fail-fast on missing POSTGRES_HOST and correct env-var resolution.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)


def test_missing_postgres_host_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelPostgresPoolConfig raises KeyError when POSTGRES_HOST is unset."""
    monkeypatch.delenv("POSTGRES_HOST", raising=False)

    with pytest.raises(KeyError):
        ModelPostgresPoolConfig(database="testdb")


def test_host_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelPostgresPoolConfig uses POSTGRES_HOST when set."""
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")

    config = ModelPostgresPoolConfig(database="testdb")

    assert config.host == "db.internal"
    assert config.port == 5432
    assert config.user == "postgres"
    assert config.database == "testdb"
    assert config.min_size == 2
    assert config.max_size == 10
