# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for ModelPostgresPoolConfig env-fallback behaviour.

Verifies that the model returns sensible defaults when POSTGRES_HOST is absent,
exercising the os.environ.get fallback added in OMN-8605.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)


def test_defaults_without_postgres_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelPostgresPoolConfig should fall back to 'localhost' when POSTGRES_HOST is unset."""
    monkeypatch.delenv("POSTGRES_HOST", raising=False)

    config = ModelPostgresPoolConfig(database="testdb")

    assert config.host == "localhost"
    assert config.port == 5432
    assert config.user == "postgres"
    assert config.database == "testdb"
    assert config.min_size == 2
    assert config.max_size == 10


def test_host_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelPostgresPoolConfig should use POSTGRES_HOST when set."""
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")

    config = ModelPostgresPoolConfig(database="testdb")

    assert config.host == "db.internal"
