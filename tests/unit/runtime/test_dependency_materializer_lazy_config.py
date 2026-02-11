# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Regression tests: DependencyMaterializer lazy config resolution.

These tests prove that ``DependencyMaterializer`` does NOT require
``OMNIBASE_INFRA_DB_URL`` (or any other provider env var) at construction
time.  The env var is only validated when a contract actually declares a
``postgres_pool`` dependency and materialization attempts to create the
pool.

Regression context (OMN-2065, PR #290):
    PR #290 made ``ModelPostgresPoolConfig.from_env()`` fail-fast when
    ``OMNIBASE_INFRA_DB_URL`` is absent.  A companion change makes
    ``DependencyMaterializer`` resolve provider configs lazily -- only
    when the specific resource type is requested during ``materialize()``.
    These tests lock in that lazy behavior so nobody refactors it back
    to eager init.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.dependency_materializer import DependencyMaterializer
from omnibase_infra.runtime.models.model_materialized_resources import (
    ModelMaterializedResources,
)

# ---------------------------------------------------------------------------
# Environment isolation helper
# ---------------------------------------------------------------------------

# Env vars that provider configs read -- must be unset for isolation.
_PROVIDER_ENV_VARS = (
    "OMNIBASE_INFRA_DB_URL",
    "KAFKA_BOOTSTRAP_SERVERS",
    "POSTGRES_POOL_MIN_SIZE",
    "POSTGRES_POOL_MAX_SIZE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all provider env vars so tests start with a clean slate."""
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


# Override the autouse fixture from tests/unit/conftest.py that patches
# RuntimeHostProcess._materialize_dependencies.  These tests exercise the
# materializer directly and do not touch RuntimeHostProcess.
@pytest.fixture(autouse=True)
def _skip_materialize_dependencies() -> None:
    """No-op override -- let real materialisation run."""
    return


# ---------------------------------------------------------------------------
# Contract YAML helpers
# ---------------------------------------------------------------------------


def _write_contract(
    tmp_path: Path,
    filename: str,
    *,
    dependencies: list[dict[str, Any]] | None = None,
) -> Path:
    """Write a minimal contract YAML and return its path.

    Args:
        tmp_path: Pytest-provided temporary directory.
        filename: Contract filename (e.g., "contract.yaml").
        dependencies: Optional list of dependency dicts.  Omitted entirely
            when ``None`` to test contracts with no ``dependencies`` key.

    Returns:
        Absolute path to the written contract file.
    """
    data: dict[str, Any] = {
        "name": "test_node",
        "node_type": "EFFECT_GENERIC",
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
    }
    if dependencies is not None:
        data["dependencies"] = dependencies

    path = tmp_path / filename
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


# ===========================================================================
# Test 1: Construction succeeds with no env vars
# ===========================================================================


class TestInitWithoutDbUrl:
    """DependencyMaterializer() must construct without OMNIBASE_INFRA_DB_URL."""

    def test_init_without_db_url_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Construction must succeed even when OMNIBASE_INFRA_DB_URL is unset.

        This is the core lazy-config invariant: the materializer delays
        provider-config validation until ``materialize()`` actually needs
        to create a resource of that type.
        """
        # Double-check the var is truly absent (autouse fixture already does
        # this, but be explicit for clarity).
        monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)

        # Must not raise ValueError / any exception.
        materializer = DependencyMaterializer()
        assert materializer is not None


# ===========================================================================
# Test 2: Non-postgres deps succeed without OMNIBASE_INFRA_DB_URL
# ===========================================================================


class TestMaterializeNonPostgresDeps:
    """Contracts with only non-postgres deps must not require DB URL."""

    @pytest.mark.asyncio
    async def test_materialize_no_postgres_deps_without_db_url(
        self,
        tmp_path: Path,
    ) -> None:
        """Materializing an http_client dep must succeed without DB URL.

        We mock ProviderHttpClient.create() to avoid a real HTTP client,
        but the important assertion is that no ValueError about
        OMNIBASE_INFRA_DB_URL is raised.
        """
        contract_path = _write_contract(
            tmp_path,
            "contract.yaml",
            dependencies=[
                {"name": "my_http", "type": "http_client", "required": True},
            ],
        )

        materializer = DependencyMaterializer()

        mock_client = AsyncMock()
        with patch(
            "omnibase_infra.runtime.dependency_materializer.ProviderHttpClient"
        ) as mock_provider_cls:
            mock_provider_cls.return_value.create = AsyncMock(
                return_value=mock_client,
            )
            mock_provider_cls.close = AsyncMock()

            resources = await materializer.materialize([contract_path])

        assert isinstance(resources, ModelMaterializedResources)
        assert resources.has("my_http")

        # Cleanup
        await materializer.shutdown()


# ===========================================================================
# Test 3: Postgres dep WITHOUT OMNIBASE_INFRA_DB_URL fails fast
# ===========================================================================


class TestMaterializePostgresFailsFast:
    """Requesting postgres_pool without OMNIBASE_INFRA_DB_URL must fail."""

    @pytest.mark.asyncio
    async def test_materialize_postgres_dep_without_db_url_fails_fast(
        self,
        tmp_path: Path,
    ) -> None:
        """Materializing a postgres_pool dep must raise when
        OMNIBASE_INFRA_DB_URL is not set, proving fail-fast at use-time.

        The materializer wraps the original ValueError in a
        ProtocolConfigurationError (with a sanitised message), but the
        original ValueError mentioning OMNIBASE_INFRA_DB_URL is preserved
        in the exception cause chain.
        """
        contract_path = _write_contract(
            tmp_path,
            "contract.yaml",
            dependencies=[
                {"name": "test_pool", "type": "postgres_pool", "required": True},
            ],
        )

        materializer = DependencyMaterializer()

        # The materializer calls ModelPostgresPoolConfig.from_env() which
        # raises ValueError.  The materializer's error handling wraps it
        # in ProtocolConfigurationError with the message sanitised.
        with pytest.raises(
            (ValueError, ProtocolConfigurationError),
        ) as exc_info:
            await materializer.materialize([contract_path])

        # The original ValueError must mention the missing env var.
        # It may be the direct exception or chained via __cause__.
        cause_chain_msgs: list[str] = [str(exc_info.value)]
        cause: BaseException | None = exc_info.value.__cause__
        while cause is not None:
            cause_chain_msgs.append(str(cause))
            cause = cause.__cause__

        assert any("OMNIBASE_INFRA_DB_URL" in msg for msg in cause_chain_msgs), (
            f"Expected 'OMNIBASE_INFRA_DB_URL' in exception cause chain, "
            f"got: {cause_chain_msgs}"
        )


# ===========================================================================
# Test 4: Empty contracts without OMNIBASE_INFRA_DB_URL
# ===========================================================================


class TestMaterializeEmptyContracts:
    """Contracts with no dependencies section must return empty resources."""

    @pytest.mark.asyncio
    async def test_materialize_empty_contracts_without_db_url(
        self,
        tmp_path: Path,
    ) -> None:
        """Contracts with no ``dependencies`` key must succeed and return
        an empty ``ModelMaterializedResources``.
        """
        # Contract with no dependencies key at all
        no_deps_path = _write_contract(
            tmp_path,
            "no_deps_contract.yaml",
            dependencies=None,
        )

        # Contract with empty dependencies list
        empty_deps_path = _write_contract(
            tmp_path,
            "empty_deps_contract.yaml",
            dependencies=[],
        )

        materializer = DependencyMaterializer()

        resources = await materializer.materialize([no_deps_path, empty_deps_path])

        assert isinstance(resources, ModelMaterializedResources)
        assert len(resources) == 0
        assert not resources  # __bool__ returns False for empty


__all__: list[str] = []
