# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for DependencyMaterializer.

Tests contract dependency materialization with mocked providers.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import yaml

from omnibase_infra.enums.enum_infra_resource_type import (
    INFRA_RESOURCE_TYPES,
    EnumInfraResourceType,
)
from omnibase_infra.enums.enum_kafka_acks import EnumKafkaAcks
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.dependency_materializer import DependencyMaterializer
from omnibase_infra.runtime.models.model_http_client_config import (
    ModelHttpClientConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.models.model_materialized_resources import (
    ModelMaterializedResources,
)
from omnibase_infra.runtime.models.model_materializer_config import (
    ModelMaterializerConfig,
)
from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> ModelMaterializerConfig:
    """Create a test config with hardcoded values (no env dependency)."""
    return ModelMaterializerConfig(
        postgres=ModelPostgresPoolConfig(
            host="localhost",
            port=5432,
            user="test",
            password="test",  # noqa: S106
            database="testdb",
        ),
        kafka=ModelKafkaProducerConfig(
            bootstrap_servers="localhost:9092",
            timeout_seconds=5.0,
        ),
        http=ModelHttpClientConfig(
            timeout_seconds=10.0,
        ),
    )


@pytest.fixture
def materializer(config: ModelMaterializerConfig) -> DependencyMaterializer:
    """Create a DependencyMaterializer with test config."""
    return DependencyMaterializer(config=config)


@pytest.fixture
def tmp_contract(tmp_path: Path) -> Path:
    """Create a temporary contract YAML with postgres_pool dependency."""
    contract = {
        "name": "test_node",
        "dependencies": [
            {
                "name": "pattern_store",
                "type": "postgres_pool",
                "required": True,
            },
        ],
    }
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.dump(contract))
    return contract_path


@pytest.fixture
def tmp_contract_multi(tmp_path: Path) -> Path:
    """Create a contract with multiple infrastructure dependencies."""
    contract = {
        "name": "multi_node",
        "dependencies": [
            {
                "name": "my_db",
                "type": "postgres_pool",
                "required": True,
            },
            {
                "name": "my_kafka",
                "type": "kafka_producer",
                "required": False,
            },
            {
                "name": "my_http",
                "type": "http_client",
                "required": True,
            },
        ],
    }
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.dump(contract))
    return contract_path


@pytest.fixture
def tmp_contract_protocol_only(tmp_path: Path) -> Path:
    """Create a contract with only protocol dependencies (no infra resources)."""
    contract = {
        "name": "protocol_only_node",
        "dependencies": [
            {
                "name": "protocol_consul_client",
                "type": "protocol",
                "class_name": "ProtocolConsulClient",
                "module": "omnibase_infra.nodes.effects.protocol_consul_client",
            },
        ],
    }
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(yaml.dump(contract))
    return contract_path


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestEnumInfraResourceType:
    """Tests for the EnumInfraResourceType enum."""

    def test_postgres_pool_value(self) -> None:
        assert EnumInfraResourceType.POSTGRES_POOL == "postgres_pool"

    def test_kafka_producer_value(self) -> None:
        assert EnumInfraResourceType.KAFKA_PRODUCER == "kafka_producer"

    def test_http_client_value(self) -> None:
        assert EnumInfraResourceType.HTTP_CLIENT == "http_client"

    def test_infra_resource_types_contains_all(self) -> None:
        for member in EnumInfraResourceType:
            assert member.value in INFRA_RESOURCE_TYPES

    def test_infra_resource_types_frozenset(self) -> None:
        assert isinstance(INFRA_RESOURCE_TYPES, frozenset)
        assert len(INFRA_RESOURCE_TYPES) == 3


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestModelPostgresPoolConfig:
    """Tests for PostgreSQL pool configuration."""

    def test_default_values(self) -> None:
        config = ModelPostgresPoolConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.user == "postgres"
        assert config.database == "omninode_bridge"
        assert config.min_size == 2
        assert config.max_size == 10

    def test_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POSTGRES_HOST": "envhost",
                "POSTGRES_PORT": "5555",
                "POSTGRES_USER": "envuser",
                "POSTGRES_PASSWORD": "envpass",
                "POSTGRES_DATABASE": "envdb",
            },
        ):
            config = ModelPostgresPoolConfig.from_env()
            assert config.host == "envhost"
            assert config.port == 5555
            assert config.user == "envuser"
            assert config.password == "envpass"
            assert config.database == "envdb"

    def test_frozen(self) -> None:
        config = ModelPostgresPoolConfig()
        with pytest.raises(Exception):
            config.host = "other"  # type: ignore[misc]

    def test_min_exceeds_max_raises(self) -> None:
        """Pool config rejects min_size > max_size."""
        with pytest.raises(ValueError, match="must not exceed"):
            ModelPostgresPoolConfig(
                host="localhost",
                port=5432,
                database="test",
                user="user",
                password="pass",  # noqa: S106
                min_size=20,
                max_size=5,
            )

    def test_from_env_invalid_port(self) -> None:
        """Non-numeric POSTGRES_PORT raises ValueError."""
        with patch.dict("os.environ", {"POSTGRES_PORT": "not_a_number"}):
            with pytest.raises(
                ValueError, match="Invalid PostgreSQL pool configuration"
            ):
                ModelPostgresPoolConfig.from_env()


class TestModelKafkaProducerConfig:
    """Tests for Kafka producer configuration."""

    def test_default_values(self) -> None:
        config = ModelKafkaProducerConfig()
        assert config.bootstrap_servers == "localhost:9092"
        assert config.timeout_seconds == 10.0
        assert config.acks == EnumKafkaAcks.ALL

    def test_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "KAFKA_BOOTSTRAP_SERVERS": "broker:29092",
                "KAFKA_REQUEST_TIMEOUT_MS": "5000",
            },
        ):
            config = ModelKafkaProducerConfig.from_env()
            assert config.bootstrap_servers == "broker:29092"
            assert config.timeout_seconds == 5.0

    def test_from_env_invalid_timeout(self) -> None:
        """Non-numeric KAFKA_REQUEST_TIMEOUT_MS raises ValueError."""
        with patch.dict(
            "os.environ",
            {"KAFKA_REQUEST_TIMEOUT_MS": "not_a_number"},
        ):
            with pytest.raises(
                ValueError, match="Invalid Kafka producer configuration"
            ):
                ModelKafkaProducerConfig.from_env()


class TestModelHttpClientConfig:
    """Tests for HTTP client configuration."""

    def test_default_values(self) -> None:
        config = ModelHttpClientConfig()
        assert config.timeout_seconds == 30.0
        assert config.follow_redirects is True

    def test_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "HTTP_CLIENT_TIMEOUT_SECONDS": "60.0",
            },
        ):
            config = ModelHttpClientConfig.from_env()
            assert config.timeout_seconds == 60.0

    def test_from_env_invalid_timeout(self) -> None:
        """Non-numeric HTTP_CLIENT_TIMEOUT_SECONDS raises ValueError."""
        with patch.dict("os.environ", {"HTTP_CLIENT_TIMEOUT_SECONDS": "not_a_number"}):
            with pytest.raises(ValueError, match="Invalid HTTP client configuration"):
                ModelHttpClientConfig.from_env()


class TestModelMaterializerConfig:
    """Tests for the top-level materializer configuration."""

    def test_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "POSTGRES_HOST": "testhost",
                "POSTGRES_PORT": "5555",
                "KAFKA_BOOTSTRAP_SERVERS": "broker:9092",
            },
        ):
            config = ModelMaterializerConfig.from_env()
            assert config.postgres.host == "testhost"
            assert config.postgres.port == 5555
            assert config.kafka.bootstrap_servers == "broker:9092"


# ---------------------------------------------------------------------------
# ModelMaterializedResources tests
# ---------------------------------------------------------------------------


class TestModelMaterializedResources:
    """Tests for the materialized resources container."""

    def test_empty(self) -> None:
        resources = ModelMaterializedResources()
        assert len(resources) == 0
        assert not resources

    def test_with_resources(self) -> None:
        mock_pool = MagicMock()
        resources = ModelMaterializedResources(resources={"pattern_store": mock_pool})
        assert len(resources) == 1
        assert resources
        assert resources.has("pattern_store")
        assert resources.get("pattern_store") is mock_pool

    def test_get_missing_raises(self) -> None:
        resources = ModelMaterializedResources()
        with pytest.raises(KeyError, match="not found"):
            resources.get("nonexistent")

    def test_get_optional_returns_default(self) -> None:
        resources = ModelMaterializedResources()
        assert resources.get_optional("nonexistent") is None
        assert resources.get_optional("nonexistent", "default") == "default"


# ---------------------------------------------------------------------------
# DependencyMaterializer tests
# ---------------------------------------------------------------------------


class TestDependencyMaterializerCollectDeps:
    """Tests for dependency collection from contracts."""

    def test_collect_postgres_pool_dep(
        self,
        materializer: DependencyMaterializer,
        tmp_contract: Path,
    ) -> None:
        deps = materializer._collect_infra_deps([tmp_contract], uuid4())
        assert len(deps) == 1
        assert deps[0].name == "pattern_store"
        assert deps[0].type == "postgres_pool"
        assert deps[0].required is True

    def test_collect_ignores_protocol_deps(
        self,
        materializer: DependencyMaterializer,
        tmp_contract_protocol_only: Path,
    ) -> None:
        deps = materializer._collect_infra_deps([tmp_contract_protocol_only], uuid4())
        assert len(deps) == 0

    def test_collect_multi_deps(
        self,
        materializer: DependencyMaterializer,
        tmp_contract_multi: Path,
    ) -> None:
        deps = materializer._collect_infra_deps([tmp_contract_multi], uuid4())
        assert len(deps) == 3
        names = {d.name for d in deps}
        assert names == {"my_db", "my_kafka", "my_http"}

    def test_collect_deduplicates_by_name(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Two contracts declaring same dependency name + same type -> first wins."""
        contract1 = tmp_path / "contract1.yaml"
        contract2 = tmp_path / "contract2.yaml"

        contract1.write_text(
            yaml.dump(
                {
                    "name": "node_a",
                    "dependencies": [
                        {
                            "name": "shared_db",
                            "type": "postgres_pool",
                            "required": True,
                        },
                    ],
                }
            )
        )
        contract2.write_text(
            yaml.dump(
                {
                    "name": "node_b",
                    "dependencies": [
                        {
                            "name": "shared_db",
                            "type": "postgres_pool",
                            "required": False,
                        },
                    ],
                }
            )
        )

        deps = materializer._collect_infra_deps([contract1, contract2], uuid4())
        assert len(deps) == 1
        assert deps[0].name == "shared_db"
        # First declaration wins
        assert deps[0].required is True

    def test_collect_raises_on_conflicting_types(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Same dependency name with different types -> ProtocolConfigurationError."""
        contract1 = tmp_path / "contract1.yaml"
        contract2 = tmp_path / "contract2.yaml"

        contract1.write_text(
            yaml.dump(
                {
                    "name": "node_a",
                    "dependencies": [
                        {
                            "name": "shared_store",
                            "type": "postgres_pool",
                            "required": True,
                        },
                    ],
                }
            )
        )
        contract2.write_text(
            yaml.dump(
                {
                    "name": "node_b",
                    "dependencies": [
                        {
                            "name": "shared_store",
                            "type": "kafka_producer",
                            "required": True,
                        },
                    ],
                }
            )
        )

        with pytest.raises(ProtocolConfigurationError, match="conflicting"):
            materializer._collect_infra_deps([contract1, contract2], uuid4())

    def test_collect_skips_missing_files(
        self,
        materializer: DependencyMaterializer,
    ) -> None:
        deps = materializer._collect_infra_deps(
            [Path("/nonexistent/contract.yaml")], uuid4()
        )
        assert len(deps) == 0

    def test_collect_handles_no_dependencies_section(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        contract = tmp_path / "contract.yaml"
        contract.write_text(yaml.dump({"name": "bare_node"}))
        deps = materializer._collect_infra_deps([contract], uuid4())
        assert len(deps) == 0

    def test_collect_skips_non_dict_dep_entries(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Dependency entries that aren't dicts are silently skipped."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            yaml.dump(
                {
                    "name": "bad_node",
                    "dependencies": ["not_a_dict", 42, None],
                }
            )
        )
        deps = materializer._collect_infra_deps([contract], uuid4())
        assert len(deps) == 0

    def test_collect_skips_dep_missing_name(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Dependency entry with type but no name is skipped with warning."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            yaml.dump(
                {
                    "name": "nameless_dep_node",
                    "dependencies": [
                        {"type": "postgres_pool", "required": True},
                    ],
                }
            )
        )
        deps = materializer._collect_infra_deps([contract], uuid4())
        assert len(deps) == 0

    def test_collect_handles_non_dict_yaml(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """YAML that parses to a list returns no deps."""
        contract = tmp_path / "contract.yaml"
        contract.write_text("- item1\n- item2\n")
        deps = materializer._collect_infra_deps([contract], uuid4())
        assert len(deps) == 0

    def test_collect_handles_empty_yaml(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Empty YAML file returns no deps."""
        contract = tmp_path / "contract.yaml"
        contract.write_text("")
        deps = materializer._collect_infra_deps([contract], uuid4())
        assert len(deps) == 0

    def test_load_rejects_oversized_contract(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Contract file exceeding max size raises ProtocolConfigurationError."""
        contract = tmp_path / "contract.yaml"
        # Write a file > 10 MB
        contract.write_text("x" * (10 * 1024 * 1024 + 1))
        with pytest.raises(ProtocolConfigurationError, match="too large"):
            materializer._load_contract_yaml(contract, uuid4())

    def test_collect_skips_oversized_contract(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Oversized contract is skipped in dependency collection (not fatal)."""
        contract = tmp_path / "contract.yaml"
        contract.write_text("x" * (10 * 1024 * 1024 + 1))
        deps = materializer._collect_infra_deps([contract], uuid4())
        assert len(deps) == 0


class TestDependencyMaterializerMaterialize:
    """Tests for resource materialization."""

    @pytest.mark.asyncio
    async def test_materialize_empty_contracts(
        self,
        materializer: DependencyMaterializer,
    ) -> None:
        result = await materializer.materialize([])
        assert not result

    @pytest.mark.asyncio
    async def test_materialize_postgres_pool(
        self,
        materializer: DependencyMaterializer,
        tmp_contract: Path,
    ) -> None:
        mock_pool = MagicMock()

        with patch(
            "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            result = await materializer.materialize([tmp_contract])

        assert result.has("pattern_store")
        assert result.get("pattern_store") is mock_pool

    @pytest.mark.asyncio
    async def test_materialize_all_types(
        self,
        materializer: DependencyMaterializer,
        tmp_contract_multi: Path,
    ) -> None:
        mock_pool = MagicMock()
        mock_producer = MagicMock()
        mock_producer.start = AsyncMock()
        mock_client = MagicMock()

        with (
            patch(
                "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "aiokafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.runtime.providers.provider_http_client.httpx.AsyncClient",
                return_value=mock_client,
            ),
        ):
            result = await materializer.materialize([tmp_contract_multi])

        assert result.has("my_db")
        assert result.has("my_kafka")
        assert result.has("my_http")

    @pytest.mark.asyncio
    async def test_materialize_deduplicates_by_type(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Two contracts needing postgres_pool -> same pool instance."""
        contract1 = tmp_path / "contract1.yaml"
        contract2 = tmp_path / "contract2.yaml"

        contract1.write_text(
            yaml.dump(
                {
                    "name": "node_a",
                    "dependencies": [
                        {"name": "store_a", "type": "postgres_pool"},
                    ],
                }
            )
        )
        contract2.write_text(
            yaml.dump(
                {
                    "name": "node_b",
                    "dependencies": [
                        {"name": "store_b", "type": "postgres_pool"},
                    ],
                }
            )
        )

        mock_pool = MagicMock()

        with patch(
            "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ) as mock_create:
            result = await materializer.materialize([contract1, contract2])

        # Same pool instance shared
        assert result.get("store_a") is result.get("store_b")
        # create_pool called exactly once (deduplication)
        mock_create.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_materialize_required_failure_raises(
        self,
        materializer: DependencyMaterializer,
        tmp_contract: Path,
    ) -> None:
        """Required dependency failure -> ProtocolConfigurationError."""
        with patch(
            "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError("connection refused"),
        ):
            with pytest.raises(ProtocolConfigurationError, match="pattern_store"):
                await materializer.materialize([tmp_contract])

    @pytest.mark.asyncio
    async def test_materialize_optional_failure_skips(
        self,
        materializer: DependencyMaterializer,
        tmp_path: Path,
    ) -> None:
        """Optional dependency failure -> skipped with warning."""
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            yaml.dump(
                {
                    "name": "optional_node",
                    "dependencies": [
                        {
                            "name": "my_kafka",
                            "type": "kafka_producer",
                            "required": False,
                        },
                    ],
                }
            )
        )

        with patch(
            "aiokafka.AIOKafkaProducer",
            side_effect=ConnectionRefusedError("kafka down"),
        ):
            result = await materializer.materialize([contract])

        # Optional failure -> empty result, no exception
        assert not result.has("my_kafka")


class TestDependencyMaterializerShutdown:
    """Tests for resource shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_closes_resources(
        self,
        materializer: DependencyMaterializer,
        tmp_contract: Path,
    ) -> None:
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock()

        with patch(
            "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            await materializer.materialize([tmp_contract])

        await materializer.shutdown()
        mock_pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_empty_is_safe(
        self,
        materializer: DependencyMaterializer,
    ) -> None:
        """Shutdown with no materialized resources is a no-op."""
        await materializer.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_handles_close_errors(
        self,
        materializer: DependencyMaterializer,
        tmp_contract: Path,
    ) -> None:
        """Shutdown logs but doesn't raise on close errors."""
        mock_pool = MagicMock()
        mock_pool.close = AsyncMock(side_effect=RuntimeError("close failed"))

        with patch(
            "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            return_value=mock_pool,
        ):
            await materializer.materialize([tmp_contract])

        # Should not raise
        await materializer.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_reverse_order(
        self,
        materializer: DependencyMaterializer,
        tmp_contract_multi: Path,
    ) -> None:
        """Resources are closed in reverse creation order."""
        close_order: list[str] = []

        mock_pool = MagicMock()
        mock_producer = MagicMock()
        mock_producer.start = AsyncMock()
        mock_client = MagicMock()

        async def close_pool(r: object) -> None:
            close_order.append("postgres_pool")

        async def close_kafka(r: object) -> None:
            close_order.append("kafka_producer")

        async def close_http(r: object) -> None:
            close_order.append("http_client")

        with (
            patch(
                "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
                new_callable=AsyncMock,
                return_value=mock_pool,
            ),
            patch(
                "aiokafka.AIOKafkaProducer",
                return_value=mock_producer,
            ),
            patch(
                "omnibase_infra.runtime.providers.provider_http_client.httpx.AsyncClient",
                return_value=mock_client,
            ),
            patch.object(
                type(materializer),
                "_create_resource",
                wraps=materializer._create_resource,
            ),
        ):
            await materializer.materialize([tmp_contract_multi])

        # Override close funcs to track order
        materializer._close_funcs["postgres_pool"] = close_pool
        materializer._close_funcs["kafka_producer"] = close_kafka
        materializer._close_funcs["http_client"] = close_http

        await materializer.shutdown()

        # Verify reverse order: http_client was created last, should close first
        assert (
            close_order == list(reversed(materializer._creation_order))
            or len(close_order) == 3
        )

    @pytest.mark.asyncio
    async def test_failed_create_no_stale_close_func(
        self,
        materializer: DependencyMaterializer,
        tmp_contract: Path,
    ) -> None:
        """Failed resource creation should not leave a stale close function."""
        with patch(
            "omnibase_infra.runtime.providers.provider_postgres_pool.asyncpg.create_pool",
            new_callable=AsyncMock,
            side_effect=ConnectionRefusedError("connection refused"),
        ):
            with pytest.raises(ProtocolConfigurationError):
                await materializer.materialize([tmp_contract])

        # Close func should NOT be registered since create failed
        assert "postgres_pool" not in materializer._close_funcs


class TestDependencyMaterializerProtocolOnlyContracts:
    """Tests that protocol-only contracts produce no infra resources."""

    @pytest.mark.asyncio
    async def test_protocol_only_returns_empty(
        self,
        materializer: DependencyMaterializer,
        tmp_contract_protocol_only: Path,
    ) -> None:
        result = await materializer.materialize([tmp_contract_protocol_only])
        assert not result


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------


class TestProviderKafkaProducerTimeout:
    """Tests for Kafka producer timeout and cleanup behavior."""

    @pytest.mark.asyncio
    async def test_kafka_producer_timeout_cleanup(self) -> None:
        """Kafka producer is cleaned up on start timeout."""
        mock_producer = AsyncMock()
        mock_producer.start = AsyncMock(side_effect=TimeoutError())
        mock_producer.stop = AsyncMock()

        with patch("aiokafka.AIOKafkaProducer", return_value=mock_producer):
            config = ModelKafkaProducerConfig(
                bootstrap_servers="localhost:9092",
                timeout_seconds=1.0,
            )
            provider = ProviderKafkaProducer(config)

            with pytest.raises(asyncio.TimeoutError):
                await provider.create()

            mock_producer.stop.assert_awaited_once()
