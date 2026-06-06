# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for TopicProvisioner.

Tests topic auto-creation logic including:
- Successful topic creation
- Topic already exists (idempotent)
- Connection failure (best-effort, returns unavailable)
- Missing aiokafka (graceful degradation)
- Single topic creation
- Construction with missing contracts_root raises immediately

Related:
    - OMN-1990: Kafka topic auto-creation
    - OMN-5132: Kill fallback paths in TopicProvisioner
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


@pytest.fixture
def contracts_root(tmp_path: Path) -> Path:
    """Create a minimal contracts directory with a contract.yaml for extraction."""
    node_dir = tmp_path / "node_example"
    node_dir.mkdir()
    contract = node_dir / "contract.yaml"
    # Topic must be 5-segment: onex.<kind>.<producer>.<event-name>.<version>
    contract.write_text(
        "name: node_example\n"
        "version: 1.0.0\n"
        "namespace: onex.stamped\n"
        "event_bus:\n"
        "  publish_topics:\n"
        "    - onex.evt.test-producer.example-event.v1\n"
    )
    return tmp_path


def _make_provisioner(
    contracts_root: Path,
    bootstrap_servers: str = "localhost:9092",
) -> TopicProvisioner:
    """Helper to construct a TopicProvisioner with contracts_root."""
    return TopicProvisioner(
        bootstrap_servers=bootstrap_servers,
        contracts_root=contracts_root,
    )


def _topic_spec(manager: TopicProvisioner, suffix: str) -> ModelTopicSpec:
    return next(spec for spec in manager._topic_specs if spec.suffix == suffix)


class TestTopicProvisioner:
    """Tests for TopicProvisioner."""

    def test_init_defaults(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default initialization reads KAFKA_BOOTSTRAP_SERVERS from env."""
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
        manager = TopicProvisioner(contracts_root=contracts_root)
        assert manager._bootstrap_servers == "redpanda:9092"
        assert manager._request_timeout_ms == 30000

    def test_init_custom(self, contracts_root: Path) -> None:
        """Custom bootstrap servers and timeout."""
        manager = TopicProvisioner(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            request_timeout_ms=5000,
            contracts_root=contracts_root,
        )
        assert manager._bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert manager._request_timeout_ms == 5000

    def test_topic_partition_cap_from_env(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Local brokers can cap contract topic partition creation."""
        monkeypatch.setenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", "1")

        manager = _make_provisioner(contracts_root)

        spec = _topic_spec(manager, "onex.evt.test-producer.example-event.v1")
        assert manager._creation_partitions(spec) == 1

    def test_invalid_topic_partition_cap_is_ignored(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Invalid partition caps do not override contract topic specs."""
        monkeypatch.setenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", "nope")

        manager = _make_provisioner(contracts_root)

        spec = _topic_spec(manager, "onex.evt.test-producer.example-event.v1")
        assert manager._creation_partitions(spec) == 6

    def test_empty_topic_partition_cap_is_disabled(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Empty partition cap values leave contract topic specs unchanged."""
        monkeypatch.setenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", "")

        manager = _make_provisioner(contracts_root)

        spec = _topic_spec(manager, "onex.evt.test-producer.example-event.v1")
        assert manager._creation_partitions(spec) == 6

    def test_zero_topic_partition_cap_is_disabled(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Zero partition cap values disable the cap without warnings."""
        monkeypatch.setenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", "0")

        manager = _make_provisioner(contracts_root)

        spec = _topic_spec(manager, "onex.evt.test-producer.example-event.v1")
        assert manager._creation_partitions(spec) == 6
        assert not [
            record
            for record in caplog.records
            if "ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS" in record.message
        ]

    def test_negative_one_topic_partition_cap_is_disabled(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """-1 partition cap values disable the cap without warnings."""
        monkeypatch.setenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", "-1")

        manager = _make_provisioner(contracts_root)

        spec = _topic_spec(manager, "onex.evt.test-producer.example-event.v1")
        assert manager._creation_partitions(spec) == 6
        assert not [
            record
            for record in caplog.records
            if "ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS" in record.message
        ]

    def test_init_missing_contracts_root_raises(self, tmp_path: Path) -> None:
        """Constructing with a non-existent contracts_root raises immediately."""
        bad_path = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="contracts_root"):
            TopicProvisioner(contracts_root=bad_path)

    def test_contract_priority_topics_are_ordered_first(
        self,
        tmp_path: Path,
    ) -> None:
        """Contract-declared provisioning priority controls creation order."""
        node_dir = tmp_path / "node_example"
        node_dir.mkdir()
        (node_dir / "contract.yaml").write_text(
            "name: node_example\n"
            "event_bus:\n"
            "  publish_topics:\n"
            "    - onex.cmd.omnimarket.build-loop-start.v1\n"
            "    - topic: onex.evt.platform.node-introspection.v1\n"
            "      provisioning_priority: 10\n"
            "    - topic: onex.evt.omnibase-infra.runtime-health-check.v1\n"
            "      provisioning_priority: 0\n",
            encoding="utf-8",
        )

        manager = _make_provisioner(tmp_path)

        ordered_suffixes = [spec.suffix for spec in manager._topic_specs]
        health_idx = ordered_suffixes.index(
            "onex.evt.omnibase-infra.runtime-health-check.v1"
        )
        introspection_idx = ordered_suffixes.index(
            "onex.evt.platform.node-introspection.v1"
        )
        default_idx = ordered_suffixes.index("onex.cmd.omnimarket.build-loop-start.v1")

        assert health_idx < introspection_idx < default_idx

    async def test_ensure_provisioned_topics_all_created(
        self, contracts_root: Path
    ) -> None:
        """All provisioned topics are created when none exist."""
        manager = _make_provisioner(contracts_root)

        mock_admin_cls = MagicMock()
        mock_admin_instance = AsyncMock()
        mock_admin_instance.start = AsyncMock()
        mock_admin_instance.close = AsyncMock()
        mock_admin_instance.create_topics = AsyncMock()
        mock_admin_cls.return_value = mock_admin_instance

        with patch.dict(
            "sys.modules",
            {
                "aiokafka": MagicMock(),
                "aiokafka.admin": MagicMock(
                    AIOKafkaAdminClient=mock_admin_cls,
                    NewTopic=MagicMock(),
                ),
                "aiokafka.errors": MagicMock(
                    TopicAlreadyExistsError=type(
                        "TopicAlreadyExistsError", (Exception,), {}
                    ),
                ),
            },
        ):
            result = await manager.ensure_provisioned_topics_exist()

        assert result["status"] == "success"
        assert len(result["created"]) > 0
        assert len(result["failed"]) == 0

    async def test_ensure_provisioned_topics_connection_failure(
        self, contracts_root: Path
    ) -> None:
        """Connection failure returns unavailable status gracefully."""
        manager = _make_provisioner(
            contracts_root, bootstrap_servers="nonexistent:9999"
        )

        # Mock the import to provide the classes but make start() fail
        mock_admin_cls = MagicMock()
        mock_admin_instance = AsyncMock()
        mock_admin_instance.start = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )
        mock_admin_instance.close = AsyncMock()
        mock_admin_cls.return_value = mock_admin_instance

        with patch.dict(
            "sys.modules",
            {
                "aiokafka": MagicMock(),
                "aiokafka.admin": MagicMock(
                    AIOKafkaAdminClient=mock_admin_cls,
                    NewTopic=MagicMock(),
                ),
                "aiokafka.errors": MagicMock(
                    TopicAlreadyExistsError=type(
                        "TopicAlreadyExistsError", (Exception,), {}
                    ),
                ),
            },
        ):
            result = await manager.ensure_provisioned_topics_exist()

        assert result["status"] == "unavailable"
        assert len(result["failed"]) > 0

    async def test_ensure_provisioned_topics_import_error(
        self, contracts_root: Path
    ) -> None:
        """Graceful degradation when aiokafka is not installed."""
        manager = _make_provisioner(contracts_root)

        import importlib

        import omnibase_infra.event_bus.service_topic_manager as mod

        # Force ImportError by temporarily removing aiokafka from sys.modules
        with patch.dict(
            "sys.modules",
            {"aiokafka": None, "aiokafka.admin": None, "aiokafka.errors": None},
        ):
            # Reload so the function-level import sees the patched modules
            importlib.reload(mod)
            reloaded_manager = mod.TopicProvisioner(
                bootstrap_servers="localhost:9092",
                contracts_root=contracts_root,
            )
            result = await reloaded_manager.ensure_provisioned_topics_exist()

        # Restore module for other tests
        importlib.reload(mod)

        assert result["status"] == "unavailable"
        assert len(result["created"]) == 0
        assert len(result["existing"]) == 0

    async def test_ensure_provisioned_topics_already_exist(
        self, contracts_root: Path
    ) -> None:
        """Topics that already exist are counted as 'existing', not 'created'."""
        manager = _make_provisioner(contracts_root)

        topic_already_exists_cls = type("TopicAlreadyExistsError", (Exception,), {})

        mock_admin_cls = MagicMock()
        mock_admin_instance = AsyncMock()
        mock_admin_instance.start = AsyncMock()
        mock_admin_instance.close = AsyncMock()
        mock_admin_instance.create_topics = AsyncMock(
            side_effect=topic_already_exists_cls("Topic exists")
        )
        mock_admin_cls.return_value = mock_admin_instance

        with patch.dict(
            "sys.modules",
            {
                "aiokafka": MagicMock(),
                "aiokafka.admin": MagicMock(
                    AIOKafkaAdminClient=mock_admin_cls,
                    NewTopic=MagicMock(),
                ),
                "aiokafka.errors": MagicMock(
                    TopicAlreadyExistsError=topic_already_exists_cls,
                ),
            },
        ):
            result = await manager.ensure_provisioned_topics_exist()

        assert result["status"] == "success"
        assert len(result["existing"]) > 0
        assert len(result["created"]) == 0
        assert len(result["failed"]) == 0

    async def test_ensure_single_topic_success(
        self,
        contracts_root: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Single topic creation returns True on success."""
        monkeypatch.setenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", "1")
        manager = _make_provisioner(contracts_root)

        mock_admin_cls = MagicMock()
        mock_new_topic_cls = MagicMock()
        mock_admin_instance = AsyncMock()
        mock_admin_instance.start = AsyncMock()
        mock_admin_instance.close = AsyncMock()
        mock_admin_instance.create_topics = AsyncMock()
        mock_admin_cls.return_value = mock_admin_instance

        with patch.dict(
            "sys.modules",
            {
                "aiokafka": MagicMock(),
                "aiokafka.admin": MagicMock(
                    AIOKafkaAdminClient=mock_admin_cls,
                    NewTopic=mock_new_topic_cls,
                ),
                "aiokafka.errors": MagicMock(
                    TopicAlreadyExistsError=type(
                        "TopicAlreadyExistsError", (Exception,), {}
                    ),
                ),
            },
        ):
            result = await manager.ensure_topic_exists("test.topic")

        assert result is True
        mock_new_topic_cls.assert_called_once_with(
            name="test.topic",
            num_partitions=1,
            replication_factor=1,
        )

    async def test_ensure_single_topic_failure(self, contracts_root: Path) -> None:
        """Single topic creation returns False on failure."""
        manager = _make_provisioner(
            contracts_root, bootstrap_servers="nonexistent:9999"
        )

        mock_admin_cls = MagicMock()
        mock_admin_instance = AsyncMock()
        mock_admin_instance.start = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )
        mock_admin_instance.close = AsyncMock()
        mock_admin_cls.return_value = mock_admin_instance

        with patch.dict(
            "sys.modules",
            {
                "aiokafka": MagicMock(),
                "aiokafka.admin": MagicMock(
                    AIOKafkaAdminClient=mock_admin_cls,
                    NewTopic=MagicMock(),
                ),
                "aiokafka.errors": MagicMock(
                    TopicAlreadyExistsError=type(
                        "TopicAlreadyExistsError", (Exception,), {}
                    ),
                ),
            },
        ):
            result = await manager.ensure_topic_exists("test.topic")

        assert result is False
