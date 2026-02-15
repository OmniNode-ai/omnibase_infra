# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for TopicProvisioner.

Tests topic auto-creation logic including:
- Successful topic creation
- Topic already exists (idempotent)
- Connection failure (best-effort, returns unavailable)
- Missing aiokafka (graceful degradation)
- Single topic creation

Related:
    - OMN-1990: Kafka topic auto-creation
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.topics import ALL_PROVISIONED_SUFFIXES

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


class TestTopicProvisioner:
    """Tests for TopicProvisioner."""

    def test_init_defaults(self) -> None:
        """Default initialization uses environment or localhost."""
        manager = TopicProvisioner()
        assert manager._bootstrap_servers is not None
        assert manager._request_timeout_ms == 30000

    def test_init_custom(self) -> None:
        """Custom bootstrap servers and timeout."""
        manager = TopicProvisioner(
            bootstrap_servers="kafka1:9092,kafka2:9092",
            request_timeout_ms=5000,
        )
        assert manager._bootstrap_servers == "kafka1:9092,kafka2:9092"
        assert manager._request_timeout_ms == 5000

    async def test_ensure_platform_topics_all_created(self) -> None:
        """All platform topics are created when none exist."""
        manager = TopicProvisioner(bootstrap_servers="localhost:9092")

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
        assert len(result["created"]) == len(ALL_PROVISIONED_SUFFIXES)
        assert len(result["failed"]) == 0

    async def test_ensure_platform_topics_connection_failure(self) -> None:
        """Connection failure returns unavailable status gracefully."""
        manager = TopicProvisioner(bootstrap_servers="nonexistent:9999")

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
        assert len(result["failed"]) == len(ALL_PROVISIONED_SUFFIXES)

    async def test_ensure_platform_topics_import_error(self) -> None:
        """Graceful degradation when aiokafka is not installed."""
        manager = TopicProvisioner(bootstrap_servers="localhost:9092")

        import importlib

        import omnibase_infra.event_bus.service_topic_manager as mod

        # Force ImportError by temporarily removing aiokafka from sys.modules
        with patch.dict(
            "sys.modules",
            {"aiokafka": None, "aiokafka.admin": None, "aiokafka.errors": None},
        ):
            # Reload so the function-level import sees the patched modules
            importlib.reload(mod)
            reloaded_manager = mod.TopicProvisioner(bootstrap_servers="localhost:9092")
            result = await reloaded_manager.ensure_provisioned_topics_exist()

        # Restore module for other tests
        importlib.reload(mod)

        assert result["status"] == "unavailable"
        assert len(result["failed"]) == len(ALL_PROVISIONED_SUFFIXES)
        assert len(result["created"]) == 0
        assert len(result["existing"]) == 0

    async def test_ensure_platform_topics_already_exist(self) -> None:
        """Topics that already exist are counted as 'existing', not 'created'."""
        manager = TopicProvisioner(bootstrap_servers="localhost:9092")

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
        assert len(result["existing"]) == len(ALL_PROVISIONED_SUFFIXES)
        assert len(result["created"]) == 0
        assert len(result["failed"]) == 0

    async def test_ensure_single_topic_success(self) -> None:
        """Single topic creation returns True on success."""
        manager = TopicProvisioner(bootstrap_servers="localhost:9092")

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
            result = await manager.ensure_topic_exists("test.topic")

        assert result is True

    async def test_ensure_single_topic_failure(self) -> None:
        """Single topic creation returns False on failure."""
        manager = TopicProvisioner(bootstrap_servers="nonexistent:9999")

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
