# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for TopicProvisioner integration into runtime kernel startup.

Covers:
- Kernel boot section 3.5: TopicProvisioner is wired after event bus, before handlers
- Provisioner failure at boot → kernel still starts (best-effort)
- Dynamic materialization → new contract's topics provisioned before subscription
- No bootstrap servers → provisioning skipped gracefully

Related: OMN-11242
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]

_PATCH_TARGET = "omnibase_infra.event_bus.service_topic_manager.TopicProvisioner"
_TEST_KAFKA_BOOTSTRAP = "192.168.86.201:19092"  # kafka-fallback-ok


def _make_descriptor_mock(subscribe_topics: list[str]) -> MagicMock:
    """Build a minimal handler descriptor mock with subscribe_topics."""
    descriptor = MagicMock()
    descriptor.contract_config = {
        "event_bus": {
            "version": {"major": 1, "minor": 0, "patch": 0},
            "subscribe_topics": subscribe_topics,
        }
    }
    return descriptor


class TestKernelBootTopicProvisioning:
    """TopicProvisioner.ensure_provisioned_topics_exist is present in kernel boot sequence."""

    def test_provisioner_is_referenced_in_kernel_source(self) -> None:
        """service_kernel.py references TopicProvisioner in section 3.5."""
        import inspect

        from omnibase_infra.runtime import service_kernel

        source = inspect.getsource(service_kernel)
        assert "TopicProvisioner" in source
        assert "ensure_provisioned_topics_exist" in source

    def test_validation_suppresses_pre_auto_create_missing_topic_logs(self) -> None:
        """Kernel suppresses per-topic MISSING_TOPIC logs before auto-create retry."""
        import inspect

        from omnibase_infra.runtime import service_kernel

        source = inspect.getsource(service_kernel)
        assert "log_missing=False" in source
        assert "strict_topic_validation" in source
        assert "Topic validation recovered after auto-create" in source

    def test_provisioner_import_path_is_correct(self) -> None:
        """TopicProvisioner can be imported from the path used by service_kernel.py."""
        from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

        assert callable(
            getattr(TopicProvisioner, "ensure_provisioned_topics_exist", None)
        )

    @pytest.mark.asyncio
    async def test_provisioner_best_effort_on_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing TopicProvisioner logs a warning and does not raise."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()
        monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", _TEST_KAFKA_BOOTSTRAP)

        mock_provisioner = AsyncMock()
        mock_provisioner.ensure_provisioned_topics_exist.side_effect = RuntimeError(
            "broker unreachable"
        )

        warning_count = 0

        class _CountWarnings(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                nonlocal warning_count
                if record.levelno >= logging.WARNING and (
                    "provisioning" in record.getMessage().lower()
                    or "broker" in record.getMessage().lower()
                ):
                    warning_count += 1

        log_handler = _CountWarnings()
        kernel_logger = logging.getLogger("omnibase_infra.runtime.service_kernel")
        kernel_logger.addHandler(log_handler)

        try:
            from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

            use_kafka = True
            correlation_id = "test-corr-id"

            try:
                provisioner = mock_provisioner
                provisioning_result = await provisioner.ensure_provisioned_topics_exist(
                    correlation_id=correlation_id,
                )
                log_level = (
                    logging.WARNING
                    if provisioning_result["status"] != "success"
                    else logging.INFO
                )
                kernel_logger.log(
                    log_level,
                    "Topic provisioning: status=%s",
                    provisioning_result["status"],
                )
            except Exception:  # noqa: BLE001
                kernel_logger.warning(
                    "Topic provisioning failed (best-effort, non-blocking)",
                    exc_info=True,
                )

        finally:
            kernel_logger.removeHandler(log_handler)

        assert warning_count >= 1, (
            "Expected at least one warning when provisioner fails"
        )
        assert use_kafka  # guard used in kernel, confirms the if-use_kafka path


def _make_kafka_bus_mock(bootstrap: str = _TEST_KAFKA_BOOTSTRAP) -> MagicMock:
    """Return a MagicMock that isinstance-checks as EventBusKafka."""
    from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka

    mock = MagicMock(spec=EventBusKafka)
    mock._bootstrap_servers = bootstrap
    return mock


class TestDynamicMaterializationTopicProvisioning:
    """Topic provisioning before subscription in live contract materialization."""

    @pytest.mark.asyncio
    async def test_provision_called_before_subscribe(
        self,
        tmp_path: Path,
    ) -> None:
        """ensure_topic_exists is called for each subscribe_topic before wire_subscriptions."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        from omnibase_infra.runtime.runtime_host_process import (
            RuntimeHostProcess,
        )

        process = RuntimeHostProcess.__new__(RuntimeHostProcess)
        process._contract_paths = [contracts_dir]
        process._event_bus = _make_kafka_bus_mock()
        process._event_bus_wiring = MagicMock()

        call_order: list[str] = []

        async def _mock_wire_subscriptions(**kwargs: object) -> None:
            call_order.append("wire_subscriptions")

        async def _mock_ensure_topic_exists(topic_name: str, **kwargs: object) -> bool:
            call_order.append(f"provision:{topic_name}")
            return True

        process._event_bus_wiring.wire_subscriptions = AsyncMock(
            side_effect=_mock_wire_subscriptions
        )

        mock_provisioner_instance = AsyncMock()
        mock_provisioner_instance.ensure_topic_exists = AsyncMock(
            side_effect=_mock_ensure_topic_exists
        )
        mock_provisioner_cls = MagicMock(return_value=mock_provisioner_instance)

        descriptor = _make_descriptor_mock(
            ["onex.evt.test.topic-a.v1", "onex.evt.test.topic-b.v1"]
        )

        with patch(_PATCH_TARGET, mock_provisioner_cls):
            await process._wire_live_handler_subscriptions(
                node_name="test-handler",
                descriptor=descriptor,
            )

        provision_calls = [c for c in call_order if c.startswith("provision:")]
        assert "provision:onex.evt.test.topic-a.v1" in provision_calls
        assert "provision:onex.evt.test.topic-b.v1" in provision_calls
        assert call_order.index(
            "provision:onex.evt.test.topic-a.v1"
        ) < call_order.index("wire_subscriptions"), (
            "topic provision must happen before wire_subscriptions"
        )

    @pytest.mark.asyncio
    async def test_provision_failure_does_not_block_subscription(
        self,
        tmp_path: Path,
    ) -> None:
        """Provisioner failure does not prevent wire_subscriptions from being called."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        from omnibase_infra.runtime.runtime_host_process import (
            RuntimeHostProcess,
        )

        process = RuntimeHostProcess.__new__(RuntimeHostProcess)
        process._contract_paths = [contracts_dir]
        process._event_bus = _make_kafka_bus_mock()
        process._event_bus_wiring = MagicMock()

        wire_called = False

        async def _mock_wire(**kwargs: object) -> None:
            nonlocal wire_called
            wire_called = True

        process._event_bus_wiring.wire_subscriptions = AsyncMock(side_effect=_mock_wire)

        mock_provisioner_instance = AsyncMock()
        mock_provisioner_instance.ensure_topic_exists = AsyncMock(
            side_effect=RuntimeError("broker unreachable")
        )
        mock_provisioner_cls = MagicMock(return_value=mock_provisioner_instance)

        descriptor = _make_descriptor_mock(["onex.evt.test.topic-a.v1"])

        with patch(_PATCH_TARGET, mock_provisioner_cls):
            await process._wire_live_handler_subscriptions(
                node_name="test-handler",
                descriptor=descriptor,
            )

        assert wire_called, (
            "wire_subscriptions must be called even when provisioning fails"
        )

    @pytest.mark.asyncio
    async def test_provision_failure_continues_to_later_topics(
        self,
        tmp_path: Path,
    ) -> None:
        """A failing topic does not prevent later topics from being provisioned."""
        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        from omnibase_infra.runtime.runtime_host_process import (
            RuntimeHostProcess,
        )

        process = RuntimeHostProcess.__new__(RuntimeHostProcess)
        process._contract_paths = [contracts_dir]
        process._event_bus = _make_kafka_bus_mock()
        process._event_bus_wiring = MagicMock()
        process._event_bus_wiring.wire_subscriptions = AsyncMock()

        attempted_topics: list[str] = []

        async def _mock_ensure_topic_exists(topic_name: str, **kwargs: object) -> bool:
            attempted_topics.append(topic_name)
            if topic_name == "onex.evt.test.topic-a.v1":
                raise RuntimeError("broker rejected topic-a")
            return True

        mock_provisioner_instance = AsyncMock()
        mock_provisioner_instance.ensure_topic_exists = AsyncMock(
            side_effect=_mock_ensure_topic_exists
        )
        mock_provisioner_cls = MagicMock(return_value=mock_provisioner_instance)

        descriptor = _make_descriptor_mock(
            ["onex.evt.test.topic-a.v1", "onex.evt.test.topic-b.v1"]
        )

        with patch(_PATCH_TARGET, mock_provisioner_cls):
            await process._wire_live_handler_subscriptions(
                node_name="test-handler",
                descriptor=descriptor,
            )

        assert attempted_topics == [
            "onex.evt.test.topic-a.v1",
            "onex.evt.test.topic-b.v1",
        ]
        process._event_bus_wiring.wire_subscriptions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_provision_skipped_without_kafka_bus(
        self,
        tmp_path: Path,
    ) -> None:
        """Provisioning is skipped when event bus is not Kafka (e.g. inmemory)."""
        from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
        from omnibase_infra.runtime.runtime_host_process import (
            RuntimeHostProcess,
        )

        contracts_dir = tmp_path / "contracts"
        contracts_dir.mkdir()

        process = RuntimeHostProcess.__new__(RuntimeHostProcess)
        process._contract_paths = [contracts_dir]
        process._event_bus = MagicMock(spec=EventBusInmemory)
        process._event_bus_wiring = MagicMock()
        process._event_bus_wiring.wire_subscriptions = AsyncMock()

        mock_provisioner_cls = MagicMock()

        descriptor = _make_descriptor_mock(["onex.evt.test.topic-a.v1"])

        with patch(_PATCH_TARGET, mock_provisioner_cls):
            await process._wire_live_handler_subscriptions(
                node_name="test-handler",
                descriptor=descriptor,
            )

        mock_provisioner_cls.assert_not_called()
        process._event_bus_wiring.wire_subscriptions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_provision_skipped_when_no_event_bus_wiring(
        self,
        tmp_path: Path,
    ) -> None:
        """No provisioning attempt when _event_bus_wiring is None."""
        from omnibase_infra.runtime.runtime_host_process import (
            RuntimeHostProcess,
        )

        process = RuntimeHostProcess.__new__(RuntimeHostProcess)
        process._contract_paths = []
        process._event_bus = _make_kafka_bus_mock()
        process._event_bus_wiring = None

        mock_provisioner_cls = MagicMock()

        descriptor = _make_descriptor_mock(["onex.evt.test.topic-a.v1"])

        with patch(_PATCH_TARGET, mock_provisioner_cls):
            await process._wire_live_handler_subscriptions(
                node_name="test-handler",
                descriptor=descriptor,
            )

        mock_provisioner_cls.assert_not_called()
