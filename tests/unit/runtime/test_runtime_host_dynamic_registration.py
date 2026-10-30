# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for post-freeze dynamic contract registration listener (OMN-11247).

Tests validate that RuntimeHostProcess._start_dynamic_contract_listener() and
_on_dynamic_contract_event() work correctly for post-freeze contract registration.

Tests use RuntimeHostProcess.__new__() to skip __init__ and manually set
internal state, isolating the methods under test from full runtime setup.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraTimeoutError, ProtocolConfigurationError
from omnibase_infra.models.errors.model_timeout_error_context import (
    ModelTimeoutErrorContext,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION

pytestmark = pytest.mark.unit

# The listener runs only where the runtime config opts in and its allowlist
# names the environment (OMN-20079).
_HOTLOAD_ON: dict[str, object] = {
    "enabled": True,
    "mode": "on_demand",
    "allowed_environments": ["dev"],
}


@pytest.fixture(autouse=True)
def _dev_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_ENVIRONMENT", "dev")


def _join_timeout() -> InfraTimeoutError:
    return InfraTimeoutError(
        "registration assignment timed out",
        context=ModelTimeoutErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="start_consumer",
            target_name=SUFFIX_NODE_REGISTRATION,
        ),
    )


def _recovery_config(
    *, initial: float, maximum: float = 1.0, attempt_timeout: float = 1.0
) -> dict[str, object]:
    return {
        "contract_hotload": _HOTLOAD_ON,
        "event_bus": {
            "environment": "dev",
            "recovery": {
                "initial_backoff_seconds": initial,
                "max_backoff_seconds": maximum,
                "backoff_multiplier": 2,
                "attempt_timeout_seconds": attempt_timeout,
            },
        },
    }


def _retry_delays(caplog: pytest.LogCaptureFixture) -> list[float]:
    return [
        float(r.args[0])
        for r in caplog.records
        if r.getMessage().startswith("Dynamic contract listener unavailable")
    ]


# ---------------------------------------------------------------------------
# Shared YAML fixture
# ---------------------------------------------------------------------------

VALID_CONTRACT_YAML = """\
name: node_dynamic_test
handler_id: proto.dynamic_test
contract_version:
  major: 1
  minor: 0
  patch: 0
description: Dynamically registered test handler
input_model:
  name: JsonDict
  module: omnibase_infra.models.types
output_model:
  name: ModelHandlerOutput
  module: omnibase_core.models.dispatch.model_handler_output
descriptor:
  node_archetype: EFFECT_GENERIC
metadata:
  handler_class: omnibase_infra.handlers.handler_http.HandlerHttp
event_bus:
  subscribe_topics:
    - onex.evt.test.dynamic-reg.v1
  publish_topics: []
handler_routing:
  routing_strategy: payload_type_match
  handlers:
    - handler:
        name: HandlerHttp
        module: omnibase_infra.handlers.handler_http
"""


def _make_msg(payload: dict) -> MagicMock:
    """Create a mock Kafka message with a JSON-encoded payload."""
    msg = MagicMock()
    msg.value = json.dumps(payload).encode("utf-8")
    msg.headers = None
    return msg


def _make_process(
    kafka_source: object | None = "AUTO",
    event_bus: object | None = "AUTO",
) -> RuntimeHostProcess:
    """Create a RuntimeHostProcess without running __init__."""
    process = RuntimeHostProcess.__new__(RuntimeHostProcess)

    if kafka_source == "AUTO":
        process._kafka_contract_source = MagicMock()
    else:
        process._kafka_contract_source = kafka_source

    if event_bus == "AUTO":
        process._event_bus = AsyncMock()
        process._event_bus.subscribe = AsyncMock(return_value=AsyncMock())
    else:
        process._event_bus = event_bus

    process._node_identity = MagicMock()
    process._node_identity.service = "test-service"
    process._node_identity.node_name = "test-node"
    process._node_identity.version = "v1"
    process._dynamic_contract_unsubscribe = None
    process._config = {
        "contract_hotload": _HOTLOAD_ON,
        "event_bus": {"environment": "dev"},
    }
    return process


# =============================================================================
# _start_dynamic_contract_listener tests
# =============================================================================


class TestStartDynamicContractListener:
    """Tests for _start_dynamic_contract_listener."""

    @pytest.mark.asyncio
    async def test_transient_join_timeout_retries_without_tearing_down_runtime(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A late registration join must not remove already-live consumers."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._config = _recovery_config(initial=0.001, maximum=0.003)
        unsubscribe = AsyncMock()
        process._event_bus.subscribe.side_effect = [
            _join_timeout(),
            _join_timeout(),
            _join_timeout(),
            unsubscribe,
        ]

        with caplog.at_level(logging.WARNING):
            await process._start_dynamic_contract_listener()

        assert process._event_bus.subscribe.await_count == 4
        assert _retry_delays(caplog) == pytest.approx([0.001, 0.002, 0.003])
        assert process._dynamic_contract_unsubscribe is unsubscribe
        process._event_bus.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_attempt_timeout_abandons_hung_join_and_retries(self) -> None:
        """The per-attempt timeout must cut a hung subscribe and try again."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._config = _recovery_config(initial=0.001, attempt_timeout=0.05)
        unsubscribe = AsyncMock()
        hung_cancelled = asyncio.Event()

        async def hang_once(**_kwargs: object) -> AsyncMock:
            if process._event_bus.subscribe.await_count == 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    hung_cancelled.set()
                    raise
            return unsubscribe

        process._event_bus.subscribe.side_effect = hang_once

        await asyncio.wait_for(process._start_dynamic_contract_listener(), timeout=5)

        assert hung_cancelled.is_set()
        assert process._event_bus.subscribe.await_count == 2
        assert process._dynamic_contract_unsubscribe is unsubscribe

    @pytest.mark.asyncio
    async def test_shutdown_during_join_propagates_without_retry(self) -> None:
        """Cancelling inside the attempt timeout window must not be retried."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._config = _recovery_config(initial=0.001, attempt_timeout=60)
        joining = asyncio.Event()

        async def hang(**_kwargs: object) -> AsyncMock:
            joining.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        process._event_bus.subscribe.side_effect = hang
        task = asyncio.create_task(process._start_dynamic_contract_listener())
        try:
            await asyncio.wait_for(joining.wait(), timeout=1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        process._event_bus.subscribe.assert_awaited_once()
        assert process._dynamic_contract_unsubscribe is None

    @pytest.mark.asyncio
    async def test_shutdown_during_backoff_uses_contract_policy_and_stops(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without host recovery config the shipped policy sets the backoff,
        and shutdown during that backoff creates no further consumer."""
        from omnibase_infra.models.model_event_bus_recovery_config import (
            ModelEventBusRecoveryConfig,
        )

        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._event_bus.subscribe.side_effect = _join_timeout()
        policy = ModelEventBusRecoveryConfig.from_contract()

        with caplog.at_level(logging.WARNING):
            task = asyncio.create_task(process._start_dynamic_contract_listener())
            try:
                async with asyncio.timeout(1):
                    while not _retry_delays(caplog):
                        await asyncio.sleep(0.001)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        assert _retry_delays(caplog) == pytest.approx([policy.initial_backoff_seconds])
        process._event_bus.subscribe.assert_awaited_once()
        assert process._dynamic_contract_unsubscribe is None

    @pytest.mark.asyncio
    async def test_timeout_not_from_attempt_deadline_is_not_retried(self) -> None:
        """A TimeoutError raised inside subscribe while the attempt deadline
        has not expired is not a transient join failure and must propagate."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._config = _recovery_config(initial=0.001, attempt_timeout=60)
        process._event_bus.subscribe.side_effect = TimeoutError("unrelated")

        with pytest.raises(TimeoutError, match="unrelated"):
            await process._start_dynamic_contract_listener()

        process._event_bus.subscribe.assert_awaited_once()
        assert process._dynamic_contract_unsubscribe is None

    @pytest.mark.asyncio
    async def test_malformed_recovery_config_fails_before_subscribing(self) -> None:
        """An invalid recovery block is a configuration error, raised before
        any listener consumer is created."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._config = {
            "contract_hotload": _HOTLOAD_ON,
            "event_bus": {
                "environment": "dev",
                "recovery": {"initial_backoff_seconds": -1},
            },
        }

        with pytest.raises(ProtocolConfigurationError):
            await process._start_dynamic_contract_listener()

        process._event_bus.subscribe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_mapping_event_bus_config_fails_before_subscribing(self) -> None:
        """A non-mapping event_bus config is not silently replaced by defaults."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._config = {
            "contract_hotload": _HOTLOAD_ON,
            "event_bus": "not-a-mapping",
        }

        with pytest.raises(ProtocolConfigurationError, match="must be a mapping"):
            await process._start_dynamic_contract_listener()

        process._event_bus.subscribe.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dynamic_listener_configuration_error_is_not_retried(self) -> None:
        """Malformed subscription declarations still fail startup explicitly."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._event_bus.subscribe.side_effect = ValueError("invalid topic")
        with pytest.raises(ValueError, match="invalid topic"):
            await process._start_dynamic_contract_listener()
        process._event_bus.subscribe.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subscribes_to_node_registration_topic(self) -> None:
        """Listener subscribes to SUFFIX_NODE_REGISTRATION after start."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"

        await process._start_dynamic_contract_listener()

        process._event_bus.subscribe.assert_called_once()
        kwargs = process._event_bus.subscribe.call_args
        assert kwargs[1]["topic"] == SUFFIX_NODE_REGISTRATION

    @pytest.mark.asyncio
    async def test_stores_unsubscribe_callback(self) -> None:
        """Listener stores the unsubscribe callback in _dynamic_contract_unsubscribe."""
        unsubscribe_fn = AsyncMock()
        process = _make_process()
        process._kafka_contract_source.environment = "dev"
        process._event_bus.subscribe = AsyncMock(return_value=unsubscribe_fn)

        await process._start_dynamic_contract_listener()

        assert process._dynamic_contract_unsubscribe is unsubscribe_fn

    @pytest.mark.asyncio
    async def test_initializes_kafka_source_when_missing(self) -> None:
        """Listener initializes a KafkaContractSource in normal hybrid mode."""
        process = _make_process(kafka_source=None)

        await process._start_dynamic_contract_listener()

        assert process._kafka_contract_source is not None
        assert process._kafka_contract_source.environment == "dev"
        process._event_bus.subscribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_when_no_event_bus(self) -> None:
        """Listener is a no-op when event bus is None."""
        process = _make_process(event_bus=None)

        await process._start_dynamic_contract_listener()

        assert process._dynamic_contract_unsubscribe is None

    @pytest.mark.asyncio
    async def test_wires_on_dynamic_contract_event_as_handler(self) -> None:
        """Listener wires _on_dynamic_contract_event as the message handler."""
        process = _make_process()
        process._kafka_contract_source.environment = "dev"

        await process._start_dynamic_contract_listener()

        kwargs = process._event_bus.subscribe.call_args[1]
        assert kwargs["on_message"] == process._on_dynamic_contract_event


# =============================================================================
# _on_dynamic_contract_event tests
# =============================================================================


class TestOnDynamicContractEvent:
    """Tests for _on_dynamic_contract_event."""

    @pytest.mark.asyncio
    async def test_registration_event_triggers_cache_and_materialize(self) -> None:
        """Registration event calls on_contract_registered then materializes."""
        process = _make_process(event_bus=None)

        mock_source = MagicMock()
        mock_source.on_contract_registered = MagicMock(return_value=True)
        mock_source.on_contract_deregistered = MagicMock()

        descriptor = MagicMock()
        mock_source.get_cached_descriptor = MagicMock(return_value=descriptor)

        process._kafka_contract_source = mock_source
        process._materialize_handler_live = AsyncMock(return_value=True)

        msg = _make_msg(
            {
                "node_name": "node_dynamic_test",
                "contract_yaml": VALID_CONTRACT_YAML,
                "event_type": "registered",
                "correlation_id": str(uuid4()),
            }
        )

        await process._on_dynamic_contract_event(msg)

        mock_source.on_contract_registered.assert_called_once()
        call_kwargs = mock_source.on_contract_registered.call_args[1]
        assert call_kwargs["node_name"] == "node_dynamic_test"
        assert call_kwargs["contract_yaml"] == VALID_CONTRACT_YAML

        process._materialize_handler_live.assert_called_once()
        mat_kwargs = process._materialize_handler_live.call_args[1]
        assert mat_kwargs["node_name"] == "node_dynamic_test"
        assert mat_kwargs["descriptor"] is descriptor

    @pytest.mark.asyncio
    async def test_deregistration_event_calls_on_contract_deregistered(self) -> None:
        """Deregistration event calls on_contract_deregistered, skips materialize."""
        process = _make_process(event_bus=None)

        mock_source = MagicMock()
        mock_source.on_contract_deregistered = MagicMock()
        process._kafka_contract_source = mock_source
        process._materialize_handler_live = AsyncMock()

        msg = _make_msg(
            {
                "node_name": "node_dynamic_test",
                "event_type": "deregistered",
            }
        )

        await process._on_dynamic_contract_event(msg)

        mock_source.on_contract_deregistered.assert_called_once()
        process._materialize_handler_live.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_node_name_logs_rejection_does_not_raise(self) -> None:
        """Missing node_name logs a structured rejection but does not raise."""
        process = _make_process(event_bus=None)
        process._materialize_handler_live = AsyncMock()

        msg = _make_msg(
            {
                "contract_yaml": VALID_CONTRACT_YAML,
                "event_type": "registered",
            }
        )

        # Must not raise
        await process._on_dynamic_contract_event(msg)

        process._materialize_handler_live.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_contract_yaml_logs_rejection_does_not_raise(self) -> None:
        """Missing contract_yaml logs a structured rejection but does not raise."""
        process = _make_process(event_bus=None)
        process._kafka_contract_source.on_contract_registered = MagicMock()
        process._materialize_handler_live = AsyncMock()

        msg = _make_msg(
            {
                "node_name": "node_dynamic_test",
                "event_type": "registered",
            }
        )

        await process._on_dynamic_contract_event(msg)

        process._kafka_contract_source.on_contract_registered.assert_not_called()
        process._materialize_handler_live.assert_not_called()

    @pytest.mark.asyncio
    async def test_caching_failure_skips_materialization(self) -> None:
        """When on_contract_registered returns False, materialization is skipped."""
        process = _make_process(event_bus=None)

        mock_source = MagicMock()
        mock_source.on_contract_registered = MagicMock(return_value=False)
        process._kafka_contract_source = mock_source
        process._materialize_handler_live = AsyncMock()

        msg = _make_msg(
            {
                "node_name": "node_dynamic_test",
                "contract_yaml": VALID_CONTRACT_YAML,
                "event_type": "registered",
            }
        )

        await process._on_dynamic_contract_event(msg)

        process._materialize_handler_live.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_in_materialize_does_not_raise(self) -> None:
        """An exception in _materialize_handler_live must not crash the runtime."""
        process = _make_process(event_bus=None)

        mock_source = MagicMock()
        mock_source.on_contract_registered = MagicMock(return_value=True)
        mock_source.get_cached_descriptor = MagicMock(return_value=MagicMock())
        process._kafka_contract_source = mock_source
        process._materialize_handler_live = AsyncMock(side_effect=RuntimeError("boom"))

        msg = _make_msg(
            {
                "node_name": "node_dynamic_test",
                "contract_yaml": VALID_CONTRACT_YAML,
                "event_type": "registered",
            }
        )

        # Must not raise
        await process._on_dynamic_contract_event(msg)

    @pytest.mark.asyncio
    async def test_noop_when_no_kafka_source(self) -> None:
        """Handler is a no-op when _kafka_contract_source is None."""
        process = _make_process(kafka_source=None, event_bus=None)
        process._materialize_handler_live = AsyncMock()

        msg = _make_msg({"node_name": "x", "contract_yaml": "y"})

        await process._on_dynamic_contract_event(msg)

        process._materialize_handler_live.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_message_value_is_noop(self) -> None:
        """A message with no value is silently ignored."""
        process = _make_process(event_bus=None)
        process._materialize_handler_live = AsyncMock()

        msg = MagicMock()
        msg.value = None
        msg.headers = None

        await process._on_dynamic_contract_event(msg)

        process._materialize_handler_live.assert_not_called()
