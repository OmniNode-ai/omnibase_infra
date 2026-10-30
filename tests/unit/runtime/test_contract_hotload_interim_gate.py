# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Interim contract hot-load gate (OMN-20079, plan task H0).

Both runtime hot-load paths are off unless the runtime config carries
``contract_hotload.enabled: true``:

* the listener on ``onex.evt.platform.node-registration.v1``, and
* the baseline subscription on ``onex.evt.platform.contract-registered.v1``
  that ``handler_source.mode: kafka_events`` turns on.

The planted-violation fixtures are real contract payloads that the runtime
materializes when the gate is on (the control tests), so the gate-off tests
cannot pass merely because the payload was unparseable.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from omnibase_core.constants import TOPIC_SUFFIX_CONTRACT_REGISTERED
from omnibase_infra.enums import EnumConsumerGroupPurpose
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.models import ModelEventMessage
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.handlers import ModelHandlerDescriptor
from omnibase_infra.models.model_contract_hotload_config import (
    ModelContractHotloadConfig,
)
from omnibase_infra.runtime.runtime_host_process import RuntimeHostProcess
from omnibase_infra.topics import SUFFIX_NODE_REGISTRATION

pytestmark = pytest.mark.unit

_PROD_ENVIRONMENT = "onex-prod"  # k8s/onex-prod/runtime/configmap.yaml ONEX_ENVIRONMENT

# A parseable handler-shell contract: the strict shell parse accepts it, so with
# the gate on it is cached and materialized.
HANDLER_SHELL_YAML = """\
handler_id: "effect.hotload_gate.shell"
name: "node_hotload_gate_shell"
contract_version:
  major: 1
  minor: 0
  patch: 0
description: "Planted handler-shell contract for the interim gate"
descriptor:
  node_archetype: "effect"
input_model: "omnibase_infra.models.types.JsonDict"
output_model: "omnibase_core.models.dispatch.ModelHandlerOutput"
metadata:
  handler_class: "omnibase_infra.handlers.TestHandler"
"""

# An ordinary installed node contract, announced verbatim by the kernel at boot.
ORDINARY_NODE_CONTRACT = (
    Path(__file__).resolve().parents[3]
    / "src/omnibase_infra/nodes/node_contract_registry_reducer/contract.yaml"
)


def _hotload(**fields: object) -> dict[str, object]:
    return {"contract_hotload": fields}


class _RecordingBus(EventBusInmemory):
    """An in-memory bus that records the topics subscribed and whether it started."""

    def __init__(self) -> None:
        super().__init__(environment="dev", group="hotload-gate")
        self.subscribed_topics: list[str] = []
        self.start_awaited = False

    async def start(self) -> None:
        self.start_awaited = True
        await super().start()

    async def subscribe(
        self, topic: str, *args: Any, **kwargs: Any
    ) -> Callable[[], Awaitable[None]]:
        self.subscribed_topics.append(topic)
        return await super().subscribe(topic, *args, **kwargs)


class _CountingHost(RuntimeHostProcess):
    """A real host whose live materialization is counted instead of performed."""

    materialized: list[str]

    async def _materialize_handler_live(
        self,
        node_name: str,
        descriptor: ModelHandlerDescriptor,
        correlation_id: UUID,
    ) -> bool:
        self.materialized.append(node_name)
        return True


def _host(config: dict[str, object], bus: _RecordingBus | None = None) -> _CountingHost:
    host = _CountingHost(
        event_bus=bus or _RecordingBus(),
        config={"service_name": "hotload_gate", "node_name": "gate", **config},
    )
    host.materialized = []
    host._is_running = True
    return host


def _generation_consumer_registration() -> bytes:
    """The payload node_generation_consumer publishes on node-registration.v1."""
    return json.dumps(
        {
            "event_type": "registered",
            "correlation_id": "6f1c3f0e-0a55-4c19-9a52-7d1c2f4f6a11",
            "node_name": "node_hotload_gate_shell",
            "service_name": "node_hotload_gate_shell",
            "contract_yaml": HANDLER_SHELL_YAML,
            "source": "node_generation_consumer",
        }
    ).encode()


def _boot_announcement(node_name: str, contract_yaml: str) -> bytes:
    """The kernel's boot announcement of an installed contract."""
    return json.dumps(
        {
            "event_type": "registered",
            "node_name": node_name,
            "contract_yaml": contract_yaml,
        }
    ).encode()


async def _wire_both_paths(host: RuntimeHostProcess) -> None:
    """Run the two startup steps that create the hot-load subscriptions."""
    await host._resolve_handler_descriptors()
    await host._wire_baseline_subscriptions()
    await host._start_dynamic_contract_listener()


async def _publish_all_three(bus: _RecordingBus) -> None:
    ordinary = ORDINARY_NODE_CONTRACT.read_text(encoding="utf-8")
    await bus.publish(
        SUFFIX_NODE_REGISTRATION, None, _generation_consumer_registration()
    )
    await bus.publish(
        TOPIC_SUFFIX_CONTRACT_REGISTERED,
        None,
        _boot_announcement("node_hotload_gate_shell", HANDLER_SHELL_YAML),
    )
    await bus.publish(
        TOPIC_SUFFIX_CONTRACT_REGISTERED,
        None,
        _boot_announcement("node_contract_registry_reducer", ordinary),
    )


# ---------------------------------------------------------------------------
# The configuration model
# ---------------------------------------------------------------------------


class TestModelContractHotloadConfig:
    def test_default_is_off(self) -> None:
        assert ModelContractHotloadConfig().enabled is False

    def test_mode_is_required_when_enabled(self) -> None:
        with pytest.raises(ValueError, match="mode is required"):
            ModelContractHotloadConfig(enabled=True)

    @pytest.mark.parametrize("mode", ["on_demand", "full_load"])
    def test_enabled_with_a_known_mode(self, mode: str) -> None:
        gate = ModelContractHotloadConfig.model_validate(
            {"enabled": True, "mode": mode}
        )
        assert gate.enabled is True

    def test_unknown_mode_and_unknown_field_are_refused(self) -> None:
        with pytest.raises(ValueError):
            ModelContractHotloadConfig.model_validate(
                {"enabled": True, "mode": "always"}
            )
        with pytest.raises(ValueError):
            ModelContractHotloadConfig.model_validate({"enabled": False, "extra": 1})

    def test_model_is_frozen(self) -> None:
        assert ModelContractHotloadConfig.model_config["frozen"] is True


# ---------------------------------------------------------------------------
# The gate and its environment check
# ---------------------------------------------------------------------------


class TestGateResolution:
    def test_no_block_is_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ONEX_ENVIRONMENT", raising=False)
        assert _host({})._resolve_contract_hotload_gate().enabled is False

    def test_enabled_in_a_non_production_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_ENVIRONMENT", "dev")
        gate = _host(
            _hotload(enabled=True, mode="on_demand")
        )._resolve_contract_hotload_gate()
        assert gate.enabled is True

    def test_enabled_with_the_real_prod_environment_value_is_a_boot_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_ENVIRONMENT", _PROD_ENVIRONMENT)
        host = _host(_hotload(enabled=True, mode="on_demand"))
        with pytest.raises(ProtocolConfigurationError, match="production"):
            host._resolve_contract_hotload_gate()

    def test_enabled_with_a_prod_event_bus_environment_is_a_boot_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ONEX_ENVIRONMENT", raising=False)
        config = {
            **_hotload(enabled=True, mode="full_load"),
            "event_bus": {"environment": _PROD_ENVIRONMENT},
        }
        with pytest.raises(ProtocolConfigurationError, match="production"):
            _host(config)._resolve_contract_hotload_gate()

    def test_disabled_in_prod_is_the_normal_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_ENVIRONMENT", _PROD_ENVIRONMENT)
        assert _host({})._resolve_contract_hotload_gate().enabled is False
        assert (
            _host(_hotload(enabled=False))._resolve_contract_hotload_gate().enabled
            is False
        )

    @pytest.mark.parametrize(
        "block",
        [
            "yes",
            {"enabled": True},
            {"enabled": True, "mode": "always"},
            {"enabled": False, "surprise": 1},
        ],
    )
    def test_malformed_block_is_a_boot_error(
        self, block: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ONEX_ENVIRONMENT", raising=False)
        with pytest.raises(ProtocolConfigurationError):
            _host({"contract_hotload": block})._resolve_contract_hotload_gate()

    @pytest.mark.asyncio
    async def test_start_refuses_before_the_event_bus_starts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_ENVIRONMENT", _PROD_ENVIRONMENT)
        bus = _RecordingBus()
        host = _host(_hotload(enabled=True, mode="on_demand"), bus)
        host._is_running = False
        with pytest.raises(ProtocolConfigurationError, match="production"):
            await host.start()
        assert bus.start_awaited is False


# ---------------------------------------------------------------------------
# AC1: disabled by default, both paths
# ---------------------------------------------------------------------------


class TestDisabledByDefault:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "block",
        [{}, _hotload(enabled=False), _hotload(enabled=False, mode="on_demand")],
        ids=["no-block", "explicit-off", "off-with-mode"],
    )
    async def test_no_subscription_and_nothing_materialized(
        self,
        block: dict[str, object],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RUNTIME_PROFILE", "main")
        monkeypatch.delenv("ONEX_ENVIRONMENT", raising=False)
        bus = _RecordingBus()
        host = _host({"handler_source": {"mode": "kafka_events"}, **block}, bus)
        await bus.start()

        await _wire_both_paths(host)

        # The kafka_events source exists, so the baseline path is armed apart
        # from the gate: the zero below is the gate's doing.
        assert host._kafka_contract_source is not None
        assert SUFFIX_NODE_REGISTRATION not in bus.subscribed_topics
        assert TOPIC_SUFFIX_CONTRACT_REGISTERED not in bus.subscribed_topics
        assert host._dynamic_contract_unsubscribe is None

        await _publish_all_three(bus)

        assert host.materialized == []
        await bus.close()


class TestControlGateOnMaterializes:
    """The same fixtures do materialize with the gate on: the falsifier.

    The two handler-shell payloads (the generation consumer's registration and
    a boot announcement) each materialize through their own path. The ordinary
    node contract fails the strict handler-shell parse (the plan's 1.1), so it
    materializes in neither state; the gate-off test still publishes it so that
    a future relaxed parse cannot slip it past the gate.
    """

    @pytest.mark.asyncio
    async def test_both_paths_materialize_the_planted_shell_contracts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUNTIME_PROFILE", "main")
        monkeypatch.setenv("ONEX_ENVIRONMENT", "dev")
        bus = _RecordingBus()
        host = _host(
            {
                "handler_source": {"mode": "kafka_events"},
                **_hotload(enabled=True, mode="on_demand"),
            },
            bus,
        )
        await bus.start()

        await _wire_both_paths(host)
        assert SUFFIX_NODE_REGISTRATION in bus.subscribed_topics
        assert TOPIC_SUFFIX_CONTRACT_REGISTERED in bus.subscribed_topics

        await bus.publish(
            SUFFIX_NODE_REGISTRATION, None, _generation_consumer_registration()
        )
        await bus.publish(
            TOPIC_SUFFIX_CONTRACT_REGISTERED,
            None,
            _boot_announcement("node_hotload_gate_shell", HANDLER_SHELL_YAML),
        )
        await bus.publish(
            TOPIC_SUFFIX_CONTRACT_REGISTERED,
            None,
            _boot_announcement(
                "node_contract_registry_reducer",
                ORDINARY_NODE_CONTRACT.read_text(encoding="utf-8"),
            ),
        )

        assert host.materialized == ["node_hotload_gate_shell"] * 2
        await bus.close()
