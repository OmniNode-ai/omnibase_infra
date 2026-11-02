# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The registration orchestrator takes its operator switches by injection.

The auto-ACK switch (dispatcher) and the liveness interval (handler module) come
from an injected ``ModelRegistrationSettings``. Only a caller that injects
nothing falls back to the node contract's ``descriptor`` declarations, which the
overlay resolves; the process environment never changes an injected value.

Related Tickets:
    - OMN-21698: Registration orchestrator clock and settings injection
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.registration.events.model_node_registration_accepted import (
    ModelNodeRegistrationAccepted,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.registration.model_registration_settings import (
    DEFAULT_LIVENESS_INTERVAL_SECONDS,
    ModelRegistrationSettings,
)
from omnibase_infra.nodes.node_registration_orchestrator.dispatchers.dispatcher_node_introspected import (
    DispatcherNodeIntrospected,
)
from omnibase_infra.nodes.node_registration_orchestrator.handlers.handler_node_registration_acked import (
    get_liveness_interval_seconds,
)

ENV_AUTO_ACK = "ONEX_REGISTRATION_AUTO_ACK"
ENV_INTERVAL = "ONEX_LIVENESS_INTERVAL_SECONDS"
NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)


def _dispatcher(
    settings: ModelRegistrationSettings | None,
) -> tuple[DispatcherNodeIntrospected, AsyncMock]:
    """A dispatcher whose handler reports an accepted registration."""
    accepted = MagicMock(spec=ModelNodeRegistrationAccepted)
    handler = MagicMock()
    handler.handle = AsyncMock(
        return_value=SimpleNamespace(events=(accepted,), intents=())
    )
    bus = AsyncMock(spec=ProtocolEventBus)
    return (
        DispatcherNodeIntrospected(handler, event_bus=bus, settings=settings),
        bus.publish_envelope,
    )


def _envelope() -> ModelEventEnvelope[object]:
    event = ModelNodeIntrospectionEvent(
        node_id=uuid4(),
        node_type=EnumNodeKind.EFFECT,
        correlation_id=uuid4(),
        timestamp=NOW,
    )
    return ModelEventEnvelope(
        envelope_id=uuid4(),
        payload=event,
        envelope_timestamp=NOW,
        correlation_id=uuid4(),
        source_tool="test",
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("env_value", [None, "true", "false"])
async def test_injected_auto_ack_on_follows_the_injection_not_the_environment(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv(ENV_AUTO_ACK, raising=False)
    else:
        monkeypatch.setenv(ENV_AUTO_ACK, env_value)
    dispatcher, publish = _dispatcher(ModelRegistrationSettings(auto_ack=True))

    await dispatcher.handle(_envelope())

    publish.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("env_value", [None, "true", "false"])
async def test_injected_auto_ack_off_follows_the_injection_not_the_environment(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv(ENV_AUTO_ACK, raising=False)
    else:
        monkeypatch.setenv(ENV_AUTO_ACK, env_value)
    dispatcher, publish = _dispatcher(ModelRegistrationSettings(auto_ack=False))

    await dispatcher.handle(_envelope())

    publish.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_uninjected_dispatcher_resolves_the_contract_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_AUTO_ACK, raising=False)
    dispatcher, publish = _dispatcher(None)
    await dispatcher.handle(_envelope())
    publish.assert_not_awaited()

    monkeypatch.setenv(ENV_AUTO_ACK, "true")
    dispatcher, publish = _dispatcher(None)
    await dispatcher.handle(_envelope())
    publish.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.parametrize("env_value", [None, "180"])
def test_injected_interval_follows_the_injection_not_the_environment(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None
) -> None:
    if env_value is None:
        monkeypatch.delenv(ENV_INTERVAL, raising=False)
    else:
        monkeypatch.setenv(ENV_INTERVAL, env_value)

    settings = ModelRegistrationSettings(liveness_interval_seconds=45)

    assert get_liveness_interval_seconds(settings=settings) == 45
    assert get_liveness_interval_seconds(120, settings=settings) == 120


@pytest.mark.unit
def test_uninjected_interval_resolves_the_contract_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ENV_INTERVAL, raising=False)
    assert get_liveness_interval_seconds() == DEFAULT_LIVENESS_INTERVAL_SECONDS
    assert ModelRegistrationSettings.from_contract() == ModelRegistrationSettings()

    monkeypatch.setenv(ENV_INTERVAL, "300")
    assert get_liveness_interval_seconds() == 300

    monkeypatch.setenv(ENV_INTERVAL, "not_a_number")
    with pytest.raises(ProtocolConfigurationError):
        get_liveness_interval_seconds()


def test_contract_fallback_reads_the_packaged_contract_resource(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fallback reads contract.yaml as package data of the orchestrator node.

    The declarations come from ``importlib.resources`` of
    ``omnibase_infra.nodes.node_registration_orchestrator``, not a path walked up
    from this module's file, so a zipped or relocated install resolves them too.
    """
    from omnibase_infra.models.registration import model_registration_settings as mrs

    (tmp_path / "contract.yaml").write_text(
        "descriptor:\n"
        '  registration_auto_ack: "true"\n'
        '  liveness_interval_seconds: "123"\n',
        encoding="utf-8",
    )
    asked: list[str] = []

    def fake_files(package: str) -> Path:
        asked.append(package)
        return tmp_path

    monkeypatch.setattr(mrs, "files", fake_files)
    mrs._declared_descriptor.cache_clear()
    try:
        settings = ModelRegistrationSettings.from_contract()
    finally:
        mrs._declared_descriptor.cache_clear()

    assert asked == ["omnibase_infra.nodes.node_registration_orchestrator"]
    assert settings.auto_ack is True
    assert settings.liveness_interval_seconds == 123
