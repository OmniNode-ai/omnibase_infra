# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Container-wired integration for the registration bootstrap effect (OMN-22446).

Resolves HandlerRegistrationBootstrap through the real container binding and
drives its configuration and DSN-validation boundary. No database is touched.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.errors import ContainerWiringError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.handlers.handler_registration_bootstrap import (
    HandlerRegistrationBootstrap,
)
from omnibase_infra.runtime.protocol_domain_plugin import ModelDomainPluginConfig


def _config(container: ModelONEXContainer) -> ModelDomainPluginConfig:
    return ModelDomainPluginConfig(
        container=container,
        event_bus=EventBusInmemory(),
        correlation_id=uuid4(),
        input_topic="registration-input",
        output_topic="registration-output",
        consumer_group="registration-test",
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_from_container_binds_one_handler_instance() -> None:
    container = ModelONEXContainer()
    first = await HandlerRegistrationBootstrap.from_container(container)
    second = await HandlerRegistrationBootstrap.from_container(container)
    assert first is second


@pytest.mark.integration
@pytest.mark.asyncio
async def test_missing_dsn_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIBASE_INFRA_DB_URL", "")
    container = ModelONEXContainer()
    handler = await HandlerRegistrationBootstrap.from_container(container)
    with pytest.raises(ContainerWiringError, match="OMNIBASE_INFRA_DB_URL"):
        await handler.handle(_config(container))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_dsn_scheme_is_rejected_before_pool_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIBASE_INFRA_DB_URL", "mysql://u@h/db")
    container = ModelONEXContainer()
    handler = await HandlerRegistrationBootstrap.from_container(container)
    with pytest.raises(ContainerWiringError):
        await handler.handle(_config(container))


@pytest.mark.integration
def test_projector_dir_override_and_contract_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ONEX_PROJECTOR_CONTRACTS_DIR", str(tmp_path))
    assert HandlerRegistrationBootstrap.projector_contracts_dir(Path("/x")) == tmp_path
    assert HandlerRegistrationBootstrap.load_contract(tmp_path / "missing.yaml") == {}
    good = tmp_path / "c.yaml"
    good.write_text("name: demo\n", encoding="utf-8")
    assert HandlerRegistrationBootstrap.load_contract(good) == {"name": "demo"}
