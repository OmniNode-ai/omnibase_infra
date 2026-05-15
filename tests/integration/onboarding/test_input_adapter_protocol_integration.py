# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration tests for ProtocolInputAdapter, AdapterFakeInput, AdapterCliInput (OMN-10781).

Verifies that the adapter implementations satisfy the ProtocolInputAdapter
runtime-checkable protocol and that AdapterFakeInput behaves correctly
with ModelInteractiveStep fixtures.
"""

from __future__ import annotations

import pytest

from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.enum_interactive_step_type import EnumInteractiveStepType
from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep
from omnibase_infra.onboarding.protocol_input_adapter import ProtocolInputAdapter

pytestmark = pytest.mark.integration


def _make_step(
    step_id: str,
    step_type: EnumInteractiveStepType,
    options: list[str] | None = None,
    required: bool = True,
) -> ModelInteractiveStep:
    return ModelInteractiveStep(
        id=step_id,
        type=step_type,
        prompt=f"Test prompt for {step_id}",
        options=options or [],
        required=required,
    )


def test_adapter_fake_input_satisfies_protocol() -> None:
    adapter = AdapterFakeInput(responses={})
    assert isinstance(adapter, ProtocolInputAdapter)


@pytest.mark.asyncio
async def test_fake_adapter_collect_choice() -> None:
    step = _make_step(
        "deploy_mode", EnumInteractiveStepType.CHOICE, options=["local", "cloud"]
    )
    adapter = AdapterFakeInput(responses={"deploy_mode": "local"})
    result = await adapter.collect_choice(step)
    assert result == "local"


@pytest.mark.asyncio
async def test_fake_adapter_collect_multi_choice() -> None:
    step = _make_step(
        "services",
        EnumInteractiveStepType.MULTI_CHOICE,
        options=["kafka", "postgres", "valkey"],
    )
    adapter = AdapterFakeInput(responses={"services": ["kafka", "postgres"]})
    result = await adapter.collect_multi_choice(step)
    assert result == ["kafka", "postgres"]


@pytest.mark.asyncio
async def test_fake_adapter_collect_multi_choice_comma_string() -> None:
    step = _make_step(
        "services", EnumInteractiveStepType.MULTI_CHOICE, options=["kafka", "postgres"]
    )
    adapter = AdapterFakeInput(responses={"services": "kafka, postgres"})
    result = await adapter.collect_multi_choice(step)
    assert result == ["kafka", "postgres"]


@pytest.mark.asyncio
async def test_fake_adapter_collect_text() -> None:
    step = _make_step("custom_host", EnumInteractiveStepType.TEXT, required=False)
    adapter = AdapterFakeInput(responses={"custom_host": "192.168.1.10"})
    result = await adapter.collect_text(step)
    assert result == "192.168.1.10"


@pytest.mark.asyncio
async def test_fake_adapter_collect_text_missing_optional_returns_empty() -> None:
    step = _make_step("custom_host", EnumInteractiveStepType.TEXT, required=False)
    adapter = AdapterFakeInput(responses={})
    result = await adapter.collect_text(step)
    assert result == ""


@pytest.mark.asyncio
async def test_fake_adapter_notify_action_records_step() -> None:
    step = _make_step("write_config", EnumInteractiveStepType.ACTION)
    adapter = AdapterFakeInput(responses={})
    await adapter.notify_action(step)
    assert "write_config" in adapter.notified_steps
