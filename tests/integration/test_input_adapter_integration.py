# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Integration coverage for interactive onboarding input adapters."""

from __future__ import annotations

import pytest

from omnibase_infra.onboarding.adapter_fake_input import AdapterFakeInput
from omnibase_infra.onboarding.model_interactive_step import ModelInteractiveStep
from omnibase_infra.onboarding.protocol_input_adapter import ProtocolInputAdapter

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fake_input_adapter_satisfies_protocol_for_step_sequence() -> None:
    adapter: ProtocolInputAdapter = AdapterFakeInput(
        {
            "choose_mode": "local",
            "choose_services": "kafka, postgres",
            "enter_name": "demo",
        }
    )

    choice_step = ModelInteractiveStep(
        id="choose_mode",
        prompt="Choose deployment mode",
        type="choice",
        options=["local", "cloud"],
    )
    multi_step = ModelInteractiveStep(
        id="choose_services",
        prompt="Choose services",
        type="multi_choice",
        options=["kafka", "postgres"],
    )
    text_step = ModelInteractiveStep(
        id="enter_name",
        prompt="Enter environment name",
        type="text",
        options=[],
    )

    assert isinstance(adapter, ProtocolInputAdapter)
    assert await adapter.collect_choice(choice_step) == "local"
    assert await adapter.collect_multi_choice(multi_step) == ["kafka", "postgres"]
    assert await adapter.collect_text(text_step) == "demo"
    await adapter.notify_action(text_step)
