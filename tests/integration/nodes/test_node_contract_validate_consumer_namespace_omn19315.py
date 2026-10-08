# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19315: consumer topic namespace validation through the real contract.

The node contract is read from disk and its declared command topic is the topic the
dispatch is driven for; the handler is bound through the same auto-wiring dispatch
callback the runtime uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import yaml

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.validation.model_contract_validation_result import (
    ModelContractValidationResult,
)
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_contract_validate_compute import handlers
from omnibase_infra.nodes.node_contract_validate_compute.handlers import (
    HandlerContractValidate,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
)

pytestmark = pytest.mark.integration

_CONTRACT_PATH = Path(handlers.__file__).resolve().parents[1] / "contract.yaml"
_BARE = "onex.evt.platform.node-registration.v1"
_PREFIXED = f"tenant-a.{_BARE}"


def test_real_contract_declares_command_response_and_dead_letter_topics() -> None:
    contract = yaml.safe_load(_CONTRACT_PATH.read_text(encoding="utf-8"))
    event_bus = contract["event_bus"]
    assert event_bus["subscribe_topics"]
    assert event_bus["publish_topics"]
    assert event_bus["dlq_topics"]


@pytest.mark.asyncio
async def test_mixed_consumer_source_is_refused_naming_both_topics() -> None:
    source = f'TOPICS = ["{_PREFIXED}", "{_BARE}"]\nconsumer.subscribe(topics=TOPICS)\n'
    callback = _make_dispatch_callback(
        cast("ProtocolHandleable", HandlerContractValidate())
    )
    result = await callback(
        ModelEventEnvelope(payload={"consumer_source": source}, correlation_id=uuid4())
    )
    assert isinstance(result, ModelDispatchResult)
    assert result.status is EnumDispatchStatus.SUCCESS
    (validation,) = result.output_events
    assert isinstance(validation, ModelContractValidationResult)
    assert validation.is_valid is False
    assert any(_PREFIXED in v and _BARE in v for v in validation.violations)
