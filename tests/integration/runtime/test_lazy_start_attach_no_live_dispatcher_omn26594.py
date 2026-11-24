# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26594: a real standalone-runner contract attaches on a lazy start without failing.

Nothing is faked but the bus: the contract is wired by the real ``wire_from_manifest`` and the real
``subscribe_wired_contract_topics`` excludes it as ``NO_LIVE_DISPATCHER``. The real
``LazyStartRuntimeTransaction.attach`` then reads that exclusion instead of raising ``did not attach``.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest

from omnibase_core.container import ModelONEXContainer
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.handlers.lazy_start_runtime_transaction import (
    LazyStartRuntimeTransaction,
    NodeStart,
    RecordingBus,
)
from omnibase_infra.protocols.protocol_lazy_start_bus import ProtocolLazyStartBus
from tests.unit.runtime.auto_wiring.test_omn17562_subscription_skip import (
    _RUNNER_ONLY,
    _dsns,
    _event_bus,
    _runner_only_contract,
    _wire,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_attach_of_a_really_wired_standalone_runner_returns() -> None:
    contract = _runner_only_contract()
    bus = _event_bus()
    report, engine, manifest = await _wire(contract, bus, subscribe_immediately=False)
    transaction = LazyStartRuntimeTransaction(
        {_RUNNER_ONLY: contract},
        cast("ProtocolLazyStartBus", bus),
        "local",
        None,
        ModelONEXContainer(),
        ModelTopicReadinessConfig(),
    )
    transaction._starts[_RUNNER_ONLY] = NodeStart(
        engine=engine,
        manifest=manifest,
        report=report,
        bus=RecordingBus(cast("ProtocolLazyStartBus", bus)),
    )

    await transaction.attach(
        _RUNNER_ONLY
    )  # raised "did not attach: [] []" before OMN-26594

    bus.subscribe.assert_not_called()
