# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test: consume boundary routes to the contract-declared DLQ (OMN-18879).

Drives manifest wiring, real dispatch and the real Kafka DLQ publisher together;
only the broker producer is replaced by a recording double (no broker).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    BoundaryDlqNotPersistedError,
)
from tests.unit.runtime.auto_wiring.test_contract_declared_dlq_omn18879 import (
    DECLARED_DLQ,
    RecordingKafkaBus,
    RecordingProducer,
    _message,
    _wire,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture(autouse=True)
def _dlq_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_BOUNDARY_DLQ_ENABLED", "true")


async def test_rejection_lands_on_declared_dlq(tmp_path: Path) -> None:
    producer = RecordingProducer()
    callback = await _wire(tmp_path, RecordingKafkaBus(producer))
    await callback(_message())
    assert [topic for topic, _ in producer.sent] == [DECLARED_DLQ]


async def test_unacknowledged_declared_dlq_withholds_offset(tmp_path: Path) -> None:
    producer = RecordingProducer(fail_topic=DECLARED_DLQ)
    callback = await _wire(tmp_path, RecordingKafkaBus(producer))
    with pytest.raises(BoundaryDlqNotPersistedError):
        await callback(_message())
    assert producer.sent == []
