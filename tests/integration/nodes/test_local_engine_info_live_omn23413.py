# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The engine_info read against the Docker this runner really has (OMN-23413).

No mock: the handler runs the real ``docker info`` through its dispatch
entrypoint. Docker answering is a positive CPU count and memory; Docker missing
or not answering is a typed ``unavailable`` result with the reason. Either way
the read changes nothing and never raises.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from omnibase_infra.handlers.handler_local_engine_info import HandlerLocalEngineInfo
from omnibase_infra.models.model_local_engine_info_request import (
    ModelLocalEngineInfoRequest,
)

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_the_real_engine_read_is_a_positive_read_or_a_named_failure() -> None:
    correlation_id = uuid4()
    result = await HandlerLocalEngineInfo(MagicMock()).handle_async(
        ModelLocalEngineInfoRequest(correlation_id=correlation_id)
    )
    assert result.correlation_id == correlation_id
    if result.engine_status == "read":
        assert result.success
        assert result.engine_cpus is not None and result.engine_cpus >= 1
        assert result.engine_memory_bytes is not None
        assert result.engine_memory_bytes >= 1
        assert result.error is None
    else:
        assert result.engine_status == "unavailable"
        assert not result.success
        assert result.engine_cpus is None and result.engine_memory_bytes is None
        assert result.error
