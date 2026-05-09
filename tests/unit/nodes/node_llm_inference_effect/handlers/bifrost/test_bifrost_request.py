# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ModelBifrostRequest identity fields (OMN-9159)."""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.enums.enum_cost_tier import EnumCostTier
from omnibase_infra.enums.enum_llm_operation_type import EnumLlmOperationType
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_request import (
    ModelBifrostRequest,
)

_TENANT = UUID("12345678-1234-5678-1234-567812345678")
_CORR = UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

_BASE = {
    "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
    "tenant_id": _TENANT,
    "messages": [{"role": "user", "content": "hi"}],
    "correlation_id": _CORR,
}


@pytest.mark.unit
def test_correlation_id_is_required() -> None:
    """correlation_id must be provided; missing it raises ValidationError."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ModelBifrostRequest(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            tenant_id=_TENANT,
            messages=[{"role": "user", "content": "hi"}],
            # correlation_id intentionally omitted
        )


@pytest.mark.unit
def test_session_id_and_run_id_accepted() -> None:
    """session_id and run_id can be provided and are preserved."""
    req = ModelBifrostRequest(
        **_BASE,
        session_id="session-abc-123",
        run_id="run-xyz-456",
    )
    assert req.session_id == "session-abc-123"
    assert req.run_id == "run-xyz-456"
    assert req.correlation_id == _CORR


@pytest.mark.unit
def test_session_id_and_run_id_optional() -> None:
    """session_id and run_id default to None when omitted."""
    req = ModelBifrostRequest(**_BASE)
    assert req.session_id is None
    assert req.run_id is None
