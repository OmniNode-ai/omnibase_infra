# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for ModelBifrostRequest session_id + run_id fields (OMN-9159)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums.enum_llm_operation_type import EnumLlmOperationType
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_request import (
    ModelBifrostRequest,
)


def _make_request(**kwargs: object) -> ModelBifrostRequest:
    defaults: dict[str, object] = {
        "operation_type": EnumLlmOperationType.CHAT_COMPLETION,
        "tenant_id": uuid4(),
        "correlation_id": uuid4(),
        "messages": ({"role": "user", "content": "hello"},),
    }
    defaults.update(kwargs)
    return ModelBifrostRequest(**defaults)  # type: ignore[arg-type]


@pytest.mark.integration
def test_bifrost_request_session_and_run_id_propagated() -> None:
    corr_id = uuid4()
    req = _make_request(
        correlation_id=corr_id,
        session_id="onex-session-abc",
        run_id="onex-run-001",
    )
    assert req.session_id == "onex-session-abc"
    assert req.run_id == "onex-run-001"
    assert req.correlation_id == corr_id


@pytest.mark.integration
def test_bifrost_request_session_and_run_id_optional() -> None:
    req = _make_request()
    assert req.session_id is None
    assert req.run_id is None


@pytest.mark.integration
def test_bifrost_request_correlation_id_required() -> None:
    with pytest.raises(ValidationError):
        ModelBifrostRequest(
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            tenant_id=uuid4(),
            messages=({"role": "user", "content": "hello"},),
        )
