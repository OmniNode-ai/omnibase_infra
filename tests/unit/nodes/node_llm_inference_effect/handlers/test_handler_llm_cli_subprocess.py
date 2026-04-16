# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for HandlerLlmCliSubprocess.

OMN-8735 follow-up: auto-wiring compliance — no-args construction.
"""

from __future__ import annotations

import pytest

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_cli_subprocess import (
    EnumCliBackendStatus,
    HandlerLlmCliSubprocess,
)


@pytest.mark.unit
def test_handler_llm_cli_subprocess_constructs_with_no_args() -> None:
    """HandlerLlmCliSubprocess must construct with no arguments for auto-wiring."""
    handler = HandlerLlmCliSubprocess()
    assert handler.handler_type == EnumHandlerType.INFRA_HANDLER
    assert handler.handler_category == EnumHandlerTypeCategory.EFFECT
