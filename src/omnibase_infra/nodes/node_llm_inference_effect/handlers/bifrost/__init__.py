# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Bifrost LLM gateway handler package.

Provides config-driven routing, failover, retry, and circuit-breaking
across multiple local LLM backend endpoints. Includes shadow mode for
running learned routing policies in parallel (OMN-5570).

Related:
    - OMN-2736: Adopt bifrost as LLM gateway handler for delegated task routing
    - OMN-5570: Shadow Mode + Comparison Dashboard
"""

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config.bifrost_shadow import (
    ModelBifrostShadowConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.config.model_shadow_decision_log import (
    ModelShadowDecisionLog,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.handler_bifrost_gateway import (
    HandlerBifrostGateway,
    ProtocolShadowPolicy,
    ShadowDecisionCallback,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
    ModelBifrostConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_request import (
    ModelBifrostRequest,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_response import (
    ModelBifrostResponse,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_routing_rule import (
    ModelBifrostRoutingRule,
)

__all__: list[str] = [
    "HandlerBifrostGateway",
    "ModelBifrostBackendConfig",
    "ModelBifrostConfig",
    "ModelBifrostRequest",
    "ModelBifrostResponse",
    "ModelBifrostRoutingRule",
    "ModelBifrostShadowConfig",
    "ModelShadowDecisionLog",
    "ProtocolShadowPolicy",
    "ShadowDecisionCallback",
]
