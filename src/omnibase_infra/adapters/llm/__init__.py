# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""LLM protocol adapter implementations.

This package provides concrete implementations of the SPI LLM protocols
(ProtocolLLMProvider, ProtocolModelRouter, ProtocolLLMToolProvider) that
wrap the existing infra-layer handlers (HandlerLlmOpenaiCompatible,
HandlerLlmOllama) behind the protocol interfaces, enabling container-based
dependency injection for LLM services.

Related Tickets:
    - OMN-2319: Implement SPI LLM protocol adapters
"""

from omnibase_infra.adapters.llm.adapter_code_analysis_enrichment import (
    AdapterCodeAnalysisEnrichment,
)
from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    AdapterLlmProviderOpenai,
)
from omnibase_infra.adapters.llm.adapter_llm_tool_provider import (
    AdapterLlmToolProvider,
)
from omnibase_infra.adapters.llm.adapter_model_router import AdapterModelRouter
from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)
from omnibase_infra.adapters.llm.model_llm_health_response import (
    ModelLlmHealthResponse,
)
from omnibase_infra.adapters.llm.model_llm_model_capabilities import (
    ModelLlmModelCapabilities,
)
from omnibase_infra.adapters.llm.model_llm_provider_config import (
    ModelLlmProviderConfig,
)

__all__: list[str] = [
    "AdapterLlmProviderOpenai",
    "AdapterLlmToolProvider",
    "AdapterModelRouter",
    "AdapterCodeAnalysisEnrichment",
    "ModelLlmAdapterRequest",
    "ModelLlmAdapterResponse",
    "ModelLlmHealthResponse",
    "ModelLlmModelCapabilities",
    "ModelLlmProviderConfig",
]
