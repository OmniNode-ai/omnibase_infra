# LLM-Based Business Logic Generation - Implementation Plan

**Status**: 📋 Planning Phase
**Created**: 2025-10-31
**Target**: ONEX v2.0 Code Generation Enhancement
**Repository**: omninode_bridge

## Executive Summary

This document provides a comprehensive implementation plan for enhancing the OmniNode Code Generation System with LLM-powered business logic generation. The goal is to replace template stubs with intelligent, context-aware business logic that is production-ready.

**Key Objectives**:
1. Generate complete business logic implementations (not just stubs)
2. Start with GLM 4.5 via Z.ai as baseline (fast, cost-effective)
3. Prepare architecture for future local model integration (Ollama, vLLM)
4. Track metrics (cost, latency, accuracy, model used)
5. Store learnings and context for continuous improvement

**Success Criteria**:
- Generated business logic passes tests without manual intervention (>80% success rate)
- Generation time: <15s for standard methods (Stage 4 remains within target)
- Cost: <$0.10 per node generation
- Quality: Generated code scores >0.85 on QualityValidator

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Component Specifications](#component-specifications)
3. [Integration Strategy](#integration-strategy)
4. [Implementation Roadmap](#implementation-roadmap)
5. [Testing Approach](#testing-approach)
6. [Success Metrics](#success-metrics)

## Architecture Overview

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    NodeCodegenOrchestrator                            │
│                    (9-Stage Pipeline)                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Stage 4: Code Generation (ENHANCED with LLM)                        │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │                                                              │      │
│  │  TemplateEngine (existing)                                  │      │
│  │         ↓                                                    │      │
│  │  Generate template skeleton (node.py, contract.yaml, etc.) │      │
│  │         ↓                                                    │      │
│  │  NEW: BusinessLogicGenerator                                │      │
│  │    ├─ PromptBuilder: Gather context                        │      │
│  │    │    - PRD requirements                                  │      │
│  │    │    - Contract specification                            │      │
│  │    │    - Similar code patterns (from RAG)                  │      │
│  │    │    - ONEX best practices                               │      │
│  │    │    - Node type specifics                               │      │
│  │    │                                                          │      │
│  │    ├─ NodeLLMEffect: Call LLM API                          │      │
│  │    │    - Tier: CLOUD_FAST (GLM-4.5)                       │      │
│  │    │    - Model: glm-4.5 via Z.ai                          │      │
│  │    │    - Context: 128K tokens                             │      │
│  │    │    - Circuit breaker + retry logic                     │      │
│  │    │    - Response streaming support                        │      │
│  │    │                                                          │      │
│  │    ├─ CodeValidator: Validate generated code                │      │
│  │    │    - AST parsing (Python syntax)                       │      │
│  │    │    - ONEX compliance check                             │      │
│  │    │    - Type hint validation                              │      │
│  │    │    - Security scan (no hardcoded secrets)              │      │
│  │    │                                                          │      │
│  │    ├─ CodeInjector: Insert into template                    │      │
│  │    │    - Replace stub implementations                      │      │
│  │    │    - Preserve template structure                       │      │
│  │    │    - Add imports if needed                             │      │
│  │    │                                                          │      │
│  │    └─ MetricsCollector: Track LLM metrics                   │      │
│  │         - Cost per generation                               │      │
│  │         - Latency (P50, P95, P99)                           │      │
│  │         - Token usage (input/output)                        │      │
│  │         - Model used                                         │      │
│  │         - Success/failure rate                              │      │
│  │                                                              │      │
│  │  Generated Artifacts (enhanced with real business logic)    │      │
│  │                                                              │      │
│  └────────────────────────────────────────────────────────────┘      │
│                                                                        │
│  Intelligence Storage (NEW)                                           │
│  ┌────────────────────────────────────────────────────────────┐      │
│  │  PostgreSQL:                                                │      │
│  │    - llm_generation_metrics (cost, latency, model)         │      │
│  │    - llm_context_windows (model -> max_tokens)             │      │
│  │    - llm_generation_history (prompts, responses, quality)  │      │
│  │    - llm_patterns (learned best practices)                 │      │
│  └────────────────────────────────────────────────────────────┘      │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### Configuration

**API Credentials Location**: `/Volumes/PRO-G40/Code/omniclaude/.env`

The Z.ai API credentials are already configured in the omniclaude project. This file contains:

```bash
# Z.ai API Configuration (GLM-4.5 and GLM-4.6)
ZAI_API_KEY=<your-api-key>           # Line 186
Z_AI_API_KEY=<your-api-key>          # Line 115 (alias)
ZAI_ENDPOINT=https://api.z.ai/api/coding/paas/v4  # Line 116
```

**Required Environment Variables** for this implementation:
- `ZAI_API_KEY` - Z.ai API key for GLM model access
- `ZAI_ENDPOINT` - Z.ai API endpoint (default: `https://api.z.ai/api/coding/paas/v4`)

**Note**: Copy these credentials from the omniclaude `.env` file or reference it directly in your development environment. Do not commit actual API keys to version control.

### 3-Tier LLM System

The system is designed with 3 tiers to support different use cases:

| Tier | Model | Context Window | Use Case | Status |
|------|-------|---------------|----------|--------|
| **LOCAL** | Ollama/vLLM | 8K-16K | Fast, private, no cost | 🚧 Future |
| **CLOUD_FAST** | GLM-4.5 via Z.ai | 128K | Baseline, cost-effective | ✅ Phase 1 |
| **CLOUD_PREMIUM** | GLM-4.6 via Z.ai | 128K | Complex logic, high quality | 🚧 Future |

**Phase 1 Focus**: CLOUD_FAST (GLM-4.5) only
**Phase 2+**: Add LOCAL and CLOUD_PREMIUM tiers

### Data Flow

```
1. TemplateEngine generates skeleton code (node.py with stubs)
2. BusinessLogicGenerator extracts method stubs needing implementation
3. PromptBuilder gathers context for each method:
   - Method signature (args, return type)
   - PRD requirements (what this method should do)
   - Contract specification (performance, error handling)
   - Similar patterns from RAG (if available)
   - ONEX best practices (from intelligence)
4. NodeLLMEffect calls Z.ai GLM API with structured prompt
5. LLM generates business logic implementation
6. CodeValidator validates generated code:
   - AST parsing for syntax
   - ONEX compliance check
   - Type hint validation
   - Security scan
7. CodeInjector replaces stub with generated implementation
8. MetricsCollector records generation metrics
9. Enhanced artifacts returned to Stage 4
```

## Component Specifications

### 1. NodeLLMEffect (Core LLM Caller)

**Purpose**: ONEX Effect node for calling LLM APIs with circuit breaker, retry logic, and metrics.

**Location**: `src/omninode_bridge/nodes/llm_effect/v1_0_0/node.py`

**Type**: Effect Node (external I/O via HTTP)

**Contract**: `ModelContractEffect` with custom input/output models

#### Data Models

```python
# models/model_llm_tier.py
from enum import Enum

class EnumLLMTier(str, Enum):
    """LLM tier selection for different use cases."""

    LOCAL = "local"              # Ollama/vLLM (future)
    CLOUD_FAST = "cloud_fast"    # z.ai (baseline)
    CLOUD_PREMIUM = "cloud_premium"  # Claude Opus (future)


# models/model_llm_request.py
from pydantic import BaseModel, Field
from typing import Optional

class ModelLLMRequest(BaseModel):
    """Request for LLM generation."""

    # Core fields
    prompt: str = Field(..., description="Prompt for LLM generation")
    tier: EnumLLMTier = Field(
        default=EnumLLMTier.CLOUD_FAST,
        description="LLM tier to use"
    )

    # Model configuration
    model_override: Optional[str] = Field(
        None,
        description="Override default model for tier (e.g., 'gpt-4o-mini')"
    )
    max_tokens: int = Field(
        default=2000,
        ge=100,
        le=4096,
        description="Maximum tokens to generate"
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, 1=creative)"
    )

    # Context
    system_prompt: Optional[str] = Field(
        None,
        description="System prompt for instruction tuning"
    )
    context_data: dict = Field(
        default_factory=dict,
        description="Additional context data (PRD, contract, patterns)"
    )

    # Metadata
    correlation_id: Optional[str] = None
    operation_type: str = Field(
        default="code_generation",
        description="Operation type for metrics tracking"
    )


# models/model_llm_response.py
from pydantic import BaseModel, Field
from typing import Optional

class ModelLLMResponse(BaseModel):
    """Response from LLM generation."""

    # Generated content
    generated_text: str = Field(..., description="Generated text from LLM")

    # Metrics
    model_used: str = Field(..., description="Actual model used")
    tier_used: EnumLLMTier = Field(..., description="Tier used")

    # Token usage
    tokens_input: int = Field(..., ge=0, description="Input tokens consumed")
    tokens_output: int = Field(..., ge=0, description="Output tokens generated")
    tokens_total: int = Field(..., ge=0, description="Total tokens used")

    # Performance
    latency_ms: float = Field(..., ge=0, description="Generation latency in milliseconds")
    cost_usd: float = Field(..., ge=0, description="Estimated cost in USD")

    # Quality indicators
    finish_reason: str = Field(..., description="Completion reason (stop, length, error)")
    truncated: bool = Field(default=False, description="Response was truncated")

    # Metadata
    timestamp: str = Field(..., description="Generation timestamp (ISO 8601)")
    correlation_id: Optional[str] = None
```

#### Node Implementation

```python
# node.py
#!/usr/bin/env python3
"""
NodeLLMEffect - ONEX Effect node for LLM API calls.

Supports 3-tier LLM system:
- LOCAL: Ollama/vLLM (future)
- CLOUD_FAST: GLM-4.5 via Z.ai (baseline, Phase 1)
- CLOUD_PREMIUM: GLM-4.6 via Z.ai (future)

Features:
- Circuit breaker pattern for resilience
- Retry logic with exponential backoff
- Response streaming support
- Metrics collection (cost, latency, tokens)
- Context window tracking
"""

import asyncio
import time
from typing import Any, Optional
from uuid import uuid4

import httpx
from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.enums.enum_log_level import EnumLogLevel as LogLevel
from omnibase_core.logging.structured import emit_log_event_sync as emit_log_event
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.core import ModelContainer
from omnibase_core.nodes.node_effect import NodeEffect

from .models.model_llm_request import EnumLLMTier, ModelLLMRequest
from .models.model_llm_response import ModelLLMResponse


class NodeLLMEffect(NodeEffect):
    """
    LLM Effect Node for multi-tier LLM API calls.

    Tier Configuration:
    - LOCAL: Not implemented yet (future Ollama/vLLM support)
    - CLOUD_FAST: GLM-4.5 via Z.ai (128K context)
    - CLOUD_PREMIUM: GLM-4.6 via Z.ai (128K context, future)

    Circuit Breaker:
    - Failure threshold: 5 consecutive failures
    - Recovery timeout: 60 seconds
    - Protected: All external LLM calls

    Retry Policy:
    - Max attempts: 3
    - Backoff: Exponential (1s, 2s, 4s)
    - Retryable errors: Timeout, 429 (rate limit), 5xx (server errors)
    """

    def __init__(self, container: ModelContainer) -> None:
        """Initialize LLM Effect node with tier configuration."""
        super().__init__(container)

        # Configuration from container
        config = container.value if isinstance(container.value, dict) else {}

        # API configuration
        self.zai_api_key = config.get("zai_api_key", "")
        self.zai_base_url = config.get(
            "zai_base_url",
            "https://api.z.ai/api/anthropic"
        )

        # Model configuration per tier
        self.tier_models = {
            EnumLLMTier.LOCAL: None,  # Future: Ollama endpoint
            EnumLLMTier.CLOUD_FAST: "glm-4.5",  # GLM-4.5 via Z.ai
            EnumLLMTier.CLOUD_PREMIUM: "glm-4.6",  # GLM-4.6 via Z.ai (future)
        }

        # Context windows (tokens)
        self.context_windows = {
            EnumLLMTier.LOCAL: 8192,         # Future
            EnumLLMTier.CLOUD_FAST: 128000,  # GLM-4.5 (128K context)
            EnumLLMTier.CLOUD_PREMIUM: 128000,  # GLM-4.6 (128K context)
        }

        # Cost per 1M tokens (USD) - Z.ai GLM pricing
        self.cost_per_1m_input = {
            EnumLLMTier.CLOUD_FAST: 0.20,  # GLM-4.5 pricing
            EnumLLMTier.CLOUD_PREMIUM: 0.30,  # GLM-4.6 pricing (estimate)
        }
        self.cost_per_1m_output = {
            EnumLLMTier.CLOUD_FAST: 0.20,  # GLM-4.5 pricing
            EnumLLMTier.CLOUD_PREMIUM: 0.30,  # GLM-4.6 pricing (estimate)
        }

        # Circuit breaker state
        self.circuit_breaker = {
            "failures": 0,
            "last_failure_time": 0.0,
            "threshold": 5,
            "timeout_seconds": 60,
            "is_open": False,
        }

        # HTTP client (persistent connection pooling)
        self.http_client: Optional[httpx.AsyncClient] = None

        emit_log_event(
            LogLevel.INFO,
            "NodeLLMEffect initialized",
            {
                "node_id": str(self.node_id),
                "supported_tiers": [tier.value for tier in EnumLLMTier],
            },
        )

    async def execute_effect(
        self, contract: ModelContractEffect
    ) -> ModelLLMResponse:
        """
        Execute LLM generation request.

        Args:
            contract: Effect contract with input_state containing ModelLLMRequest

        Returns:
            ModelLLMResponse with generated text and metrics

        Raises:
            ModelOnexError: On LLM API failures or validation errors
        """
        start_time = time.time()

        # Extract request from contract
        request = ModelLLMRequest.model_validate(contract.input_state)
        correlation_id = request.correlation_id or str(uuid4())

        emit_log_event(
            LogLevel.INFO,
            "Executing LLM generation",
            {
                "node_id": str(self.node_id),
                "correlation_id": correlation_id,
                "tier": request.tier.value,
                "operation_type": request.operation_type,
            },
        )

        # Check circuit breaker
        if self._is_circuit_open():
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.SERVICE_UNAVAILABLE,
                message="Circuit breaker is OPEN - LLM service temporarily unavailable",
                details={
                    "failures": self.circuit_breaker["failures"],
                    "threshold": self.circuit_breaker["threshold"],
                },
            )

        try:
            # Initialize HTTP client if needed
            if self.http_client is None:
                self.http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(60.0),
                    limits=httpx.Limits(max_connections=10),
                )

            # Route to appropriate tier
            if request.tier == EnumLLMTier.CLOUD_FAST:
                response = await self._generate_cloud_fast(request, correlation_id)
            elif request.tier == EnumLLMTier.LOCAL:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.NOT_IMPLEMENTED,
                    message="LOCAL tier not implemented yet (future Ollama support)",
                )
            elif request.tier == EnumLLMTier.CLOUD_PREMIUM:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.NOT_IMPLEMENTED,
                    message="CLOUD_PREMIUM tier not implemented yet (future GLM-4.6)",
                )
            else:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    message=f"Unknown LLM tier: {request.tier}",
                )

            # Success - reset circuit breaker
            self._reset_circuit_breaker()

            latency_ms = (time.time() - start_time) * 1000
            emit_log_event(
                LogLevel.INFO,
                "LLM generation completed",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": correlation_id,
                    "latency_ms": latency_ms,
                    "tokens_total": response.tokens_total,
                    "cost_usd": response.cost_usd,
                },
            )

            return response

        except Exception as e:
            # Record failure for circuit breaker
            self._record_failure()

            emit_log_event(
                LogLevel.ERROR,
                f"LLM generation failed: {e}",
                {
                    "node_id": str(self.node_id),
                    "correlation_id": correlation_id,
                    "error": str(e),
                },
            )

            if isinstance(e, ModelOnexError):
                raise

            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"LLM generation failed: {e}",
                details={"original_error": str(e)},
            )

    async def _generate_cloud_fast(
        self, request: ModelLLMRequest, correlation_id: str
    ) -> ModelLLMResponse:
        """
        Generate using CLOUD_FAST tier (GLM-4.5 via Z.ai).

        Implements retry logic with exponential backoff.
        """
        model = request.model_override or self.tier_models[EnumLLMTier.CLOUD_FAST]

        # Build Z.ai request (Anthropic-compatible API)
        zai_request = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": request.system_prompt or "You are a helpful coding assistant.",
                },
                {"role": "user", "content": request.prompt},
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # Retry configuration
        max_attempts = 3
        backoff_seconds = 1

        for attempt in range(1, max_attempts + 1):
            try:
                start_time = time.time()

                response = await self.http_client.post(
                    f"{self.zai_base_url}/chat/completions",
                    json=zai_request,
                    headers={
                        "Authorization": f"Bearer {self.zai_api_key}",
                        "HTTP-Referer": "https://omninode.ai",
                        "X-Title": "OmniNode Bridge",
                    },
                )

                response.raise_for_status()
                data = response.json()

                latency_ms = (time.time() - start_time) * 1000

                # Extract response
                generated_text = data["choices"][0]["message"]["content"]
                finish_reason = data["choices"][0]["finish_reason"]

                # Token usage
                usage = data.get("usage", {})
                tokens_input = usage.get("prompt_tokens", 0)
                tokens_output = usage.get("completion_tokens", 0)
                tokens_total = tokens_input + tokens_output

                # Calculate cost
                cost_input = (
                    tokens_input / 1_000_000
                ) * self.cost_per_1m_input[EnumLLMTier.CLOUD_FAST]
                cost_output = (
                    tokens_output / 1_000_000
                ) * self.cost_per_1m_output[EnumLLMTier.CLOUD_FAST]
                cost_usd = cost_input + cost_output

                return ModelLLMResponse(
                    generated_text=generated_text,
                    model_used=model,
                    tier_used=EnumLLMTier.CLOUD_FAST,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    tokens_total=tokens_total,
                    latency_ms=latency_ms,
                    cost_usd=cost_usd,
                    finish_reason=finish_reason,
                    truncated=finish_reason == "length",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    correlation_id=correlation_id,
                )

            except httpx.HTTPStatusError as e:
                # Retry on rate limits and server errors
                if e.response.status_code in (429, 500, 502, 503, 504):
                    if attempt < max_attempts:
                        emit_log_event(
                            LogLevel.WARNING,
                            f"LLM API error {e.response.status_code}, retrying (attempt {attempt}/{max_attempts})",
                            {"correlation_id": correlation_id},
                        )
                        await asyncio.sleep(backoff_seconds)
                        backoff_seconds *= 2
                        continue

                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.EXTERNAL_SERVICE_ERROR,
                    message=f"Z.ai API error: {e.response.status_code}",
                    details={"response": e.response.text},
                )

            except httpx.TimeoutException:
                if attempt < max_attempts:
                    emit_log_event(
                        LogLevel.WARNING,
                        f"LLM API timeout, retrying (attempt {attempt}/{max_attempts})",
                        {"correlation_id": correlation_id},
                    )
                    await asyncio.sleep(backoff_seconds)
                    backoff_seconds *= 2
                    continue

                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.TIMEOUT,
                    message="Z.ai API timeout after retries",
                )

        # Should not reach here (loop always raises or returns)
        raise ModelOnexError(
            error_code=EnumCoreErrorCode.EXECUTION_ERROR,
            message="LLM generation failed after all retry attempts",
        )

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.circuit_breaker["is_open"]:
            return False

        # Check if timeout has expired
        elapsed = time.time() - self.circuit_breaker["last_failure_time"]
        if elapsed > self.circuit_breaker["timeout_seconds"]:
            # Try to close circuit (half-open state)
            self.circuit_breaker["is_open"] = False
            emit_log_event(
                LogLevel.INFO,
                "Circuit breaker transitioning to HALF-OPEN",
                {"node_id": str(self.node_id)},
            )
            return False

        return True

    def _record_failure(self) -> None:
        """Record a failure and potentially open circuit."""
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure_time"] = time.time()

        if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
            self.circuit_breaker["is_open"] = True
            emit_log_event(
                LogLevel.ERROR,
                "Circuit breaker OPEN - too many LLM failures",
                {
                    "node_id": str(self.node_id),
                    "failures": self.circuit_breaker["failures"],
                },
            )

    def _reset_circuit_breaker(self) -> None:
        """Reset circuit breaker on successful call."""
        if self.circuit_breaker["failures"] > 0:
            emit_log_event(
                LogLevel.INFO,
                "Circuit breaker CLOSED - LLM service recovered",
                {"node_id": str(self.node_id)},
            )

        self.circuit_breaker["failures"] = 0
        self.circuit_breaker["is_open"] = False

    async def shutdown(self) -> None:
        """Clean up HTTP client on shutdown."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        emit_log_event(
            LogLevel.INFO,
            "NodeLLMEffect shut down",
            {"node_id": str(self.node_id)},
        )


__all__ = ["NodeLLMEffect"]
```

### 2. BusinessLogicGenerator (Orchestrator)

**Purpose**: High-level orchestrator that coordinates LLM-based business logic generation.

**Location**: `src/omninode_bridge/codegen/business_logic_generator.py`

**Dependencies**: NodeLLMEffect, PromptBuilder, CodeValidator, CodeInjector

#### Implementation

```python
#!/usr/bin/env python3
"""
Business Logic Generator for ONEX nodes.

Orchestrates LLM-based generation of business logic implementations to replace
template stubs with intelligent, context-aware code.

Pipeline:
1. Extract method stubs from generated templates
2. Build context-rich prompts for each method
3. Call NodeLLMEffect to generate implementation
4. Validate generated code (AST, ONEX compliance, security)
5. Inject validated code back into template
6. Collect metrics for learning

Patterns from omniclaude:
- Pattern detection (CRUD, Aggregation, Transformation)
- Multi-pass generation (if needed for complex logic)
- Quality scoring for generated code
"""

import ast
import logging
import re
from pathlib import Path
from typing import Any, Optional

from omnibase_core.models.core import ModelContainer
from pydantic import BaseModel, Field

from .node_classifier import EnumNodeType
from .prd_analyzer import ModelPRDRequirements
from .template_engine import ModelGeneratedArtifacts
from ..nodes.llm_effect.v1_0_0.node import NodeLLMEffect
from ..nodes.llm_effect.v1_0_0.models.model_llm_request import (
    EnumLLMTier,
    ModelLLMRequest,
)

logger = logging.getLogger(__name__)


class ModelMethodStub(BaseModel):
    """Method stub extracted from template."""

    method_name: str = Field(..., description="Method name (e.g., execute_effect)")
    signature: str = Field(..., description="Full method signature")
    docstring: Optional[str] = Field(None, description="Existing docstring")
    line_number: int = Field(..., ge=1, description="Line number in file")
    needs_implementation: bool = Field(
        default=True, description="Whether stub needs LLM generation"
    )


class ModelBusinessLogicContext(BaseModel):
    """Context for business logic generation."""

    # PRD context
    node_type: str = Field(..., description="Node type (effect/compute/reducer/orchestrator)")
    service_name: str = Field(..., description="Service name")
    business_description: str = Field(..., description="What this node does")
    operations: list[str] = Field(..., description="Operations to implement")
    features: list[str] = Field(..., description="Key features")

    # Method context
    method_name: str = Field(..., description="Method to generate")
    method_signature: str = Field(..., description="Method signature with types")
    method_docstring: Optional[str] = None

    # Code patterns (from RAG)
    similar_patterns: list[str] = Field(
        default_factory=list, description="Similar code patterns from intelligence"
    )
    best_practices: list[str] = Field(
        default_factory=list, description="ONEX best practices"
    )

    # Contract specifics
    performance_requirements: dict = Field(
        default_factory=dict, description="Performance requirements"
    )
    error_handling_patterns: list[str] = Field(
        default_factory=list, description="Error handling patterns"
    )


class ModelGeneratedMethod(BaseModel):
    """Generated method implementation."""

    method_name: str
    generated_code: str = Field(..., description="Generated implementation")

    # Quality metrics
    syntax_valid: bool = Field(..., description="AST parsing succeeded")
    onex_compliant: bool = Field(..., description="ONEX patterns followed")
    has_type_hints: bool = Field(..., description="Type hints present")
    has_docstring: bool = Field(..., description="Docstring present")
    security_issues: list[str] = Field(
        default_factory=list, description="Security issues found"
    )

    # LLM metrics
    tokens_used: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)
    latency_ms: float = Field(..., ge=0.0)
    model_used: str


class ModelEnhancedArtifacts(BaseModel):
    """Enhanced artifacts with LLM-generated business logic."""

    # Original artifacts
    original_artifacts: ModelGeneratedArtifacts

    # Enhanced node file
    enhanced_node_file: str = Field(..., description="Node file with LLM implementations")

    # Generation details
    methods_generated: list[ModelGeneratedMethod] = Field(default_factory=list)
    total_tokens_used: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    generation_success_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class BusinessLogicGenerator:
    """
    Generate intelligent business logic for ONEX nodes using LLMs.

    Replaces template stubs with context-aware implementations.

    Example:
        >>> generator = BusinessLogicGenerator(enable_llm=True)
        >>> enhanced = await generator.enhance_artifacts(
        ...     artifacts=artifacts,
        ...     requirements=requirements,
        ...     context_data={"patterns": ["..."]}
        ... )
        >>> print(f"Generated {len(enhanced.methods_generated)} methods")
    """

    def __init__(
        self,
        enable_llm: bool = True,
        llm_tier: EnumLLMTier = EnumLLMTier.CLOUD_FAST,
        zai_api_key: Optional[str] = None,
    ):
        """
        Initialize business logic generator.

        Args:
            enable_llm: Enable LLM generation (if False, returns original artifacts)
            llm_tier: LLM tier to use (CLOUD_FAST by default)
            zai_api_key: Z.ai API key (required if enable_llm=True)
        """
        self.enable_llm = enable_llm
        self.llm_tier = llm_tier
        self.zai_api_key = zai_api_key

        # Initialize NodeLLMEffect if LLM enabled
        if self.enable_llm:
            if not self.zai_api_key:
                raise ValueError("zai_api_key required when enable_llm=True")

            container = ModelContainer(
                value={"zai_api_key": self.zai_api_key}
            )
            self.llm_node = NodeLLMEffect(container)
        else:
            self.llm_node = None

        logger.info(
            f"BusinessLogicGenerator initialized (enable_llm={enable_llm}, tier={llm_tier.value})"
        )

    async def enhance_artifacts(
        self,
        artifacts: ModelGeneratedArtifacts,
        requirements: ModelPRDRequirements,
        context_data: Optional[dict] = None,
    ) -> ModelEnhancedArtifacts:
        """
        Enhance generated artifacts with LLM business logic.

        Args:
            artifacts: Generated artifacts from TemplateEngine
            requirements: PRD requirements
            context_data: Additional context (patterns, best practices)

        Returns:
            ModelEnhancedArtifacts with enhanced node file and metrics
        """
        if not self.enable_llm:
            logger.info("LLM disabled - returning original artifacts")
            return ModelEnhancedArtifacts(
                original_artifacts=artifacts,
                enhanced_node_file=artifacts.node_file,
            )

        logger.info(
            f"Enhancing artifacts for {artifacts.node_name} with LLM ({self.llm_tier.value})"
        )

        # Step 1: Extract method stubs from node file
        stubs = self._extract_method_stubs(artifacts.node_file, artifacts.node_type)
        logger.info(f"Found {len(stubs)} method stubs to implement")

        # Step 2: Generate implementations for each stub
        generated_methods = []
        enhanced_node_file = artifacts.node_file

        for stub in stubs:
            if not stub.needs_implementation:
                logger.debug(f"Skipping {stub.method_name} (already implemented)")
                continue

            # Build context for this method
            method_context = self._build_method_context(
                stub=stub,
                requirements=requirements,
                node_type=artifacts.node_type,
                context_data=context_data or {},
            )

            # Generate implementation
            generated_method = await self._generate_method_implementation(
                context=method_context
            )
            generated_methods.append(generated_method)

            # Inject into node file
            enhanced_node_file = self._inject_implementation(
                node_file=enhanced_node_file,
                method_name=stub.method_name,
                implementation=generated_method.generated_code,
            )

        # Calculate aggregate metrics
        total_tokens = sum(m.tokens_used for m in generated_methods)
        total_cost = sum(m.cost_usd for m in generated_methods)
        total_latency = sum(m.latency_ms for m in generated_methods)
        success_count = sum(1 for m in generated_methods if m.syntax_valid)
        success_rate = (
            success_count / len(generated_methods) if generated_methods else 1.0
        )

        logger.info(
            f"Enhanced {len(generated_methods)} methods "
            f"(tokens={total_tokens}, cost=${total_cost:.4f}, "
            f"latency={total_latency:.1f}ms, success_rate={success_rate:.1%})"
        )

        return ModelEnhancedArtifacts(
            original_artifacts=artifacts,
            enhanced_node_file=enhanced_node_file,
            methods_generated=generated_methods,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            total_latency_ms=total_latency,
            generation_success_rate=success_rate,
        )

    def _extract_method_stubs(
        self, node_file: str, node_type: str
    ) -> list[ModelMethodStub]:
        """
        Extract method stubs that need implementation.

        Looks for methods with stub comments like:
        - # IMPLEMENTATION REQUIRED
        - # TODO: Implement
        - pass  # Stub
        """
        stubs = []

        try:
            tree = ast.parse(node_file)

            # Find class definition
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Find methods
                    for item in node.body:
                        if isinstance(item, ast.AsyncFunctionDef):
                            # Check if method needs implementation
                            method_source = ast.get_source_segment(node_file, item)

                            # Check for stub indicators
                            needs_impl = (
                                "# IMPLEMENTATION REQUIRED" in method_source
                                or "# TODO:" in method_source
                                or "pass  # Stub" in method_source
                            )

                            if needs_impl:
                                # Extract docstring
                                docstring = ast.get_docstring(item)

                                stubs.append(
                                    ModelMethodStub(
                                        method_name=item.name,
                                        signature=method_source.split("\n")[0],
                                        docstring=docstring,
                                        line_number=item.lineno,
                                        needs_implementation=True,
                                    )
                                )

        except SyntaxError as e:
            logger.error(f"Failed to parse node file: {e}")
            return []

        return stubs

    def _build_method_context(
        self,
        stub: ModelMethodStub,
        requirements: ModelPRDRequirements,
        node_type: str,
        context_data: dict,
    ) -> ModelBusinessLogicContext:
        """Build context for method generation."""
        return ModelBusinessLogicContext(
            node_type=node_type,
            service_name=requirements.service_name,
            business_description=requirements.business_description,
            operations=requirements.operations,
            features=requirements.features,
            method_name=stub.method_name,
            method_signature=stub.signature,
            method_docstring=stub.docstring,
            similar_patterns=context_data.get("patterns", []),
            best_practices=context_data.get("best_practices", []),
            performance_requirements=requirements.performance_requirements,
        )

    async def _generate_method_implementation(
        self, context: ModelBusinessLogicContext
    ) -> ModelGeneratedMethod:
        """
        Generate method implementation using LLM.

        Builds structured prompt with context and calls NodeLLMEffect.
        """
        # Build prompt
        prompt = self._build_generation_prompt(context)

        # Call LLM
        llm_request = ModelLLMRequest(
            prompt=prompt,
            tier=self.llm_tier,
            max_tokens=2000,
            temperature=0.3,  # Lower temp for code generation
            system_prompt=(
                "You are an expert Python developer specializing in ONEX v2.0 node implementations. "
                "Generate production-ready business logic that follows ONEX patterns, includes proper "
                "error handling, type hints, and docstrings. Return ONLY the method body code."
            ),
            operation_type="method_implementation",
        )

        try:
            llm_response = await self.llm_node.execute_effect(
                contract=self._build_llm_contract(llm_request)
            )

            generated_code = llm_response.generated_text.strip()

            # Validate generated code
            validation = self._validate_generated_code(generated_code)

            return ModelGeneratedMethod(
                method_name=context.method_name,
                generated_code=generated_code,
                syntax_valid=validation["syntax_valid"],
                onex_compliant=validation["onex_compliant"],
                has_type_hints=validation["has_type_hints"],
                has_docstring=validation["has_docstring"],
                security_issues=validation["security_issues"],
                tokens_used=llm_response.tokens_total,
                cost_usd=llm_response.cost_usd,
                latency_ms=llm_response.latency_ms,
                model_used=llm_response.model_used,
            )

        except Exception as e:
            logger.error(f"Failed to generate {context.method_name}: {e}")
            # Return stub as fallback
            return ModelGeneratedMethod(
                method_name=context.method_name,
                generated_code="        pass  # Generation failed",
                syntax_valid=True,
                onex_compliant=False,
                has_type_hints=False,
                has_docstring=False,
                security_issues=[f"Generation failed: {e}"],
                tokens_used=0,
                cost_usd=0.0,
                latency_ms=0.0,
                model_used="none",
            )

    def _build_generation_prompt(self, context: ModelBusinessLogicContext) -> str:
        """Build structured prompt for method generation."""
        prompt_parts = [
            f"# Task: Implement {context.method_name} for {context.service_name}",
            "",
            f"## Context",
            f"Node Type: {context.node_type}",
            f"Purpose: {context.business_description}",
            f"Operations: {', '.join(context.operations)}",
            f"Features: {', '.join(context.features)}",
            "",
            f"## Method Signature",
            "```python",
            context.method_signature,
            "```",
        ]

        if context.method_docstring:
            prompt_parts.extend([
                "",
                f"## Docstring",
                context.method_docstring,
            ])

        if context.similar_patterns:
            prompt_parts.extend([
                "",
                f"## Similar Patterns (for reference)",
                *[f"- {pattern}" for pattern in context.similar_patterns[:3]],
            ])

        if context.best_practices:
            prompt_parts.extend([
                "",
                f"## ONEX Best Practices",
                *[f"- {practice}" for practice in context.best_practices[:5]],
            ])

        prompt_parts.extend([
            "",
            f"## Requirements",
            "1. Return ONLY the method body (indented, starting with try/except)",
            "2. Include proper error handling with ModelOnexError",
            "3. Add emit_log_event calls for INFO and ERROR",
            "4. Use type hints for all variables",
            "5. Follow ONEX patterns (correlation tracking, structured logging)",
            "6. No hardcoded secrets or sensitive data",
            "",
            f"Generate the implementation:",
        ])

        return "\n".join(prompt_parts)

    def _validate_generated_code(self, code: str) -> dict:
        """
        Validate generated code for syntax, ONEX compliance, security.

        Returns dict with validation results.
        """
        validation = {
            "syntax_valid": False,
            "onex_compliant": False,
            "has_type_hints": False,
            "has_docstring": False,
            "security_issues": [],
        }

        # AST parsing for syntax
        try:
            ast.parse(code)
            validation["syntax_valid"] = True
        except SyntaxError:
            return validation

        # Check for ONEX patterns
        validation["onex_compliant"] = (
            "ModelOnexError" in code
            and "emit_log_event" in code
        )

        # Check for type hints (basic check)
        validation["has_type_hints"] = "->" in code or ": " in code

        # Security checks
        security_patterns = [
            (r'password\s*=\s*["\']', "Hardcoded password detected"),
            (r'api_key\s*=\s*["\']', "Hardcoded API key detected"),
            (r'secret\s*=\s*["\']', "Hardcoded secret detected"),
        ]

        for pattern, message in security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                validation["security_issues"].append(message)

        return validation

    def _inject_implementation(
        self, node_file: str, method_name: str, implementation: str
    ) -> str:
        """
        Inject generated implementation into node file.

        Replaces stub marker with actual implementation.
        """
        # Find method stub and replace
        # Look for: "# IMPLEMENTATION REQUIRED" or "pass  # Stub"

        # Simple regex replacement (can be improved with AST manipulation)
        stub_pattern = (
            rf"(async def {method_name}\(.*?\):.*?\n"
            r"(?:        \"\"\".*?\"\"\"\n)?"
            r"        # IMPLEMENTATION REQUIRED.*?\n"
            r"        .*?pass)"
        )

        replacement = f"\\1\n{implementation}"

        enhanced = re.sub(stub_pattern, replacement, node_file, flags=re.DOTALL)

        return enhanced

    def _build_llm_contract(self, request: ModelLLMRequest):
        """Build ModelContractEffect for LLM call."""
        from omnibase_core.models.contracts.model_contract_effect import (
            ModelContractEffect,
        )

        return ModelContractEffect(
            name="llm_generation",
            version={"major": 1, "minor": 0, "patch": 0},
            node_type="EFFECT",
            input_state=request.model_dump(),
            io_operations=[{"operation_type": "http_request"}],
        )


__all__ = ["BusinessLogicGenerator", "ModelEnhancedArtifacts", "ModelBusinessLogicContext"]
```

### 3. Integration with TemplateEngine

**Location**: Modify `src/omninode_bridge/codegen/template_engine.py`

#### Enhanced `generate()` Method

```python
# In template_engine.py, modify generate() method

async def generate(
    self,
    requirements: ModelPRDRequirements,
    classification: ModelClassificationResult,
    output_directory: Path,
    run_tests: bool = False,
    strict_mode: bool = False,
    enable_llm: bool = False,  # NEW: Enable LLM business logic generation
    llm_tier: EnumLLMTier = EnumLLMTier.CLOUD_FAST,  # NEW
    zai_api_key: Optional[str] = None,  # NEW
) -> ModelGeneratedArtifacts:
    """
    Generate complete node implementation from templates.

    NEW in Phase 3: LLM-based business logic generation.

    Args:
        requirements: Extracted PRD requirements
        classification: Node type classification
        output_directory: Target directory for generated files
        run_tests: Execute generated tests after code generation
        strict_mode: Raise exception if tests fail (default: attach to artifacts)
        enable_llm: Enable LLM business logic generation (replaces stubs)
        llm_tier: LLM tier to use (CLOUD_FAST by default)
        zai_api_key: Z.ai API key (required if enable_llm=True)

    Returns:
        ModelGeneratedArtifacts with all generated code and test results

    Raises:
        ValueError: If strict_mode=True and tests fail
    """
    # ... existing code ...

    # Create artifacts (before test execution)
    artifacts = ModelGeneratedArtifacts(
        node_file=node_content,
        contract_file=contract_content,
        init_file=init_content,
        models=models,
        tests=tests,
        documentation=documentation,
        node_type=classification.node_type.value,
        node_name=context["node_class_name"],
        service_name=context["service_name"],
        output_directory=output_directory,
    )

    # NEW: LLM business logic generation
    if enable_llm:
        logger.info(f"Enhancing {artifacts.node_name} with LLM business logic")
        try:
            from .business_logic_generator import BusinessLogicGenerator

            generator = BusinessLogicGenerator(
                enable_llm=True,
                llm_tier=llm_tier,
                zai_api_key=zai_api_key,
            )

            enhanced = await generator.enhance_artifacts(
                artifacts=artifacts,
                requirements=requirements,
                context_data={
                    "patterns": requirements.code_examples,
                    "best_practices": requirements.best_practices,
                },
            )

            # Replace node_file with enhanced version
            artifacts.node_file = enhanced.enhanced_node_file

            logger.info(
                f"LLM enhancement complete: "
                f"{len(enhanced.methods_generated)} methods, "
                f"${enhanced.total_cost_usd:.4f} cost, "
                f"{enhanced.generation_success_rate:.1%} success rate"
            )

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            # Continue with stub implementations if LLM fails

    # Execute tests if requested
    if run_tests:
        # ... existing test execution code ...

    return artifacts
```

### 4. Metrics Collection & Storage

**Purpose**: Track LLM usage metrics and store learnings for continuous improvement.

**Location**: `src/omninode_bridge/codegen/llm_metrics_collector.py`

#### Database Schema (PostgreSQL)

```sql
-- Add to migrations/schema.sql

-- LLM generation metrics
CREATE TABLE IF NOT EXISTS llm_generation_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL,
    session_id UUID,  -- Links to codegen session

    -- Context
    node_type VARCHAR(50) NOT NULL,  -- effect, compute, reducer, orchestrator
    service_name VARCHAR(255) NOT NULL,
    method_name VARCHAR(255) NOT NULL,
    operation_type VARCHAR(100),  -- method_implementation, requirement_extraction, etc.

    -- LLM details
    tier VARCHAR(50) NOT NULL,  -- local, cloud_fast, cloud_premium
    model_used VARCHAR(100) NOT NULL,

    -- Token usage
    tokens_input INTEGER NOT NULL CHECK (tokens_input >= 0),
    tokens_output INTEGER NOT NULL CHECK (tokens_output >= 0),
    tokens_total INTEGER NOT NULL CHECK (tokens_total >= 0),

    -- Performance
    latency_ms FLOAT NOT NULL CHECK (latency_ms >= 0),
    cost_usd NUMERIC(10, 6) NOT NULL CHECK (cost_usd >= 0),

    -- Quality
    syntax_valid BOOLEAN NOT NULL DEFAULT FALSE,
    onex_compliant BOOLEAN NOT NULL DEFAULT FALSE,
    has_type_hints BOOLEAN NOT NULL DEFAULT FALSE,
    security_issues JSONB DEFAULT '[]'::jsonb,

    -- Metadata
    finish_reason VARCHAR(50),  -- stop, length, error
    truncated BOOLEAN DEFAULT FALSE,
    prompt_length INTEGER,
    response_length INTEGER,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes
    INDEX idx_llm_metrics_correlation (correlation_id),
    INDEX idx_llm_metrics_session (session_id),
    INDEX idx_llm_metrics_node_type (node_type),
    INDEX idx_llm_metrics_model (model_used),
    INDEX idx_llm_metrics_created (created_at DESC)
);

-- LLM context windows (model capabilities)
CREATE TABLE IF NOT EXISTS llm_context_windows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tier VARCHAR(50) NOT NULL UNIQUE,
    model_name VARCHAR(100) NOT NULL,
    max_tokens INTEGER NOT NULL CHECK (max_tokens > 0),
    cost_per_1m_input_usd NUMERIC(10, 4),
    cost_per_1m_output_usd NUMERIC(10, 4),
    supports_streaming BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default context windows
INSERT INTO llm_context_windows (tier, model_name, max_tokens, cost_per_1m_input_usd, cost_per_1m_output_usd, supports_streaming)
VALUES
    ('cloud_fast', 'glm-4.5', 128000, 0.20, 0.20, TRUE),
    ('local', 'ollama/codellama:7b', 16384, 0.00, 0.00, TRUE),
    ('cloud_premium', 'glm-4.6', 128000, 0.30, 0.30, TRUE)
ON CONFLICT (tier) DO NOTHING;

-- LLM generation history (for learning)
CREATE TABLE IF NOT EXISTS llm_generation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    correlation_id UUID NOT NULL,

    -- Context
    node_type VARCHAR(50) NOT NULL,
    method_name VARCHAR(255) NOT NULL,

    -- Prompt and response
    prompt_text TEXT NOT NULL,
    generated_code TEXT NOT NULL,

    -- Quality scores
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    validation_passed BOOLEAN NOT NULL,

    -- Feedback (for learning)
    user_accepted BOOLEAN,
    user_feedback TEXT,
    manual_edits_required BOOLEAN,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    INDEX idx_llm_history_correlation (correlation_id),
    INDEX idx_llm_history_quality (quality_score DESC)
);
```

#### Metrics Collector Implementation

```python
#!/usr/bin/env python3
"""
LLM Metrics Collector.

Tracks LLM usage metrics for cost analysis, performance monitoring,
and continuous learning.
"""

import logging
from typing import Optional
from uuid import UUID

import asyncpg
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelLLMMetrics(BaseModel):
    """LLM generation metrics for storage."""

    correlation_id: UUID
    session_id: Optional[UUID] = None

    # Context
    node_type: str
    service_name: str
    method_name: str
    operation_type: str = "method_implementation"

    # LLM details
    tier: str
    model_used: str

    # Token usage
    tokens_input: int = Field(..., ge=0)
    tokens_output: int = Field(..., ge=0)
    tokens_total: int = Field(..., ge=0)

    # Performance
    latency_ms: float = Field(..., ge=0.0)
    cost_usd: float = Field(..., ge=0.0)

    # Quality
    syntax_valid: bool = False
    onex_compliant: bool = False
    has_type_hints: bool = False
    security_issues: list[str] = Field(default_factory=list)

    # Metadata
    finish_reason: str = "stop"
    truncated: bool = False
    prompt_length: Optional[int] = None
    response_length: Optional[int] = None


class LLMMetricsCollector:
    """
    Collect and store LLM generation metrics.

    Supports:
    - Cost tracking
    - Performance analysis
    - Quality scoring
    - Learning from successful patterns
    """

    def __init__(self, postgres_url: str):
        """
        Initialize metrics collector.

        Args:
            postgres_url: PostgreSQL connection URL
        """
        self.postgres_url = postgres_url
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish PostgreSQL connection pool."""
        if not self.pool:
            self.pool = await asyncpg.create_pool(self.postgres_url, min_size=2, max_size=10)
            logger.info("LLM metrics collector connected to PostgreSQL")

    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def record_generation(self, metrics: ModelLLMMetrics) -> None:
        """
        Record LLM generation metrics to database.

        Args:
            metrics: LLM metrics to store
        """
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO llm_generation_metrics (
                    correlation_id, session_id, node_type, service_name, method_name,
                    operation_type, tier, model_used, tokens_input, tokens_output, tokens_total,
                    latency_ms, cost_usd, syntax_valid, onex_compliant, has_type_hints,
                    security_issues, finish_reason, truncated, prompt_length, response_length
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
                """,
                metrics.correlation_id,
                metrics.session_id,
                metrics.node_type,
                metrics.service_name,
                metrics.method_name,
                metrics.operation_type,
                metrics.tier,
                metrics.model_used,
                metrics.tokens_input,
                metrics.tokens_output,
                metrics.tokens_total,
                metrics.latency_ms,
                metrics.cost_usd,
                metrics.syntax_valid,
                metrics.onex_compliant,
                metrics.has_type_hints,
                metrics.security_issues,
                metrics.finish_reason,
                metrics.truncated,
                metrics.prompt_length,
                metrics.response_length,
            )

        logger.info(
            f"Recorded LLM metrics: {metrics.method_name} "
            f"(tokens={metrics.tokens_total}, cost=${metrics.cost_usd:.4f})"
        )

    async def get_aggregate_metrics(
        self, node_type: Optional[str] = None, days: int = 30
    ) -> dict:
        """
        Get aggregate metrics for cost/performance analysis.

        Args:
            node_type: Filter by node type (optional)
            days: Number of days to look back

        Returns:
            Dict with aggregate metrics
        """
        if not self.pool:
            await self.connect()

        async with self.pool.acquire() as conn:
            query = """
                SELECT
                    COUNT(*) as generation_count,
                    SUM(tokens_total) as total_tokens,
                    SUM(cost_usd) as total_cost_usd,
                    AVG(latency_ms) as avg_latency_ms,
                    AVG(CASE WHEN syntax_valid THEN 1 ELSE 0 END) as syntax_valid_rate,
                    AVG(CASE WHEN onex_compliant THEN 1 ELSE 0 END) as onex_compliant_rate
                FROM llm_generation_metrics
                WHERE created_at >= NOW() - INTERVAL '1 day' * $1
            """

            params = [days]

            if node_type:
                query += " AND node_type = $2"
                params.append(node_type)

            row = await conn.fetchrow(query, *params)

            return {
                "generation_count": row["generation_count"],
                "total_tokens": row["total_tokens"],
                "total_cost_usd": float(row["total_cost_usd"] or 0),
                "avg_latency_ms": float(row["avg_latency_ms"] or 0),
                "syntax_valid_rate": float(row["syntax_valid_rate"] or 0),
                "onex_compliant_rate": float(row["onex_compliant_rate"] or 0),
            }


__all__ = ["LLMMetricsCollector", "ModelLLMMetrics"]
```

## Integration Strategy

### Phase 1: GLM-4.5 Baseline (Weeks 1-2)

**Goal**: Get LLM business logic generation working with GLM-4.5 via Z.ai.

**Components**:
1. ✅ NodeLLMEffect (CLOUD_FAST tier only)
2. ✅ BusinessLogicGenerator (basic orchestration)
3. ✅ PromptBuilder (embedded in BusinessLogicGenerator)
4. ✅ CodeValidator (basic AST + security checks)
5. ✅ CodeInjector (regex-based stub replacement)
6. ✅ LLMMetricsCollector (PostgreSQL storage)

**Integration Points**:
- Modify `TemplateEngine.generate()` to accept `enable_llm` flag
- Call `BusinessLogicGenerator.enhance_artifacts()` after template generation
- Store metrics in PostgreSQL

**Success Criteria**:
- Generated methods compile without syntax errors (>95%)
- ONEX compliance score >0.8
- Generation latency <15s per node
- Cost <$0.10 per node

### Phase 2: Local Models (Weeks 3-4)

**Goal**: Add LOCAL tier support for fast, cost-free generation.

**New Components**:
1. Ollama client integration in NodeLLMEffect
2. Model selection logic (fall back to CLOUD_FAST if local unavailable)
3. Context window optimization for smaller models

**Success Criteria**:
- Local generation works offline
- Latency <5s per method (local models are fast)
- Quality >0.7 (acceptable trade-off for speed/cost)

### Phase 3: Premium Models & Learning (Weeks 5-6)

**Goal**: Add CLOUD_PREMIUM tier and intelligence learning.

**New Components**:
1. GLM-4.6 integration (via Z.ai)
2. Intelligence storage and retrieval
3. Pattern learning from successful generations
4. Multi-pass generation for complex logic

**Success Criteria**:
- Premium tier quality >0.9
- System learns from successful patterns
- Continuous improvement visible in metrics

## Implementation Roadmap

### Week 1: Core Infrastructure

**Sequential Tasks**:
1. **Day 1-2**: Implement NodeLLMEffect
   - Data models (ModelLLMRequest, ModelLLMResponse)
   - CLOUD_FAST tier (z.ai integration)
   - Circuit breaker + retry logic
   - Unit tests

2. **Day 3-4**: Implement BusinessLogicGenerator
   - Stub extraction (AST-based)
   - Context building
   - LLM orchestration
   - Code injection
   - Unit tests

3. **Day 5**: Database schema and metrics
   - PostgreSQL migrations
   - LLMMetricsCollector implementation
   - Integration tests

**Parallel Tasks** (can be done by different devs):
- Documentation writing (this plan → implementation guide)
- CI/CD setup for LLM testing
- Cost monitoring dashboard design

### Week 2: Integration & Testing

**Sequential Tasks**:
1. **Day 1-2**: TemplateEngine integration
   - Add `enable_llm` flag to `generate()`
   - Call BusinessLogicGenerator
   - Handle errors gracefully
   - Integration tests

2. **Day 3-4**: End-to-end testing
   - Generate test nodes with LLM
   - Validate quality scores
   - Measure cost and latency
   - Fix issues

3. **Day 5**: Documentation and examples
   - Update CODE_GENERATION_GUIDE.md
   - Add examples/llm_codegen_example.py
   - CLI flag `--enable-llm`

**Parallel Tasks**:
- Performance benchmarking
- Security review (prompt injection, code validation)
- Cost analysis and optimization

### Week 3-4: Local Models (Phase 2)

**Sequential Tasks**:
1. LOCAL tier implementation in NodeLLMEffect
2. Ollama client integration
3. Model selection logic (tier fallback)
4. Testing with codellama:7b
5. Documentation updates

### Week 5-6: Premium & Learning (Phase 3)

**Sequential Tasks**:
1. CLOUD_PREMIUM tier (GLM-4.6)
2. Intelligence storage schema
3. Pattern learning from successful generations
4. Multi-pass generation
5. A/B testing framework

## Testing Approach

### Unit Tests

**NodeLLMEffect**:
```python
# tests/unit/nodes/test_llm_effect.py

@pytest.mark.asyncio
async def test_llm_effect_cloud_fast_generation():
    """Test CLOUD_FAST generation with GLM-4.5."""
    container = ModelContainer(value={"zai_api_key": os.getenv("ZAI_API_KEY")})
    node = NodeLLMEffect(container)

    request = ModelLLMRequest(
        prompt="Write a Python function to calculate fibonacci(n)",
        tier=EnumLLMTier.CLOUD_FAST,
        max_tokens=500,
    )

    contract = build_contract(request)
    response = await node.execute_effect(contract)

    assert response.generated_text
    assert response.model_used == "glm-4.5"
    assert response.tokens_total > 0
    assert response.cost_usd > 0
    assert response.finish_reason in ("stop", "length")

@pytest.mark.asyncio
async def test_llm_effect_circuit_breaker():
    """Test circuit breaker opens after failures."""
    # Mock LLM to always fail
    # Verify circuit opens after threshold failures
    pass
```

**BusinessLogicGenerator**:
```python
# tests/unit/codegen/test_business_logic_generator.py

@pytest.mark.asyncio
async def test_extract_method_stubs():
    """Test extraction of method stubs from template."""
    generator = BusinessLogicGenerator(enable_llm=False)

    node_file = '''
    class NodeTestEffect(NodeEffect):
        async def execute_effect(self, contract):
            """Execute effect."""
            # IMPLEMENTATION REQUIRED: Add effect logic here
            pass
    '''

    stubs = generator._extract_method_stubs(node_file, "effect")

    assert len(stubs) == 1
    assert stubs[0].method_name == "execute_effect"
    assert stubs[0].needs_implementation == True

@pytest.mark.asyncio
async def test_business_logic_generation_integration():
    """Test end-to-end business logic generation."""
    # Full integration test with real LLM
    # Requires API key in environment
    pass
```

### Integration Tests

```python
# tests/integration/test_llm_codegen_e2e.py

@pytest.mark.integration
@pytest.mark.asyncio
async def test_codegen_with_llm_enhancement():
    """Test complete code generation with LLM enhancement."""

    # Analyze prompt
    analyzer = PRDAnalyzer()
    requirements = await analyzer.analyze_prompt(
        "Create PostgreSQL CRUD Effect node with connection pooling"
    )

    # Classify
    classifier = NodeClassifier()
    classification = classifier.classify(requirements)

    # Generate with LLM
    engine = TemplateEngine(enable_inline_templates=True)
    artifacts = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=Path("./test_output"),
        enable_llm=True,
        llm_tier=EnumLLMTier.CLOUD_FAST,
        zai_api_key=os.getenv("ZAI_API_KEY"),
    )

    # Verify LLM enhanced code
    assert "# IMPLEMENTATION REQUIRED" not in artifacts.node_file
    assert "pass  # Stub" not in artifacts.node_file

    # Validate quality
    validator = QualityValidator()
    validation = await validator.validate(artifacts)
    assert validation.quality_score >= 0.85

    # Verify executable
    ast.parse(artifacts.node_file)  # Should not raise SyntaxError
```

### Performance Tests

```python
# tests/performance/test_llm_generation_performance.py

@pytest.mark.performance
@pytest.mark.asyncio
async def test_llm_generation_latency():
    """Measure LLM generation latency."""
    metrics = []

    for i in range(10):
        start = time.time()
        # Generate method
        latency = (time.time() - start) * 1000
        metrics.append(latency)

    p50 = percentile(metrics, 50)
    p95 = percentile(metrics, 95)

    assert p50 < 5000  # 5s
    assert p95 < 15000  # 15s

@pytest.mark.performance
@pytest.mark.asyncio
async def test_llm_generation_cost():
    """Measure LLM generation cost."""
    costs = []

    for i in range(10):
        # Generate method
        costs.append(response.cost_usd)

    avg_cost = sum(costs) / len(costs)
    total_cost_per_node = avg_cost * 3  # Assume 3 methods per node

    assert total_cost_per_node < 0.10  # $0.10 per node
```

## Success Metrics

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Generation Success Rate** | >80% | Methods that compile without syntax errors |
| **ONEX Compliance** | >0.85 | QualityValidator ONEX compliance score |
| **Generation Latency** | <15s | Time to generate all methods in a node |
| **Cost per Node** | <$0.10 | Total LLM cost for one node generation |
| **Test Pass Rate** | >70% | Generated tests that pass without manual fixes |

### Quality Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Syntax Valid** | >95% | AST parsing success rate |
| **Type Hints** | >80% | Generated methods with complete type hints |
| **Error Handling** | 100% | All methods use ModelOnexError |
| **Logging** | 100% | All methods use emit_log_event |
| **Security Issues** | 0 | No hardcoded secrets detected |

### Business Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Developer Time Saved** | >60% | Time to production-ready node vs manual |
| **Code Quality Improvement** | >30% | QualityValidator score vs template stubs |
| **Learning Rate** | Improving | Quality scores improve over time |

### Tracking Dashboard

Create Grafana dashboard with:
- LLM usage over time (requests/day)
- Cost tracking (daily, weekly, monthly)
- Latency percentiles (P50, P95, P99)
- Quality scores distribution
- Success rate trends
- Model comparison (LOCAL vs CLOUD_FAST vs CLOUD_PREMIUM)

## Next Steps

1. **Review this plan** with team
2. **Configure API credentials**:
   - API credentials are already available in `/Volumes/PRO-G40/Code/omniclaude/.env`
   - Copy `ZAI_API_KEY` and `ZAI_ENDPOINT` to your environment
   - Or reference the omniclaude `.env` file directly during development
3. **Create Jira tickets** from roadmap
4. **Set up development branch**: `feature/llm-business-logic`
5. **Start Week 1, Day 1**: Implement NodeLLMEffect

## References

**Related Documentation**:
- [CODE_GENERATION_GUIDE.md](../guides/CODE_GENERATION_GUIDE.md) - Current codegen system
- [ONEX Architecture](../../Archon/docs/ONEX_ARCHITECTURE_PATTERNS_COMPLETE.md) - ONEX v2.0 patterns
- [omniclaude](https://github.com/your-org/omniclaude) - BusinessLogicGenerator patterns
- [omniagent](https://github.com/your-org/omniagent) - Smart Responder Chain, Ollama integration

**External Resources**:
- [Z.ai API Documentation](https://z.ai/docs) - GLM model access
- [Ollama Documentation](https://ollama.com/docs) - Local model deployment
- [GLM-4 Technical Report](https://github.com/THUDM/GLM-4) - Model architecture and capabilities

---

**Document Status**: ✅ Ready for Implementation
**Last Updated**: 2025-10-31
**Next Review**: After Phase 1 completion
