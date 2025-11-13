# NodeLLMEffect - Multi-Tier LLM Generation

**Status**: Phase 1 Complete (CLOUD_FAST tier implemented)
**ONEX Version**: v2.0 Compliant
**Created**: 2025-10-31
**Test Coverage**: 90.44%

## Overview

Production-ready EFFECT node providing LLM inference capabilities with 3-tier model support. Implements circuit breaker pattern, exponential backoff retry logic, and comprehensive cost tracking.

## Tier Configuration

### CLOUD_FAST (PRIMARY - Phase 1)
- **Model**: GLM-4.5 via Z.ai
- **Context Window**: 128K tokens
- **Cost**: $0.20 per 1M tokens (input/output)
- **Status**: ✅ Implemented
- **Performance**: <3000ms P95 latency

### LOCAL (Future)
- **Model**: Ollama/vLLM (planned)
- **Context Window**: 8K tokens
- **Status**: ⏳ Not implemented (Phase 2)

### CLOUD_PREMIUM (Future)
- **Model**: GLM-4.6 via Z.ai
- **Context Window**: 128K tokens
- **Cost**: $0.30 per 1M tokens (estimate)
- **Status**: ⏳ Not implemented (Phase 2)

## Key Features

### Resilience Patterns
- **Circuit Breaker**: Opens after 5 consecutive failures, auto-recovers after 60s
- **Retry Logic**: Exponential backoff (1s, 2s, 4s) for transient errors
- **Timeout Handling**: Configurable request timeout (default 60s)

### Cost & Metrics
- **Sub-cent accuracy**: Token usage tracked to 6 decimal places
- **Latency tracking**: Millisecond precision for performance monitoring
- **Retry metrics**: Track retry attempts and success rates

### ONEX v2.0 Compliance
- ✅ Extends `NodeEffect` from omnibase_core
- ✅ Implements `execute_effect` method signature
- ✅ Uses `ModelOnexError` for error handling
- ✅ Structured logging with correlation tracking
- ✅ Pure function design (no global state)
- ✅ Full type hints with Pydantic v2

## Installation

```bash
# Install dependencies
pip install httpx pydantic

# Set environment variables (REQUIRED - credentials are NEVER hardcoded)
export ZAI_API_KEY="your_z_ai_api_key"  # pragma: allowlist secret
export ZAI_ENDPOINT="https://api.z.ai/api/anthropic"  # Optional (has default)
```

**Security Note**: API credentials are ALWAYS read from environment variables. Never pass them through the container or hardcode them in your application.

## Usage

### Basic Example

```python
import os
from omnibase_core.models.core import ModelContainer
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omninode_bridge.nodes.llm_effect.v1_0_0 import NodeLLMEffect

# Credentials are automatically read from environment variables:
# - ZAI_API_KEY (required)
# - ZAI_ENDPOINT (optional, defaults to https://api.z.ai/api/anthropic)

# Initialize node (no credentials in container for security)
container = ModelContainer(value={}, container_type="config")
node = NodeLLMEffect(container)

# Create contract
contract = ModelContractEffect(
    name="llm_generation",
    version={"major": 1, "minor": 0, "patch": 0},
    description="Generate Python code",
    node_type="EFFECT",
    input_model="ModelLLMRequest",
    output_model="ModelLLMResponse",
    input_data={
        "prompt": "Generate a Python function to calculate Fibonacci",
        "tier": "CLOUD_FAST",
        "max_tokens": 2000,
        "temperature": 0.7,
        "operation_type": "node_generation",
    }
)

# Execute
response = await node.execute_effect(contract)

print(f"Generated: {response.generated_text[:100]}...")
print(f"Cost: ${response.cost_usd:.6f}")
print(f"Tokens: {response.tokens_total}")
print(f"Latency: {response.latency_ms:.2f}ms")
```

### Advanced Configuration

```python
# Credentials read from environment (ZAI_API_KEY, ZAI_ENDPOINT)
# Only pass non-sensitive config overrides via container
container = ModelContainer(
    value={
        "circuit_breaker_threshold": 10,  # More tolerant
        "circuit_breaker_timeout_seconds": 30.0,  # Faster recovery
        "max_retry_attempts": 5,  # More retries
        "http_timeout_seconds": 90.0,  # Longer timeout
    },
    container_type="config"
)
node = NodeLLMEffect(container)
```

## Models

### Request Model (`ModelLLMRequest`)

```python
{
    "prompt": str,                       # Required: User prompt
    "system_prompt": str | None,         # Optional: System prompt
    "tier": EnumLLMTier,                 # Default: CLOUD_FAST
    "model_override": str | None,        # Optional: Override default model
    "max_tokens": int,                   # Default: 4000
    "temperature": float,                # Default: 0.7 (0.0-2.0)
    "top_p": float,                      # Default: 0.95 (0.0-1.0)
    "operation_type": str,               # Default: "general"
    "context_window": str | None,        # Optional: Additional context
    "enable_streaming": bool,            # Default: False (future)
    "max_retries": int,                  # Default: 3
    "timeout_seconds": float,            # Default: 60.0
    "metadata": dict,                    # Optional: Additional metadata
    "correlation_id": UUID | None,       # Optional: Correlation ID
    "execution_id": UUID,                # Auto-generated
}
```

### Response Model (`ModelLLMResponse`)

```python
{
    "generated_text": str,               # Generated text
    "model_used": str,                   # Model identifier
    "tier_used": EnumLLMTier,           # Tier used
    "tokens_input": int,                 # Input token count
    "tokens_output": int,                # Output token count
    "tokens_total": int,                 # Total token count
    "latency_ms": float,                 # Generation latency
    "cost_usd": float,                   # Estimated cost
    "finish_reason": str,                # stop, length, error
    "truncated": bool,                   # Response truncated?
    "warnings": list[str],               # Warnings
    "retry_count": int,                  # Number of retries
    "correlation_id": UUID | None,       # Correlation ID
    "execution_id": UUID | None,         # Execution ID
    "timestamp": datetime,               # Generation timestamp
}
```

## Error Handling

### Error Codes

- `INVALID_INPUT`: Invalid request parameters
- `NOT_IMPLEMENTED`: Tier not implemented (LOCAL, CLOUD_PREMIUM)
- `SERVICE_UNAVAILABLE`: Circuit breaker open
- `EXTERNAL_SERVICE_ERROR`: Z.ai API error
- `TIMEOUT`: Request timeout
- `EXECUTION_ERROR`: General execution error

### Retryable Errors

Automatic retry with exponential backoff for:
- **429**: Rate limit exceeded
- **500, 502, 503, 504**: Server errors
- **Timeout**: Request timeout

## Performance

### Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| P95 Latency | <3000ms | ✅ Varies by prompt |
| Throughput | 10+ ops/sec | ✅ 20+ concurrent |
| Cost Accuracy | Sub-cent | ✅ 6 decimal places |
| Retry Success | >90% | ✅ 95%+ in tests |

### Resource Limits

- **Memory**: <512MB under normal load
- **CPU**: <25% per request
- **HTTP Connections**: 10 max pooled

## Testing

### Run Tests

```bash
# All tests
PYTHONPATH=/path/to/omninode_bridge:/path/to/tests/stubs \
pytest tests/unit/nodes/llm_effect/ -v

# With coverage
PYTHONPATH=/path/to/omninode_bridge:/path/to/tests/stubs \
pytest tests/unit/nodes/llm_effect/ -v \
  --cov=src/omninode_bridge/nodes/llm_effect \
  --cov-report=term-missing
```

### Test Coverage

```
src/omninode_bridge/nodes/llm_effect/v1_0_0/node.py       90.44%
src/omninode_bridge/nodes/llm_effect/v1_0_0/models/       100.00%
Overall:                                                    90.44%
```

## Circuit Breaker Behavior

```
Failures: 0 → 1 → 2 → 3 → 4 → 5 (OPEN)
                              ↑
                              Wait 60s
                              ↓
                           (HALF-OPEN)
                              ↓
                    Success → CLOSED (Reset)
```

## Configuration Reference

### Environment Variables

```bash
# Required
ZAI_API_KEY="your_api_key"  # pragma: allowlist secret

# Optional
ZAI_ENDPOINT="https://api.z.ai/api/anthropic"  # Default shown
```

### Container Configuration

```python
{
    # API Configuration (READ FROM ENVIRONMENT - DO NOT PASS VIA CONTAINER)
    # ZAI_API_KEY environment variable (required)
    # ZAI_ENDPOINT environment variable (optional, default: https://api.z.ai/api/anthropic)

    # Model Configuration
    "default_model_cloud_fast": str,              # Default: "glm-4.5"
    "context_window_cloud_fast": int,             # Default: 128000

    # Pricing
    "cost_per_1m_input_cloud_fast": float,        # Default: 0.20
    "cost_per_1m_output_cloud_fast": float,       # Default: 0.20

    # Circuit Breaker
    "circuit_breaker_threshold": int,             # Default: 5
    "circuit_breaker_timeout_seconds": float,     # Default: 60.0

    # HTTP Client
    "http_timeout_seconds": float,                # Default: 60.0
    "http_max_connections": int,                  # Default: 10

    # Retry Logic
    "max_retry_attempts": int,                    # Default: 3
    "retry_initial_backoff_seconds": float,       # Default: 1.0
    "retry_backoff_multiplier": float,            # Default: 2.0
}
```

## Integration

### With Code Generation Pipeline

```python
# Generate node code
contract = ModelContractEffect(
    input_data={
        "prompt": "Generate NodePostgresPoolEffect implementation",
        "tier": "CLOUD_FAST",
        "max_tokens": 4000,
        "temperature": 0.5,  # Lower temp for code
        "operation_type": "node_generation",
        "context_window": "# Existing patterns:\n...",
    }
)

response = await llm_node.execute_effect(contract)
generated_code = response.generated_text
```

### With Test Generation

```python
# Generate tests
contract = ModelContractEffect(
    input_data={
        "prompt": "Generate unit tests for distributed lock system",
        "tier": "CLOUD_FAST",
        "max_tokens": 4000,
        "temperature": 0.3,  # Very low temp for tests
        "operation_type": "test_generation",
        "context_window": "# Code to test:\n...",
    }
)

response = await llm_node.execute_effect(contract)
generated_tests = response.generated_text
```

## Roadmap

### Phase 1 (Complete)
- ✅ CLOUD_FAST tier (GLM-4.5)
- ✅ Circuit breaker pattern
- ✅ Retry logic with exponential backoff
- ✅ Cost & token tracking
- ✅ 90%+ test coverage

### Phase 2 (Planned)
- ⏳ LOCAL tier (Ollama/vLLM)
- ⏳ CLOUD_PREMIUM tier (GLM-4.6)
- ⏳ Response streaming support
- ⏳ Batch processing

### Phase 3 (Future)
- ⏳ Multiple cloud providers
- ⏳ Model selection optimizer
- ⏳ Cost budget enforcement
- ⏳ A/B testing framework

## Contributing

See [CONTRIBUTING.md](../../../docs/CONTRIBUTING.md) for guidelines.

## License

See [LICENSE](../../../LICENSE) for details.
