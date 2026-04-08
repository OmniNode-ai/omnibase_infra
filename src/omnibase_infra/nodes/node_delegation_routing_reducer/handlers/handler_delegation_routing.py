# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for delegation routing decisions.

Pure function that maps (task_type, prompt_length) to a ModelRoutingDecision.
Reads endpoint URLs from environment. No I/O beyond os.environ.

Routing table (from existing _HANDLER_ROUTING in delegation_orchestrator.py):
    test     -> Qwen3-Coder-30B-A3B (LLM_CODER_URL, 64K context)
    research -> Qwen3-Coder-30B-A3B (LLM_CODER_URL, 64K context)
    document -> DeepSeek-R1-32B     (LLM_DEEPSEEK_R1_URL, 32K context)

Token-count optimization:
    If prompt tokens <= 24K, test/research tasks are eligible for
    DeepSeek-R1-14B (LLM_CODER_FAST_URL, 24K context) as a faster alternative.

Related:
    - OMN-7040: Node-based delegation pipeline
"""

from __future__ import annotations

import os
from uuid import UUID, uuid4

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)

# System prompts by task type
_SYSTEM_PROMPTS: dict[str, str] = {
    "test": (
        "You are a test generation assistant. Write comprehensive pytest unit tests "
        "for the provided code. Include edge cases, error paths, and clear assertions. "
        "Use @pytest.mark.unit decorator on all tests."
    ),
    "document": (
        "You are a documentation assistant. Write clear, comprehensive docstrings "
        "and documentation for the provided code. Follow Google-style docstrings "
        "with Args, Returns, and Raises sections."
    ),
    "research": (
        "You are a code research assistant. Analyze the provided code and answer "
        "questions about its behavior, architecture, and design decisions. "
        "Be thorough and cite specific lines when relevant."
    ),
}

# Routing table: task_type -> (model_name, env_var, cost_tier, max_context)
_ROUTING_TABLE: dict[str, tuple[str, str, str, int]] = {
    "test": ("Qwen3-Coder-30B-A3B", "LLM_CODER_URL", "low", 65536),
    "research": ("Qwen3-Coder-30B-A3B", "LLM_CODER_URL", "low", 65536),
    "document": ("DeepSeek-R1-32B", "LLM_DEEPSEEK_R1_URL", "low", 32768),
}

# Fast-path threshold: if prompt fits within 24K tokens, use faster model
_FAST_PATH_TOKEN_THRESHOLD: int = 24576
_FAST_PATH_MODEL: str = "deepseek-r1-14b"
_FAST_PATH_ENV_VAR: str = "LLM_CODER_FAST_URL"
_FAST_PATH_MAX_CONTEXT: int = 24576
_FAST_PATH_ELIGIBLE_TASKS: frozenset[str] = frozenset({"test", "research"})


def _estimate_prompt_tokens(prompt: str) -> int:
    """Estimate token count from prompt character length.

    Uses a rough 4 chars/token heuristic. This is sufficient for
    routing decisions; exact counts come from the LLM response.
    """
    return len(prompt) // 4


def delta(request: ModelDelegationRequest) -> ModelRoutingDecision:
    """Compute routing decision for a delegation request.

    Pure function: reads endpoint URLs from environment, applies
    routing rules, returns immutable decision.

    Args:
        request: The delegation request to route.

    Returns:
        A routing decision with selected model, endpoint, and config.

    Raises:
        ValueError: If task_type is unknown or required endpoint is not configured.
    """
    task_type = request.task_type

    if task_type not in _ROUTING_TABLE:
        msg = f"Unknown task_type: {task_type}"
        raise ValueError(msg)

    model_name, env_var, cost_tier, max_context = _ROUTING_TABLE[task_type]
    estimated_tokens = _estimate_prompt_tokens(request.prompt)

    # Check fast-path eligibility
    fast_url = os.environ.get(_FAST_PATH_ENV_VAR, "")
    if (
        task_type in _FAST_PATH_ELIGIBLE_TASKS
        and estimated_tokens <= _FAST_PATH_TOKEN_THRESHOLD
        and fast_url
    ):
        model_name = _FAST_PATH_MODEL
        env_var = _FAST_PATH_ENV_VAR
        max_context = _FAST_PATH_MAX_CONTEXT
        rationale = (
            f"Task '{task_type}' with ~{estimated_tokens} tokens fits within "
            f"fast-path threshold ({_FAST_PATH_TOKEN_THRESHOLD}); "
            f"routing to {_FAST_PATH_MODEL} for lower latency."
        )
    else:
        rationale = (
            f"Task '{task_type}' routed to {model_name} "
            f"(~{estimated_tokens} estimated tokens, {max_context} max context)."
        )

    endpoint_url = os.environ.get(env_var, "")
    if not endpoint_url:
        msg = f"Required endpoint {env_var} is not configured in environment"
        raise ValueError(msg)

    return ModelRoutingDecision(
        correlation_id=request.correlation_id,
        task_type=task_type,
        selected_model=model_name,
        selected_backend_id=_backend_id_for_model(model_name),
        endpoint_url=endpoint_url,
        cost_tier=cost_tier,
        max_context_tokens=max_context,
        system_prompt=_SYSTEM_PROMPTS[task_type],
        rationale=rationale,
    )


def _backend_id_for_model(model_name: str) -> UUID:
    """Generate a stable UUID for a model name using UUID5 with DNS namespace."""
    from uuid import NAMESPACE_DNS, uuid5

    return uuid5(NAMESPACE_DNS, f"omninode.ai/backends/{model_name}")


__all__: list[str] = ["delta"]
