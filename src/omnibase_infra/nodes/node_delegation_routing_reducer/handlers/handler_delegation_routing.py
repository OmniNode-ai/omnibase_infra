# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for delegation routing decisions.

Iterates routing tiers declared in routing_tiers.yaml (local → cheap_cloud → claude)
and returns the first tier that has a configured endpoint for the given task type.
All tier order, model assignments, and retry counts come from the YAML config —
no constants are hardcoded here.

Related:
    - OMN-7040: Node-based delegation pipeline
    - OMN-8029: Delegation pipeline — local→cheap-cloud→claude routing
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import NAMESPACE_DNS, UUID, uuid5

from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_delegation_config import (
    ModelDelegationConfig,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_tier_model import (
    ModelTierModel,
)

# System prompts by task type — kept here because they are presentation strings,
# not routing configuration.
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
    "code_generation": (
        "You are a code generation assistant. Implement the requested functionality "
        "following existing patterns, conventions, and architecture in the codebase."
    ),
    "code_review": (
        "You are a code review assistant. Identify bugs, style violations, and "
        "architectural issues in the provided code. Be specific and actionable."
    ),
    "refactor": (
        "You are a refactoring assistant. Improve the structure, readability, and "
        "maintainability of the provided code without changing its behavior."
    ),
    "reasoning": (
        "You are a reasoning assistant. Think through the problem step by step "
        "and provide a well-structured analysis."
    ),
    "planning": (
        "You are a planning assistant. Break down the requested work into clear, "
        "actionable steps with explicit acceptance criteria."
    ),
    "review": (
        "You are a review assistant. Evaluate the provided artifacts against "
        "the stated requirements and report any gaps or issues."
    ),
    "summarization": (
        "You are a summarization assistant. Produce a concise, accurate summary "
        "of the provided content."
    ),
    "simple_tasks": (
        "You are a helpful assistant. Complete the requested task accurately."
    ),
    "escalation": (
        "You are an expert assistant handling a complex task that requires deep "
        "reasoning and careful consideration. Approach this methodically."
    ),
    "complex_reasoning": (
        "You are an expert reasoning assistant. Analyze the problem deeply, "
        "consider edge cases, and provide a comprehensive solution."
    ),
    "agent_orchestration": (
        "You are an orchestration assistant. Coordinate the required sub-tasks "
        "and ensure each is completed correctly before proceeding."
    ),
}


def _estimate_prompt_tokens(prompt: str) -> int:
    """Estimate token count from prompt character length (4 chars/token heuristic)."""
    return len(prompt) // 4


def _backend_id_for_model(model_id: str) -> UUID:
    """Generate a stable UUID for a model ID."""
    return uuid5(NAMESPACE_DNS, f"omninode.ai/backends/{model_id}")


def _select_model_for_task(
    tier_models: tuple[ModelTierModel, ...],
    task_type: str,
    estimated_tokens: int,
) -> ModelTierModel | None:
    """Select the best model from a tier for the given task and token count.

    Prefers fast-path models when prompt fits within their threshold.
    Falls back to any model that declares the task type in use_for.
    """
    # Fast-path check: prefer model with threshold if tokens fit
    for model in tier_models:
        if (
            task_type in model.use_for
            and model.fast_path_threshold_tokens is not None
            and estimated_tokens <= model.fast_path_threshold_tokens
            and os.environ.get(model.env_var, "")
        ):
            return model

    # Standard selection: first model that handles this task and has endpoint set
    for model in tier_models:
        if task_type in model.use_for and os.environ.get(model.env_var, ""):
            return model

    return None


# Module-level config singleton — loaded once at import time.
# Tests can override by replacing this variable before calling delta().
_config: ModelDelegationConfig | None = None


def _get_config() -> ModelDelegationConfig:
    global _config  # noqa: PLW0603
    if _config is None:
        _config = ModelDelegationConfig.from_yaml()
    return _config


def delta(request: ModelDelegationRequest) -> ModelRoutingDecision:
    """Compute routing decision for a delegation request.

    Iterates tiers in declaration order (local → cheap_cloud → claude).
    Returns the first tier that has a configured endpoint and handles the
    requested task type. Claude tier is always the final fallback.

    Args:
        request: The delegation request to route.

    Returns:
        A routing decision with selected model, endpoint, and config.

    Raises:
        ValueError: If no tier has a configured endpoint for the task type.
    """
    config = _get_config()
    task_type = request.task_type
    estimated_tokens = _estimate_prompt_tokens(request.prompt)

    for tier in config.tiers:
        selected = _select_model_for_task(tier.models, task_type, estimated_tokens)
        if selected is None:
            continue

        endpoint_url = os.environ.get(selected.env_var, "")
        if not endpoint_url:
            continue

        system_prompt = _SYSTEM_PROMPTS.get(
            task_type,
            f"You are a helpful assistant completing a {task_type} task.",
        )

        rationale = (
            f"Task '{task_type}' (~{estimated_tokens} tokens) routed to "
            f"{selected.id} via tier '{tier.name}' "
            f"(max_context={selected.max_context_tokens})."
        )
        if (
            selected.fast_path_threshold_tokens
            and estimated_tokens <= selected.fast_path_threshold_tokens
        ):
            rationale += f" Fast-path: tokens within {selected.fast_path_threshold_tokens} threshold."

        # Map cost tier from tier name
        cost_tier_map = {"local": "low", "cheap_cloud": "medium", "claude": "high"}
        cost_tier = cost_tier_map.get(tier.name, tier.name)

        return ModelRoutingDecision(
            correlation_id=request.correlation_id,
            task_type=task_type,
            selected_model=selected.id,
            selected_backend_id=_backend_id_for_model(selected.id),
            endpoint_url=endpoint_url,
            cost_tier=cost_tier,
            max_context_tokens=selected.max_context_tokens,
            system_prompt=system_prompt,
            rationale=rationale,
        )

    msg = (
        f"No tier has a configured endpoint for task_type='{task_type}'. "
        f"Set at least one of the required env vars in routing_tiers.yaml "
        f"(e.g., LLM_CODER_URL for local tier, ANTHROPIC_API_KEY for claude tier)."
    )
    raise ValueError(msg)


__all__: list[str] = ["delta"]
