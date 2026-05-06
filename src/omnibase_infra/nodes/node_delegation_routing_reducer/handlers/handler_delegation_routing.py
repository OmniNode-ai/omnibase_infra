# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handler for delegation routing decisions.

Iterates routing tiers declared in routing_tiers.yaml (local → cheap_cloud → claude)
and returns the first tier that has a configured endpoint for the given task type.
All tier order, model assignments, and retry counts come from the YAML config —
no constants are hardcoded here.

Task-class contracts (task_class_contracts.v1.yaml) augment tier routing with
per-class pricing ceilings and cloud routing policies. When the contract file is
present (via TASK_CLASS_CONTRACT_PATH env var or the default location), routing
additionally enforces:
  - cloud_routing_policy: "blocked" skips non-local tiers for that task class
  - pricing_ceiling_per_1k_tokens: tiers whose cost tier exceeds the ceiling
    are skipped (local=low, cheap_cloud=medium, claude=high)
  - escalation_policy.tier_order: when present, overrides the default tier
    iteration order declared in routing_tiers.yaml

Related:
    - OMN-7040: Node-based delegation pipeline
    - OMN-8029: Delegation pipeline — local→cheap-cloud→claude routing
    - OMN-10615: Wire routing reducer to read task-class contracts
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import NAMESPACE_DNS, UUID, uuid5

import yaml

from omnibase_core.enums.enum_agent_capability import EnumAgentCapability
from omnibase_core.enums.enum_invocation_kind import EnumInvocationKind
from omnibase_core.models.common.model_schema_value import ModelSchemaValue
from omnibase_core.models.delegation.model_invocation_command import (
    ModelInvocationCommand,
)
from omnibase_core.models.delegation.model_routing_rule import ModelRoutingRule
from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.nodes.node_delegation_orchestrator.models.model_delegation_request import (
    ModelDelegationRequest,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_delegation_config import (
    ModelDelegationConfig,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_decision import (
    ModelRoutingDecision,
)
from omnibase_infra.nodes.node_delegation_routing_reducer.models.model_routing_tier import (
    ModelRoutingTier,
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

# Cost tier ordinal — lower is cheaper.
# Used to enforce pricing_ceiling_per_1k_tokens from task-class contracts.
_TIER_COST_ORDINAL: dict[str, int] = {
    "local": 0,
    "cheap_cloud": 1,
    "claude": 2,
    "cli_agents": 1,
}

# Approximate per-1k-token cost by tier (USD).
# These are conservative estimates used to compare against pricing ceiling.
_TIER_COST_PER_1K: dict[str, float] = {
    "local": 0.0,
    "cheap_cloud": 0.002,
    "claude": 0.015,
    "cli_agents": 0.002,
}

# cloud_routing_policy values that block routing to non-local tiers.
_CLOUD_BLOCKED_POLICY = "blocked"
_LOCAL_TIERS = {"local", "cli_agents"}


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
    # Fast-path check: prefer model with threshold if tokens fit within both the
    # fast-path threshold and the model's declared max context window.
    for model in tier_models:
        endpoint = os.environ.get(
            model.env_var, ""
        )  # ONEX_EXCLUDE: env_access - routing reads env to discover available backends
        if (
            task_type in model.use_for
            and estimated_tokens <= model.max_context_tokens
            and model.fast_path_threshold_tokens is not None
            and estimated_tokens <= model.fast_path_threshold_tokens
            and endpoint
        ):
            return model

    # Standard selection: first model that handles this task, has endpoint set,
    # and whose context window can accommodate the estimated token count.
    for model in tier_models:
        endpoint = os.environ.get(
            model.env_var, ""
        )  # ONEX_EXCLUDE: env_access - routing reads env to discover available backends
        if (
            task_type in model.use_for
            and endpoint
            and estimated_tokens <= model.max_context_tokens
        ):
            return model

    return None


_DEFAULT_CONFIG_PATH = (
    Path(__file__).parent.parent.parent.parent / "configs" / "routing_tiers.yaml"
)

_DEFAULT_TASK_CLASS_CONTRACT_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "configs"
    / "task_class_contracts.v1.yaml"
)

# Module-level config singletons — loaded once on first call.
# Tests can override by replacing these variables before calling delta().
_config: ModelDelegationConfig | None = None
_task_class_contract: dict[str, object] | None = None
_task_class_contract_loaded: bool = False


def _get_config() -> ModelDelegationConfig:
    global _config  # noqa: PLW0603
    if _config is None:
        # I/O is performed here in the handler (EFFECT boundary), not in the pure model.
        yaml_text = _DEFAULT_CONFIG_PATH.read_text()
        _config = ModelDelegationConfig.from_yaml_text(yaml_text)
    return _config


def _get_task_class_contract() -> dict[str, object] | None:
    """Load task-class contracts from YAML, returning None if not available.

    Reads from TASK_CLASS_CONTRACT_PATH env var, or the default location in
    configs/task_class_contracts.v1.yaml. Returns None when the file is absent
    so that callers can gracefully degrade to tier-only routing.
    """
    global _task_class_contract, _task_class_contract_loaded  # noqa: PLW0603
    if _task_class_contract_loaded:
        return _task_class_contract

    env_path = os.environ.get(
        "TASK_CLASS_CONTRACT_PATH", ""
    )  # ONEX_EXCLUDE: env_access - contract path configuration
    contract_path = Path(env_path) if env_path else _DEFAULT_TASK_CLASS_CONTRACT_PATH

    if not contract_path.exists():
        _task_class_contract_loaded = True
        _task_class_contract = None
        return None

    raw = yaml.safe_load(contract_path.read_text())
    _task_class_contract = raw if isinstance(raw, dict) else None
    _task_class_contract_loaded = True
    return _task_class_contract


def _task_class_entry(
    contract: dict[str, object] | None, task_type: str
) -> dict[str, object] | None:
    """Return the task-class entry for task_type, or None if not declared."""
    if contract is None:
        return None
    task_classes = contract.get("task_classes")
    if not isinstance(task_classes, dict):
        return None
    entry = task_classes.get(task_type)
    if not isinstance(entry, dict):
        return None
    return entry


def _tier_allowed_by_contract(
    tier: ModelRoutingTier,
    entry: dict[str, object] | None,
) -> bool:
    """Return True if the tier is permitted by task-class contract constraints.

    When no entry is declared, all tiers are allowed (graceful degradation).
    Enforces:
      - cloud_routing_policy: "blocked" → only local tiers permitted
      - pricing_ceiling_per_1k_tokens: tier cost must not exceed ceiling
    """
    if entry is None:
        return True

    policy = entry.get("cloud_routing_policy")
    if policy == _CLOUD_BLOCKED_POLICY and tier.name not in _LOCAL_TIERS:
        return False

    ceiling_raw = entry.get("pricing_ceiling_per_1k_tokens")
    if ceiling_raw is not None and isinstance(ceiling_raw, (int, float)):
        tier_cost = _TIER_COST_PER_1K.get(tier.name, 0.0)
        if tier_cost > float(ceiling_raw):
            return False

    return True


def _tier_order_from_contract(
    config: ModelDelegationConfig,
    entry: dict[str, object] | None,
) -> tuple[ModelRoutingTier, ...]:
    """Return tiers in contract-declared escalation order, or config default.

    When the task-class entry declares escalation_policy.tier_order, tiers are
    reordered to match. Tiers not mentioned in tier_order are appended in their
    original config order after declared tiers.
    """
    if entry is None:
        return config.tiers

    escalation = entry.get("escalation_policy")
    if not isinstance(escalation, dict):
        return config.tiers
    tier_order = escalation.get("tier_order")
    if not isinstance(tier_order, list) or not tier_order:
        return config.tiers

    tier_by_name = {t.name: t for t in config.tiers}
    ordered: list[ModelRoutingTier] = []
    seen: set[str] = set()

    for name in tier_order:
        if name in tier_by_name:
            ordered.append(tier_by_name[name])
            seen.add(name)

    # Append any tiers not mentioned in tier_order (maintains coverage).
    for tier in config.tiers:
        if tier.name not in seen:
            ordered.append(tier)

    return tuple(ordered)


def resolve_invocation_command(
    *,
    rules: tuple[ModelRoutingRule, ...],
    capability: EnumAgentCapability,
    payload: dict[str, object],
    task_id: UUID,
    correlation_id: UUID,
) -> ModelInvocationCommand:
    """Resolve a capability to a typed invocation command.

    Part 1 supports AGENT rules only. MODEL dispatch remains deferred to Part 2.
    """
    for rule in rules:
        if rule.capability is not capability:
            continue
        if rule.invocation_kind is EnumInvocationKind.MODEL:
            raise NotImplementedError("MODEL deferred to Part 2")
        return ModelInvocationCommand(
            task_id=task_id,
            correlation_id=correlation_id,
            invocation_kind=rule.invocation_kind,
            agent_protocol=rule.agent_protocol,
            model_backend=rule.model_backend,
            target_ref=rule.target_ref,
            payload={
                key: value
                if isinstance(value, ModelSchemaValue)
                else ModelSchemaValue.from_value(value)
                for key, value in payload.items()
            },
        )
    raise LookupError(f"no routing rule for capability={capability.value}")


def delta(request: ModelDelegationRequest) -> ModelRoutingDecision:
    """Compute routing decision for a delegation request.

    Iterates tiers in declaration order (local → cheap_cloud → claude), with
    optional reordering from task-class contract escalation_policy.tier_order.
    Returns the first tier that has a configured endpoint, handles the requested
    task type, and satisfies task-class contract constraints (cloud routing policy
    and pricing ceiling).

    Args:
        request: The delegation request to route.

    Returns:
        A routing decision with selected model, endpoint, and config.

    Raises:
        ProtocolConfigurationError: If no tier has a configured endpoint for the task type.
    """
    config = _get_config()
    task_type = request.task_type
    estimated_tokens = _estimate_prompt_tokens(request.prompt)

    contract = _get_task_class_contract()
    entry = _task_class_entry(contract, task_type)
    tiers = _tier_order_from_contract(config, entry)

    for tier in tiers:
        if not _tier_allowed_by_contract(tier, entry):
            continue

        selected = _select_model_for_task(tier.models, task_type, estimated_tokens)
        if selected is None:
            continue

        endpoint_url = os.environ.get(
            selected.env_var, ""
        )  # ONEX_EXCLUDE: env_access - routing reads env to discover available backends
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
        if entry is not None:
            policy_val = entry.get("cloud_routing_policy")
            policy_str = policy_val if isinstance(policy_val, str) else "allowed"
            rationale += (
                f" Contract-driven: task_class='{task_type}' policy='{policy_str}'."
            )

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

    context = ModelInfraErrorContext.with_correlation(
        correlation_id=request.correlation_id,
        transport_type=EnumInfraTransportType.RUNTIME,
        operation="delegation_routing",
    )
    msg = (
        f"No tier has a configured endpoint for task_type='{task_type}'. "
        f"Set at least one of the required env vars in routing_tiers.yaml "
        f"(e.g., LLM_CODER_URL for local tier, ANTHROPIC_API_KEY for claude tier)."
    )
    raise ProtocolConfigurationError(msg, context=context)


__all__: list[str] = ["delta", "resolve_invocation_command"]
