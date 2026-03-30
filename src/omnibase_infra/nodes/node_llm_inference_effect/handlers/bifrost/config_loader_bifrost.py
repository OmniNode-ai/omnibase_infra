# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bifrost config loader from environment / Infisical.

Builds a ``ModelBifrostConfig`` from environment variables that define
the local LLM server topology. Each ``LLM_*_URL`` env var maps to a
backend entry with default routing rules based on operation type and
cost tier.

Environment variables consumed:
    LLM_CODER_URL       - Qwen3-Coder-30B-A3B (RTX 5090, 64K ctx)
    LLM_CODER_FAST_URL  - Qwen3-14B-AWQ (RTX 4090, 40K ctx)
    LLM_EMBEDDING_URL   - Qwen3-Embedding-8B-4bit (M2 Ultra)
    LLM_DEEPSEEK_R1_URL - DeepSeek-R1-Distill-Qwen-32B-bf16 (M2 Ultra)
    LLM_SMALL_URL       - Qwen2.5-Coder-7B MLX-4bit (MacBook Air, optional)
    GLM_BASE_URL        - GLM (z.ai) external provider (optional)
    GEMINI_BASE_URL     - Gemini API base URL override (optional)

Related:
    - OMN-6787: Build bifrost config loader from env/Infisical
    - ModelBifrostConfig: Configuration model
    - HandlerBifrostGateway: Gateway that consumes this config

.. versionadded:: 0.29.0
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_config import (
    ModelBifrostBackendConfig,
    ModelBifrostConfig,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.bifrost.model_bifrost_routing_rule import (
    ModelBifrostRoutingRule,
)

logger = logging.getLogger(__name__)

# Deterministic UUIDs for routing rules (generated once, stable across loads)
_RULE_UUID_CODE_PREMIUM = UUID("a1b2c3d4-0001-4000-8000-000000000001")
_RULE_UUID_CODE_STANDARD = UUID("a1b2c3d4-0002-4000-8000-000000000002")
_RULE_UUID_CODE_CHEAP = UUID("a1b2c3d4-0003-4000-8000-000000000003")
_RULE_UUID_EMBEDDING = UUID("a1b2c3d4-0004-4000-8000-000000000004")
_RULE_UUID_REASONING = UUID("a1b2c3d4-0005-4000-8000-000000000005")
_RULE_UUID_EVAL = UUID("a1b2c3d4-0006-4000-8000-000000000006")


def load_bifrost_config_from_env() -> ModelBifrostConfig:
    """Build ``ModelBifrostConfig`` from environment variables.

    Reads ``LLM_*_URL`` env vars and constructs backend entries + routing
    rules. Backends are only added when their env var is set and non-empty.

    Returns:
        A fully configured ``ModelBifrostConfig`` ready for
        ``HandlerBifrostGateway``.

    Raises:
        ValueError: If no LLM backend env vars are set.

    .. versionadded:: 0.29.0
    """
    backends: dict[str, ModelBifrostBackendConfig] = {}

    # Local backends (from env vars)
    _add_backend_if_set(backends, "local-coder-30b", "LLM_CODER_URL")
    _add_backend_if_set(backends, "local-coder-14b", "LLM_CODER_FAST_URL")
    _add_backend_if_set(backends, "local-embedding", "LLM_EMBEDDING_URL")
    _add_backend_if_set(backends, "local-deepseek-r1", "LLM_DEEPSEEK_R1_URL")
    _add_backend_if_set(backends, "local-small", "LLM_SMALL_URL")

    # External backends (only added when API key is configured)
    _add_backend_if_set(backends, "glm", "GLM_BASE_URL")
    # Gemini uses a well-known base URL; only add when API key is present
    if os.environ.get("GEMINI_API_KEY", "").strip():
        _add_backend_if_set(
            backends,
            "gemini",
            "GEMINI_BASE_URL",
            default_url="https://generativelanguage.googleapis.com",
        )

    if not backends:
        msg = (
            "No LLM backend env vars set. At least one of LLM_CODER_URL, "
            "LLM_CODER_FAST_URL, LLM_EMBEDDING_URL, LLM_DEEPSEEK_R1_URL "
            "must be configured."
        )
        raise ValueError(msg)

    routing_rules = _build_default_routing_rules(backends)
    default_backend_ids = _pick_default_backends(backends)

    config = ModelBifrostConfig(
        backends=backends,
        routing_rules=tuple(routing_rules),
        default_backends=default_backend_ids,
    )

    logger.info(
        "Loaded bifrost config from env: %d backends, %d routing rules",
        len(backends),
        len(routing_rules),
    )
    return config


def _add_backend_if_set(
    backends: dict[str, ModelBifrostBackendConfig],
    backend_id: str,
    env_var: str,
    *,
    default_url: str | None = None,
) -> None:
    """Add a backend entry if the env var is set and non-empty."""
    url = os.environ.get(env_var, "").strip()
    if not url and default_url:
        url = default_url
    if not url:
        return
    backends[backend_id] = ModelBifrostBackendConfig(
        backend_id=backend_id,
        base_url=url,
    )
    logger.debug("Added bifrost backend '%s' from %s=%s", backend_id, env_var, url)


def _build_default_routing_rules(
    backends: dict[str, ModelBifrostBackendConfig],
) -> list[ModelBifrostRoutingRule]:
    """Build default routing rules based on available backends."""
    rules: list[ModelBifrostRoutingRule] = []
    available = set(backends)

    # Premium code tasks -> local Qwen3-Coder-30B (RTX 5090, 64K ctx)
    premium_ids = _filter_available(["local-coder-30b", "local-coder-14b"], available)
    if premium_ids:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_UUID_CODE_PREMIUM,
                priority=10,
                match_cost_tiers=("premium",),
                match_operation_types=("chat_completion", "completion"),
                backend_ids=tuple(premium_ids),
            )
        )

    # Standard code tasks -> local Qwen3-14B or GLM
    standard_ids = _filter_available(
        ["local-coder-14b", "glm", "local-coder-30b"], available
    )
    if standard_ids:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_UUID_CODE_STANDARD,
                priority=20,
                match_cost_tiers=("standard",),
                match_operation_types=("chat_completion", "completion"),
                backend_ids=tuple(standard_ids),
            )
        )

    # Cheap code tasks -> Gemini Flash or local small
    cheap_ids = _filter_available(
        ["gemini", "local-small", "local-coder-14b"], available
    )
    if cheap_ids:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_UUID_CODE_CHEAP,
                priority=30,
                match_cost_tiers=("cheap",),
                match_operation_types=("chat_completion", "completion"),
                backend_ids=tuple(cheap_ids),
            )
        )

    # Embedding tasks -> local embedding backend
    embedding_ids = _filter_available(["local-embedding"], available)
    if embedding_ids:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_UUID_EMBEDDING,
                priority=10,
                match_operation_types=("embedding",),
                backend_ids=tuple(embedding_ids),
            )
        )

    # Reasoning tasks -> DeepSeek-R1
    reasoning_ids = _filter_available(
        ["local-deepseek-r1", "local-coder-30b"], available
    )
    if reasoning_ids:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_UUID_REASONING,
                priority=10,
                match_operation_types=("reasoning",),
                backend_ids=tuple(reasoning_ids),
            )
        )

    # Eval tasks -> cheapest available (Gemini, GLM, local small)
    eval_ids = _filter_available(
        ["gemini", "glm", "local-small", "local-coder-14b"], available
    )
    if eval_ids:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_UUID_EVAL,
                priority=10,
                match_operation_types=("eval",),
                backend_ids=tuple(eval_ids),
            )
        )

    return rules


def _filter_available(candidates: list[str], available: set[str]) -> list[str]:
    """Return only candidates that are in the available set."""
    return [c for c in candidates if c in available]


def _pick_default_backends(
    backends: dict[str, ModelBifrostBackendConfig],
) -> tuple[str, ...]:
    """Pick default fallback backends in preference order."""
    preference = [
        "local-coder-14b",
        "local-coder-30b",
        "local-small",
        "glm",
        "gemini",
    ]
    defaults = _filter_available(preference, set(backends))
    if not defaults:
        # Fall back to whatever is available
        defaults = list(backends.keys())[:1]
    return tuple(defaults)


__all__: list[str] = ["load_bifrost_config_from_env"]
