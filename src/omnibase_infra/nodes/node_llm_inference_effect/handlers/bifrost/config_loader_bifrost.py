# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bifrost config loader from environment variables.

Reads LLM endpoint URLs from environment variables and produces a
``ModelBifrostConfig`` with default routing rules. Each env var maps
to a backend entry; default routing rules route by operation type and
cost tier.

Environment Variables:
    LLM_CODER_URL: Long-context code generation endpoint (e.g. Qwen3-Coder-30B).
    LLM_CODER_FAST_URL: Mid-tier code/routing endpoint (e.g. Qwen3-14B).
    LLM_EMBEDDING_URL: Embedding generation endpoint (e.g. Qwen3-Embedding-8B).
    LLM_DEEPSEEK_R1_URL: Reasoning/code review endpoint (e.g. DeepSeek-R1).
    LLM_SMALL_URL: Optional lightweight/portable model endpoint.

Related:
    - Plan D WS1: Wire LLM nodes to dispatch engine
    - ModelBifrostConfig: Target configuration model
    - ModelBifrostBackendConfig: Per-backend endpoint config
    - ModelBifrostRoutingRule: Declarative routing rule

.. versionadded:: 0.27.0
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

# Stable rule UUIDs for audit logging
_RULE_CODE_PREMIUM = UUID("d1a00001-0001-4000-8000-000000000001")
_RULE_CODE_STANDARD = UUID("d1a00001-0002-4000-8000-000000000002")
_RULE_EMBEDDING = UUID("d1a00001-0003-4000-8000-000000000003")
_RULE_REASONING = UUID("d1a00001-0004-4000-8000-000000000004")
_RULE_LIGHTWEIGHT = UUID("d1a00001-0005-4000-8000-000000000005")

# Backend IDs (stable slugs for config references)
BACKEND_CODER: str = "local-qwen-coder-30b"
BACKEND_CODER_FAST: str = "local-qwen-14b"
BACKEND_EMBEDDING: str = "local-qwen-embedding-8b"
BACKEND_DEEPSEEK_R1: str = "local-deepseek-r1"
BACKEND_SMALL: str = "local-qwen-7b"


def load_bifrost_config_from_env() -> ModelBifrostConfig:
    """Load bifrost config from environment variables.

    Reads ``LLM_CODER_URL``, ``LLM_CODER_FAST_URL``, ``LLM_EMBEDDING_URL``,
    ``LLM_DEEPSEEK_R1_URL``, and optionally ``LLM_SMALL_URL`` from the
    environment and builds a ``ModelBifrostConfig`` with default routing
    rules.

    Returns:
        A ``ModelBifrostConfig`` populated from environment. Only backends
        with configured URLs are included.

    Raises:
        ValueError: If no LLM endpoint URLs are configured.
    """
    backends: dict[str, ModelBifrostBackendConfig] = {}
    default_backend_ids: list[str] = []

    # Map env vars to backend configs
    _env_backend_map = {
        "LLM_CODER_URL": (BACKEND_CODER, None),
        "LLM_CODER_FAST_URL": (BACKEND_CODER_FAST, None),
        "LLM_EMBEDDING_URL": (BACKEND_EMBEDDING, None),
        "LLM_DEEPSEEK_R1_URL": (BACKEND_DEEPSEEK_R1, None),
        "LLM_SMALL_URL": (BACKEND_SMALL, None),
    }

    for env_var, (backend_id, model_name) in _env_backend_map.items():
        url = os.environ.get(env_var)
        if url:
            backends[backend_id] = ModelBifrostBackendConfig(
                backend_id=backend_id,
                base_url=url,
                model_name=model_name,
            )
            default_backend_ids.append(backend_id)
            logger.debug("Bifrost backend configured: %s = %s", backend_id, url)

    if not backends:
        msg = (
            "No LLM endpoint URLs configured. Set at least one of: "
            "LLM_CODER_URL, LLM_CODER_FAST_URL, LLM_EMBEDDING_URL, "
            "LLM_DEEPSEEK_R1_URL, LLM_SMALL_URL"
        )
        raise ValueError(msg)

    # Build routing rules based on available backends
    routing_rules = _build_default_routing_rules(backends)

    config = ModelBifrostConfig(
        backends=backends,
        routing_rules=tuple(routing_rules),
        default_backends=tuple(default_backend_ids),
    )

    logger.info(
        "Bifrost config loaded from env: %d backends, %d routing rules",
        len(backends),
        len(routing_rules),
    )

    return config


def _build_default_routing_rules(
    backends: dict[str, ModelBifrostBackendConfig],
) -> list[ModelBifrostRoutingRule]:
    """Build default routing rules from available backends.

    Rules are ordered by priority (lower = evaluated first):
    - 10: Embedding operations -> embedding backend
    - 20: Reasoning operations -> DeepSeek-R1 backend
    - 30: Premium code operations -> coder 30B backend
    - 40: Standard code operations -> coder fast backend
    - 50: Lightweight operations -> small model backend

    Only rules with available backends are included.

    Args:
        backends: Available backend configurations.

    Returns:
        List of routing rules sorted by priority.
    """
    rules: list[ModelBifrostRoutingRule] = []

    # Embedding operations
    if BACKEND_EMBEDDING in backends:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_EMBEDDING,
                priority=10,
                match_operation_types=("embedding",),
                backend_ids=(BACKEND_EMBEDDING,),
            )
        )

    # Reasoning operations
    if BACKEND_DEEPSEEK_R1 in backends:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_REASONING,
                priority=20,
                match_operation_types=("reasoning", "code_review"),
                backend_ids=(BACKEND_DEEPSEEK_R1,),
            )
        )

    # Premium code (long context)
    if BACKEND_CODER in backends:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_CODE_PREMIUM,
                priority=30,
                match_cost_tiers=("premium",),
                backend_ids=(
                    BACKEND_CODER,
                    *((BACKEND_CODER_FAST,) if BACKEND_CODER_FAST in backends else ()),
                ),
            )
        )

    # Standard code (mid-tier)
    if BACKEND_CODER_FAST in backends:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_CODE_STANDARD,
                priority=40,
                match_cost_tiers=("standard",),
                backend_ids=(
                    BACKEND_CODER_FAST,
                    *((BACKEND_CODER,) if BACKEND_CODER in backends else ()),
                ),
            )
        )

    # Lightweight
    if BACKEND_SMALL in backends:
        rules.append(
            ModelBifrostRoutingRule(
                rule_id=_RULE_LIGHTWEIGHT,
                priority=50,
                match_cost_tiers=("cheap",),
                backend_ids=(BACKEND_SMALL,),
            )
        )

    return rules


__all__: list[str] = ["load_bifrost_config_from_env"]
