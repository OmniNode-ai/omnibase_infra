# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Static context token cost attribution service.

This module provides infrastructure for measuring token costs per static
context section (CLAUDE.md, memory files, etc.) and attributing utilization
to specific sections based on edit-distance anchoring.

Two-Pass Architecture:
    Pass 1 (Deterministic): Heading-based parser splits context into sections,
    tokenizer counts tokens per section. Zero cost, fully reproducible.

    Pass 2 (Optional LLM): Handler-driven LLM reclassifies sections into
    semantic categories (config, rules, topology, examples).

Components:
    - ServiceStaticContextParser: Deterministic heading-based parser
    - ServiceTokenCounter: Token counting per parsed section
    - ServiceUtilizationScorer: Edit-distance utilization attribution
    - ServiceLlmCategoryAugmenter: Optional LLM semantic categorization
    - ModelContextSection: Parsed section with metadata
    - ModelSectionAttribution: Full attribution result with provenance

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)
from omnibase_infra.services.observability.static_context_attribution.model_section_attribution import (
    ModelSectionAttribution,
)
from omnibase_infra.services.observability.static_context_attribution.model_static_context_report import (
    ModelStaticContextReport,
)
from omnibase_infra.services.observability.static_context_attribution.service_attribution_reporter import (
    ServiceAttributionReporter,
)
from omnibase_infra.services.observability.static_context_attribution.service_llm_category_augmenter import (
    LlmInferenceFn,
    ServiceLlmCategoryAugmenter,
)
from omnibase_infra.services.observability.static_context_attribution.service_static_context_parser import (
    ServiceStaticContextParser,
)
from omnibase_infra.services.observability.static_context_attribution.service_token_counter import (
    ServiceTokenCounter,
)
from omnibase_infra.services.observability.static_context_attribution.service_utilization_scorer import (
    ServiceUtilizationScorer,
)

__all__ = [
    "LlmInferenceFn",
    "ModelContextSection",
    "ModelSectionAttribution",
    "ModelStaticContextReport",
    "ServiceAttributionReporter",
    "ServiceLlmCategoryAugmenter",
    "ServiceStaticContextParser",
    "ServiceTokenCounter",
    "ServiceUtilizationScorer",
]
