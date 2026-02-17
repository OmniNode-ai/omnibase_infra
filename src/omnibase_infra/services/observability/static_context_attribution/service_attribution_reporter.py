# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Attribution report builder for static context token cost analysis.

Orchestrates the full attribution pipeline:
1. Parse static context into sections (deterministic)
2. Count tokens per section
3. Score utilization against a model response
4. Optionally augment with LLM semantic categories
5. Build a provenance-tracked report

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import logging

from omnibase_infra.services.observability.static_context_attribution.model_static_context_report import (
    ModelStaticContextReport,
)
from omnibase_infra.services.observability.static_context_attribution.service_llm_category_augmenter import (
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

logger = logging.getLogger(__name__)

# Service code version for provenance tracking
_CODE_VERSION = "0.1.0"


class ServiceAttributionReporter:
    """Orchestrates the full static context attribution pipeline.

    Coordinates parser, token counter, utilization scorer, and optional
    LLM augmenter to produce a complete attribution report with
    provenance metadata.

    Usage:
        >>> reporter = ServiceAttributionReporter()
        >>> report = await reporter.build_report(
        ...     context_files={"CLAUDE.md": "## Overview\\nContent here"},
        ...     response="I followed the overview guidelines",
        ... )
        >>> report.total_tokens > 0
        True
    """

    def __init__(
        self,
        parser: ServiceStaticContextParser | None = None,
        token_counter: ServiceTokenCounter | None = None,
        utilization_scorer: ServiceUtilizationScorer | None = None,
        llm_augmenter: ServiceLlmCategoryAugmenter | None = None,
    ) -> None:
        """Initialize with optional custom service instances.

        Args:
            parser: Custom parser. Defaults to ``ServiceStaticContextParser()``.
            token_counter: Custom counter. Defaults to ``ServiceTokenCounter()``.
            utilization_scorer: Custom scorer. Defaults to
                ``ServiceUtilizationScorer()``.
            llm_augmenter: Optional LLM augmenter for Pass 2.
                None skips semantic categorization.
        """
        self._parser = parser or ServiceStaticContextParser()
        self._token_counter = token_counter or ServiceTokenCounter()
        self._utilization_scorer = utilization_scorer or ServiceUtilizationScorer()
        self._llm_augmenter = llm_augmenter

    async def build_report(
        self,
        context_files: dict[str, str],
        response: str,
    ) -> ModelStaticContextReport:
        """Build a complete attribution report.

        Pipeline:
        1. Parse all context files into sections
        2. Count tokens per section
        3. Optionally augment with LLM semantic categories
        4. Score utilization against response
        5. Assemble report with provenance

        Args:
            context_files: Mapping of file path to markdown content.
            response: Model response text to attribute against.

        Returns:
            Complete attribution report with provenance metadata.
        """
        # Compute provenance hashes
        full_context = "\n".join(
            f"--- {path} ---\n{content}" for path, content in context_files.items()
        )
        input_hash = ModelStaticContextReport.compute_hash(full_context)
        response_hash = ModelStaticContextReport.compute_hash(response)

        # Pass 1: Parse and count tokens
        sections = self._parser.parse_multiple(context_files)
        sections = self._token_counter.count_sections(sections)

        # Pass 2 (optional): LLM augmentation
        llm_augmented = False
        if self._llm_augmenter is not None:
            sections = await self._llm_augmenter.augment(sections)
            llm_augmented = True

        # Score utilization
        attributions = self._utilization_scorer.score(sections, response)

        # Compute totals
        total_tokens = sum(a.section.token_count for a in attributions)
        total_attributed = sum(a.attributed_tokens for a in attributions)

        report = ModelStaticContextReport(
            attributions=tuple(attributions),
            total_tokens=total_tokens,
            total_attributed_tokens=total_attributed,
            input_hash=input_hash,
            response_hash=response_hash,
            code_version=_CODE_VERSION,
            source_files=tuple(context_files.keys()),
            llm_augmented=llm_augmented,
        )

        logger.info(
            "Attribution report built: %d sections, %d total tokens, "
            "%d attributed tokens (%.1f%% utilization), "
            "input_hash=%s, llm_augmented=%s",
            len(attributions),
            total_tokens,
            total_attributed,
            (total_attributed / total_tokens * 100) if total_tokens > 0 else 0.0,
            input_hash[:12],
            llm_augmented,
        )

        return report


__all__ = ["ServiceAttributionReporter"]
