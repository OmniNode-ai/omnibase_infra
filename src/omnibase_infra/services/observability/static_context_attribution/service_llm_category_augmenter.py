# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Optional LLM augmentation pass for semantic category classification.

Pass 2 of the two-pass architecture. Uses a handler-driven LLM to
reclassify deterministically parsed sections into semantic categories
(config, rules, topology, examples, etc.).

This pass is optional. The deterministic parser (Pass 1) assigns
UNCATEGORIZED to all sections. Pass 2 enriches sections with semantic
categories for more granular cost attribution analysis.

The augmenter uses a simple protocol: it accepts a callable that takes
a prompt string and returns a category string. This allows plugging in
any LLM backend (local, OpenAI, Anthropic) via the handler system.

Related Tickets:
    - OMN-2241: E1-T7 Static context token cost attribution
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from omnibase_infra.enums.enum_context_section_category import (
    EnumContextSectionCategory,
)
from omnibase_infra.services.observability.static_context_attribution.model_context_section import (
    ModelContextSection,
)

logger = logging.getLogger(__name__)

# Type alias for async LLM inference function: prompt -> response text
LlmInferenceFn = Callable[[str], Awaitable[str]]

_CATEGORY_PROMPT_TEMPLATE = """Classify the following markdown section into exactly one semantic category.

Categories:
- config: Configuration and environment variables
- rules: Development rules, standards, and invariants
- topology: Infrastructure topology and network architecture
- examples: Code examples, usage patterns, and snippets
- commands: CLI commands, database operations, health checks
- architecture: System architecture, node types, data flow
- documentation: Documentation references and guides
- testing: Testing standards, markers, fixtures
- error_handling: Error hierarchy, patterns, recovery

Section heading: {heading}
Section content (first 500 chars):
{content_preview}

Respond with ONLY the category name (one word, lowercase). Example: config"""

# Map from string values to enum members
_CATEGORY_MAP: dict[str, EnumContextSectionCategory] = {
    member.value: member for member in EnumContextSectionCategory
}


class ServiceLlmCategoryAugmenter:
    """Optional LLM-driven semantic category classifier for context sections.

    Accepts an async inference function and uses it to classify sections
    into semantic categories. Falls back to UNCATEGORIZED on any error.

    Usage:
        >>> async def my_llm(prompt: str) -> str:
        ...     return "config"  # Mock LLM
        >>> augmenter = ServiceLlmCategoryAugmenter(llm_fn=my_llm)
        >>> sections = [ModelContextSection(content="POSTGRES_HOST=...")]
        >>> augmented = await augmenter.augment(sections)
        >>> augmented[0].category
        <EnumContextSectionCategory.CONFIG: 'config'>
    """

    def __init__(self, llm_fn: LlmInferenceFn) -> None:
        """Initialize with an async LLM inference function.

        Args:
            llm_fn: Async callable that takes a prompt string and returns
                the LLM response text.
        """
        self._llm_fn = llm_fn

    async def augment(
        self,
        sections: list[ModelContextSection],
    ) -> list[ModelContextSection]:
        """Classify sections into semantic categories using LLM.

        Processes sections sequentially to avoid overwhelming the LLM
        endpoint. Falls back to UNCATEGORIZED on any individual error.

        Args:
            sections: Parsed sections to classify.

        Returns:
            New list of sections with updated ``category`` field.
        """
        augmented: list[ModelContextSection] = []
        classified_count = 0

        for section in sections:
            category = await self._classify_section(section)
            augmented.append(section.with_category(category))
            if category != EnumContextSectionCategory.UNCATEGORIZED:
                classified_count += 1

        logger.info(
            "LLM augmentation complete: %d/%d sections classified",
            classified_count,
            len(sections),
        )
        return augmented

    async def _classify_section(
        self,
        section: ModelContextSection,
    ) -> EnumContextSectionCategory:
        """Classify a single section using the LLM.

        Args:
            section: Section to classify.

        Returns:
            Semantic category enum member.
        """
        content_preview = section.content[:500]
        prompt = _CATEGORY_PROMPT_TEMPLATE.format(
            heading=section.heading or "(preamble)",
            content_preview=content_preview,
        )

        try:
            response = await self._llm_fn(prompt)
            raw_category = response.strip().lower().replace('"', "").replace("'", "")

            # Try direct match
            if raw_category in _CATEGORY_MAP:
                return _CATEGORY_MAP[raw_category]

            # Try partial match (LLM might return extra text)
            for key, member in _CATEGORY_MAP.items():
                if key in raw_category:
                    return member

            logger.warning(
                "LLM returned unrecognized category '%s' for section '%s', "
                "falling back to UNCATEGORIZED",
                raw_category,
                section.heading or "(preamble)",
            )
            return EnumContextSectionCategory.UNCATEGORIZED

        except Exception:
            logger.warning(
                "LLM classification failed for section '%s', "
                "falling back to UNCATEGORIZED",
                section.heading or "(preamble)",
                exc_info=True,
            )
            return EnumContextSectionCategory.UNCATEGORIZED


__all__ = ["LlmInferenceFn", "ServiceLlmCategoryAugmenter"]
