#!/usr/bin/env python3
"""
Enhanced Context Builder for LLM Code Generation.

Aggregates context from multiple Phase 3 sources to build comprehensive
LLM prompts for business logic generation.

This module integrates:
- PatternMatcher (C10): Production pattern matching
- PatternFormatter (I2): Pattern formatting for LLM
- VariantSelector (C8): Template variant selection
- MixinRecommender (C12-C15): Mixin recommendations
- ReferenceExtractor: Similar node implementations

Performance Target: <50ms to build complete context
Token Budget: ≤8K tokens (with prioritized truncation)
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader

# Import Phase 3 components
from metadata_stamping.code_gen.patterns.models import EnumNodeType
from metadata_stamping.code_gen.patterns.pattern_matcher import PatternMatcher
from metadata_stamping.code_gen.templates.variant_selector import VariantSelector

# Import models
from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements

from ..models import ModelLLMContext
from .pattern_formatter import PatternFormatter

logger = logging.getLogger(__name__)


# Node type descriptions
NODE_TYPE_DESCRIPTIONS = {
    "EFFECT": (
        "Effect nodes perform I/O operations with external systems "
        "(database, API, file system, message queues). They must handle "
        "timeouts, retries, and connection failures gracefully."
    ),
    "COMPUTE": (
        "Compute nodes perform pure computation with no side effects. "
        "They must be deterministic, fast (<10ms typical), and memory-efficient. "
        "No external I/O except for logging."
    ),
    "REDUCER": (
        "Reducer nodes aggregate streaming data with state management. "
        "They must handle high throughput (>1000 items/sec), maintain "
        "partial results, and emit aggregations efficiently."
    ),
    "ORCHESTRATOR": (
        "Orchestrator nodes coordinate multi-step workflows. "
        "They must manage FSM state transitions, invoke sub-nodes, "
        "handle rollback/compensation, and track complex error scenarios."
    ),
}


class EnhancedContextBuilder:
    """
    Build comprehensive LLM context from multiple Phase 3 sources.

    Aggregates context from:
    - Template variant selection (VariantSelector)
    - Production pattern matching (PatternMatcher + PatternFormatter)
    - Mixin recommendations (MixinRecommender)
    - Reference implementations (similar nodes)
    - ONEX compliance rules
    - Error handling patterns

    Attributes:
        pattern_matcher: PatternMatcher for finding relevant patterns
        pattern_formatter: PatternFormatter for formatting patterns
        variant_selector: VariantSelector for template variant selection
        template_env: Jinja2 environment for rendering templates
        token_budget: Maximum tokens allowed (default: 8000)

    Example:
        >>> builder = EnhancedContextBuilder()
        >>> context = builder.build_context(
        ...     requirements=requirements,
        ...     operation={"name": "execute_effect", ...},
        ...     node_type="EFFECT",
        ... )
        >>> print(f"Context: {context.total_tokens} tokens")
    """

    def __init__(
        self,
        pattern_matcher: Optional[PatternMatcher] = None,
        pattern_formatter: Optional[PatternFormatter] = None,
        variant_selector: Optional[VariantSelector] = None,
        token_budget: int = 8000,
    ):
        """
        Initialize enhanced context builder.

        Args:
            pattern_matcher: PatternMatcher instance (creates if None)
            pattern_formatter: PatternFormatter instance (creates if None)
            variant_selector: VariantSelector instance (creates if None)
            token_budget: Maximum token budget for context
        """
        self.pattern_matcher = pattern_matcher or PatternMatcher()
        self.pattern_formatter = pattern_formatter or PatternFormatter()
        self.variant_selector = variant_selector or VariantSelector()
        self.token_budget = token_budget

        # Setup Jinja2 template environment
        self.template_env = self._setup_template_env()

        logger.info(f"EnhancedContextBuilder initialized (token_budget={token_budget})")

    def _setup_template_env(self) -> Environment:
        """
        Setup Jinja2 environment for prompt templates.

        Returns:
            Configured Jinja2 Environment
        """
        # Template directory is relative to this file
        template_dir = Path(__file__).parent / "prompt_templates"

        if not template_dir.exists():
            raise FileNotFoundError(
                f"Prompt templates directory not found: {template_dir}"
            )

        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,  # We're generating code, not HTML
        )

        logger.debug(f"Jinja2 environment loaded from: {template_dir}")

        return env

    def build_context(
        self,
        requirements: ModelPRDRequirements,
        operation: dict[str, Any],
        node_type: str,
    ) -> ModelLLMContext:
        """
        Build comprehensive LLM context from all sources.

        Main entry point for context building. Aggregates all Phase 3
        intelligence into structured LLM context.

        Args:
            requirements: PRD requirements (from PRDAnalyzer)
            operation: Operation specification dict with:
                - name: str (method name)
                - description: str
                - input_model: str
                - output_model: str
                - ... other operation details
            node_type: Node type string (EFFECT/COMPUTE/etc.)

        Returns:
            ModelLLMContext with all sections populated

        Performance: <50ms target
        """
        start_time = time.perf_counter()

        logger.info(
            f"Building context for {node_type}.{operation.get('name', 'unknown')}"
        )

        # Step 1: Select template variant
        variant_selection = self.variant_selector.select_variant(
            node_type=EnumNodeType[node_type.upper()],
            operation_count=len(requirements.operations),
            required_features=set(requirements.features),
        )

        logger.debug(
            f"Selected variant: {variant_selection.variant.value} "
            f"(confidence: {variant_selection.confidence:.2f})"
        )

        # Step 2: Match production patterns
        pattern_matches = self.pattern_matcher.match_patterns(
            node_type=EnumNodeType[node_type.upper()],
            required_features=set(requirements.features),
            top_k=5,
            min_score=0.3,
        )

        logger.debug(f"Matched {len(pattern_matches)} patterns")

        # Step 3: Find similar nodes (stub for now)
        similar_nodes = self._find_similar_nodes(
            node_type=node_type,
            features=set(requirements.features),
        )

        # Step 4: Extract reference implementations (stub for now)
        reference_impls = self._get_reference_implementations(
            similar_nodes=similar_nodes,
            operation_name=operation.get("name", ""),
        )

        # Step 5: Build context sections
        sections = self._build_context_sections(
            requirements=requirements,
            operation=operation,
            node_type=node_type,
            variant_selection=variant_selection,
            pattern_matches=pattern_matches,
            reference_impls=reference_impls,
        )

        # Step 6: Manage token budget
        managed_sections, truncation_applied = self._manage_token_budget(
            sections=sections,
            budget=self.token_budget,
        )

        # Step 7: Build final context
        total_tokens = self._estimate_tokens(managed_sections)

        context = ModelLLMContext(
            system_context=managed_sections["system_context"],
            operation_spec=managed_sections["operation_spec"],
            production_patterns=managed_sections["production_patterns"],
            reference_implementations=managed_sections["reference_impls"],
            constraints=managed_sections["constraints"],
            generation_instructions=managed_sections["instructions"],
            total_tokens=total_tokens,
            truncation_applied=truncation_applied,
            node_type=node_type,
            method_name=operation.get("name", "unknown"),
            variant_selected=variant_selection.variant.value,
            patterns_included=len(pattern_matches),
            references_included=len(reference_impls),
        )

        build_time_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Context built: {total_tokens} tokens, {build_time_ms:.1f}ms, "
            f"{len(pattern_matches)} patterns, {len(reference_impls)} references"
        )

        if truncation_applied:
            logger.warning(f"Context truncated to fit {self.token_budget} token budget")

        return context

    def _build_context_sections(
        self,
        requirements: ModelPRDRequirements,
        operation: dict[str, Any],
        node_type: str,
        variant_selection: Any,  # ModelTemplateSelection
        pattern_matches: list[Any],  # list[ModelPatternMatch]
        reference_impls: list[dict[str, Any]],
    ) -> dict[str, str]:
        """
        Build all context sections using Jinja2 templates.

        Args:
            requirements: PRD requirements
            operation: Operation specification
            node_type: Node type
            variant_selection: Selected template variant
            pattern_matches: Matched production patterns
            reference_impls: Reference implementations

        Returns:
            Dictionary mapping section name to rendered content
        """
        sections = {}

        # 1. System Context
        sections["system_context"] = self._render_template(
            "system_context.j2",
            {
                "node_type": node_type.upper(),
                "node_type_description": NODE_TYPE_DESCRIPTIONS.get(
                    node_type.upper(), "Custom node type"
                ),
                "variant": variant_selection.variant.value,
                "variant_confidence": variant_selection.confidence,
                "variant_rationale": variant_selection.rationale,
                "variant_features": variant_selection.matched_features,
                "onex_version": "2.0",
            },
        )

        # 2. Operation Specification
        sections["operation_spec"] = self._render_template(
            "operation_spec.j2",
            {
                "method_name": operation.get("name", "execute_operation"),
                "method_signature": operation.get("signature", ""),
                "method_docstring": operation.get("docstring"),
                "input_model": operation.get("input_model", "ModelContractEffect"),
                "input_model_description": operation.get(
                    "input_description",
                    "Contains operation parameters and configuration",
                ),
                "output_model": operation.get("output_model", "ModelContractResponse"),
                "output_model_description": operation.get(
                    "output_description", "Contains operation results and metadata"
                ),
                "business_purpose": requirements.business_description,
                "operations": requirements.operations,
                "features": requirements.features,
                "performance_requirements": requirements.performance_requirements,
            },
        )

        # 3. Production Patterns (pre-formatted by PatternFormatter)
        formatted_patterns = self.pattern_formatter.format_patterns_for_llm(
            patterns=pattern_matches,
            max_tokens=2000,  # Initial budget, may be truncated later
        )

        sections["production_patterns"] = self._render_template(
            "pattern_context.j2",
            {
                "formatted_patterns": formatted_patterns,
                "pattern_count": len(pattern_matches),
                "total_patterns_available": len(
                    pattern_matches
                ),  # TODO: Get from registry
            },
        )

        # 4. Reference Implementations
        sections["reference_impls"] = self._render_template(
            "reference_context.j2",
            {
                "references": reference_impls,
                "has_references": len(reference_impls) > 0,
            },
        )

        # 5. Constraints
        sections["constraints"] = self._render_template(
            "constraints.j2",
            {
                "mixins": self._get_mixin_recommendations(requirements),
                "error_codes": self._get_error_code_guidance(),
                "security_rules": self._get_security_rules(),
                "performance_targets": requirements.performance_requirements,
                "onex_compliance": self._get_onex_compliance_rules(),
            },
        )

        # 6. Generation Instructions
        sections["instructions"] = self._render_template(
            "generation_instructions.j2",
            {
                "method_name": operation.get("name", "execute_operation"),
                "indentation_level": 8,  # Standard method body indentation
                "quality_requirements": self._get_quality_requirements(),
            },
        )

        return sections

    def _render_template(
        self,
        template_name: str,
        context: dict[str, Any],
    ) -> str:
        """
        Render a Jinja2 template with given context.

        Args:
            template_name: Template filename (e.g., "system_context.j2")
            context: Template variables

        Returns:
            Rendered template string
        """
        try:
            template = self.template_env.get_template(template_name)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            return f"# Error rendering {template_name}: {e}\n"

    def _find_similar_nodes(
        self,
        node_type: str,
        features: set[str],
    ) -> list[dict[str, Any]]:
        """
        Find similar production nodes.

        TODO: Implement actual node similarity search using:
        - Node type matching
        - Feature overlap (Jaccard similarity)
        - Semantic embedding similarity (optional)

        Args:
            node_type: Target node type
            features: Required features

        Returns:
            List of similar node metadata dicts
        """
        # Stub implementation - return empty list
        # TODO: Implement using node registry / codebase search
        logger.debug(
            f"Searching for similar nodes (type={node_type}, features={len(features)})"
        )
        return []

    def _get_reference_implementations(
        self,
        similar_nodes: list[dict[str, Any]],
        operation_name: str,
    ) -> list[dict[str, Any]]:
        """
        Extract reference implementations from similar nodes.

        TODO: Implement actual implementation extraction:
        - Read node source files
        - Extract relevant methods using AST parsing
        - Generate adaptation guidance

        Args:
            similar_nodes: List of similar node metadata
            operation_name: Target operation name

        Returns:
            List of reference implementation dicts
        """
        # Stub implementation - return empty list
        # TODO: Implement using AST parsing and code extraction
        logger.debug(
            f"Extracting references for {operation_name} from {len(similar_nodes)} nodes"
        )
        return []

    def _get_mixin_recommendations(
        self,
        requirements: ModelPRDRequirements,
    ) -> list[dict[str, Any]]:
        """
        Get mixin recommendations.

        TODO: Integrate with MixinRecommender (C12-C15) once available.

        Args:
            requirements: PRD requirements

        Returns:
            List of mixin recommendation dicts
        """
        # Stub implementation
        # TODO: Call MixinRecommender.recommend_mixins()
        return [
            {
                "name": "MixinErrorHandling",
                "confidence": 0.95,
                "provides": ["error_handling", "logging"],
                "usage_notes": "Provides standard error handling patterns",
            }
        ]

    def _get_error_code_guidance(self) -> list[dict[str, str]]:
        """Get error code usage guidance."""
        return [
            {
                "code": "INVALID_INPUT",
                "description": "Input validation failed",
                "use_when": "Missing required fields, invalid types, out-of-range values",
            },
            {
                "code": "TIMEOUT_ERROR",
                "description": "Operation timed out",
                "use_when": "I/O operation exceeded timeout threshold",
            },
            {
                "code": "EXECUTION_ERROR",
                "description": "Runtime execution failed",
                "use_when": "Unexpected errors during operation execution",
            },
            {
                "code": "INVALID_STATE",
                "description": "Invalid state for operation",
                "use_when": "Resources not initialized, invalid state transitions",
            },
        ]

    def _get_security_rules(self) -> list[str]:
        """Get security constraint rules."""
        return [
            "NO hardcoded credentials - use environment variables",
            "NO SQL injection - use parameterized queries",
            "NO sensitive data in logs - mask secrets and tokens",
            "Validate ALL inputs before processing",
            "Use HTTPS for external API calls",
            "Implement rate limiting for public endpoints",
        ]

    def _get_onex_compliance_rules(self) -> list[str]:
        """Get ONEX v2.0 compliance rules."""
        return [
            "All methods MUST be async (use async def)",
            "Use emit_log_event for all logging (no print/logger)",
            "Include type hints on all parameters and returns",
            "Add docstrings to all public methods",
            "Track and return metrics in response (latency_ms)",
            "Include correlation_id in all log events",
            "Wrap all exceptions in ModelOnexError",
            "Use proper error codes (EnumCoreErrorCode)",
        ]

    def _get_quality_requirements(self) -> list[str]:
        """Get code quality requirements."""
        return [
            "Use async/await throughout",
            "Include comprehensive error handling",
            "Emit log events (INFO and ERROR)",
            "Use type hints for all variables",
            "Follow ONEX patterns from examples",
            "Use recommended mixins",
            "Track and return execution metrics",
            "NO import statements (already available)",
            "NO hardcoded secrets or credentials",
        ]

    def _manage_token_budget(
        self,
        sections: dict[str, str],
        budget: int,
    ) -> tuple[dict[str, str], bool]:
        """
        Manage token budget with prioritized truncation.

        Truncation priority (truncate in this order):
        1. Reference implementations (trim from 2500 → 1500 tokens)
        2. Production patterns (trim from 2000 → 1500 tokens)
        3. Constraints (trim from 500 → 300 tokens)

        Never truncate:
        - System context
        - Operation specification
        - Generation instructions

        Args:
            sections: Dictionary of section name → content
            budget: Maximum token budget

        Returns:
            Tuple of (managed_sections, truncation_applied)
        """
        # Estimate tokens for each section
        section_tokens = {
            name: self._estimate_tokens_single(content)
            for name, content in sections.items()
        }

        total_tokens = sum(section_tokens.values())

        if total_tokens <= budget:
            # No truncation needed
            return sections, False

        logger.warning(
            f"Token budget exceeded: {total_tokens} > {budget}. Applying truncation."
        )

        managed = sections.copy()
        truncation_applied = False

        # Truncate reference implementations first
        if (
            "reference_impls" in managed
            and section_tokens.get("reference_impls", 0) > 1500
        ):
            managed["reference_impls"] = self._truncate_section(
                managed["reference_impls"], max_tokens=1500
            )
            section_tokens["reference_impls"] = 1500
            truncation_applied = True

        # Recalculate total
        total_tokens = sum(section_tokens.values())

        # Truncate patterns if still over budget
        if (
            total_tokens > budget
            and section_tokens.get("production_patterns", 0) > 1500
        ):
            managed["production_patterns"] = self._truncate_section(
                managed["production_patterns"], max_tokens=1500
            )
            section_tokens["production_patterns"] = 1500
            truncation_applied = True

        # Recalculate total
        total_tokens = sum(section_tokens.values())

        # Truncate constraints if still over budget
        if total_tokens > budget and section_tokens.get("constraints", 0) > 300:
            managed["constraints"] = self._truncate_section(
                managed["constraints"], max_tokens=300
            )
            section_tokens["constraints"] = 300
            truncation_applied = True

        return managed, truncation_applied

    def _estimate_tokens(self, sections: dict[str, str]) -> int:
        """
        Estimate total tokens across all sections.

        Uses simple heuristic: 1 token ≈ 4 characters

        Args:
            sections: Dictionary of sections

        Returns:
            Estimated token count
        """
        total_chars = sum(len(content) for content in sections.values())
        return total_chars // 4

    def _estimate_tokens_single(self, text: str) -> int:
        """
        Estimate tokens for a single text.

        Args:
            text: Text content

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def _truncate_section(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token budget.

        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated text
        """
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # Truncate and add notice
        truncated = text[:max_chars]
        truncated += "\n\n(Section truncated to fit token budget)\n"

        return truncated


__all__ = ["EnhancedContextBuilder"]
