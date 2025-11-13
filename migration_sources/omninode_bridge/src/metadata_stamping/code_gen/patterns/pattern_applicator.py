"""
Pattern Applicator for Code Generation.

This module handles applying production patterns to Jinja2 templates,
injecting code snippets, and resolving pattern dependencies.

Performance Target: <5ms per pattern application
"""

import logging
from typing import Any, Optional

from jinja2 import Environment, Template

from .models import EnumNodeType, ModelPatternMetadata

logger = logging.getLogger(__name__)


class PatternApplicationError(Exception):
    """Base exception for pattern application errors."""


class PatternApplicator:
    """
    Apply production patterns to code generation templates.

    This class handles injecting patterns into Jinja2 templates, resolving
    prerequisites, and managing pattern dependencies.

    Attributes:
        jinja_env: Jinja2 environment for template rendering
        _applied_patterns: Track which patterns have been applied

    Performance:
        - Pattern application: <5ms per pattern
        - Template injection: <2ms
        - Prerequisite resolution: <1ms
    """

    def __init__(self, jinja_env: Optional[Environment] = None):
        """
        Initialize pattern applicator.

        Args:
            jinja_env: Optional Jinja2 environment (creates default if None)
        """
        if jinja_env is None:
            jinja_env = Environment(
                autoescape=False,  # Code generation, not HTML
                trim_blocks=True,
                lstrip_blocks=True,
            )

        self.jinja_env = jinja_env
        self._applied_patterns: set[str] = set()

        logger.debug("PatternApplicator initialized")

    def apply_pattern(
        self,
        pattern: ModelPatternMetadata,
        template_context: dict[str, Any],
        node_type: EnumNodeType,
    ) -> dict[str, Any]:
        """
        Apply a pattern to a template context.

        This adds pattern-specific variables to the template context
        that can be used during code generation.

        Args:
            pattern: Pattern to apply
            template_context: Current template context (will be modified)
            node_type: Target node type

        Returns:
            Updated template context with pattern variables

        Raises:
            PatternApplicationError: If application fails

        Performance: <5ms per pattern
        """
        # Validate node type compatibility
        if node_type not in pattern.applicable_to:
            raise PatternApplicationError(
                f"Pattern '{pattern.name}' not applicable to {node_type}"
            )

        # Check if already applied
        if pattern.pattern_id in self._applied_patterns:
            logger.debug(f"Pattern {pattern.pattern_id} already applied, skipping")
            return template_context

        logger.debug(f"Applying pattern: {pattern.name} to {node_type}")

        # Add pattern to context
        pattern_key = f"pattern_{pattern.name}"
        template_context[pattern_key] = {
            "enabled": True,
            "code": pattern.code_template,
            "config": pattern.configuration,
            "prerequisites": pattern.prerequisites,
        }

        # Add pattern-specific variables
        template_context.setdefault("patterns", []).append(pattern.name)
        template_context.setdefault("pattern_configs", {})[
            pattern.name
        ] = pattern.configuration

        # Track applied pattern
        self._applied_patterns.add(pattern.pattern_id)

        return template_context

    def apply_multiple_patterns(
        self,
        patterns: list[ModelPatternMetadata],
        template_context: dict[str, Any],
        node_type: EnumNodeType,
    ) -> dict[str, Any]:
        """
        Apply multiple patterns to a template context.

        Patterns are applied in order, with prerequisites resolved.

        Args:
            patterns: List of patterns to apply
            template_context: Current template context
            node_type: Target node type

        Returns:
            Updated template context

        Performance: <5ms * len(patterns)
        """
        # Sort by category priority (structure first, then others)
        sorted_patterns = self._sort_by_priority(patterns)

        for pattern in sorted_patterns:
            try:
                template_context = self.apply_pattern(
                    pattern, template_context, node_type
                )
            except PatternApplicationError as e:
                logger.warning(
                    f"Failed to apply pattern {pattern.name}: {e}, continuing..."
                )
                continue

        return template_context

    def inject_pattern_code(
        self,
        pattern: ModelPatternMetadata,
        target_location: str,
        context_vars: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Generate code snippet from a pattern's template.

        This renders the pattern's code_template with provided variables.

        Args:
            pattern: Pattern to inject
            target_location: Where to inject (e.g., "imports", "init", "execute")
            context_vars: Optional variables for template rendering

        Returns:
            Rendered code snippet

        Raises:
            PatternApplicationError: If injection fails

        Performance: <2ms per injection
        """
        if not pattern.code_template:
            logger.warning(f"Pattern {pattern.name} has no code template")
            return ""

        try:
            # Prepare context
            context = {
                "pattern": pattern,
                "location": target_location,
                **(context_vars or {}),
            }

            # Render template
            template = Template(pattern.code_template)
            code = template.render(**context)

            logger.debug(
                f"Injected pattern {pattern.name} at {target_location} "
                f"({len(code)} chars)"
            )

            return code

        except Exception as e:
            raise PatternApplicationError(
                f"Failed to inject pattern {pattern.name}: {e}"
            ) from e

    def resolve_prerequisites(
        self,
        pattern: ModelPatternMetadata,
        available_imports: set[str],
    ) -> tuple[bool, list[str]]:
        """
        Check if pattern prerequisites are satisfied.

        Args:
            pattern: Pattern to check
            available_imports: Set of already imported modules/classes

        Returns:
            Tuple of (prerequisites_met, missing_prerequisites)

        Performance: <1ms
        """
        missing = []

        for prereq in pattern.prerequisites:
            # Simple check: is the prerequisite imported?
            # More sophisticated check could parse import statements
            if not any(
                prereq_part in available_imports for prereq_part in prereq.split()
            ):
                missing.append(prereq)

        prerequisites_met = len(missing) == 0

        if missing:
            logger.debug(f"Pattern {pattern.name} missing prerequisites: {missing}")

        return prerequisites_met, missing

    def get_required_imports(self, patterns: list[ModelPatternMetadata]) -> list[str]:
        """
        Extract all required imports from a list of patterns.

        Args:
            patterns: Patterns to analyze

        Returns:
            List of import statements needed

        Performance: <1ms
        """
        imports = set()

        for pattern in patterns:
            for prereq in pattern.prerequisites:
                # Convert prerequisites to import statements
                # Example: "ModelOnexError from omnibase_core" -> "from omnibase_core import ModelOnexError"
                if " from " in prereq:
                    parts = prereq.split(" from ")
                    if len(parts) == 2:
                        class_name, module = parts
                        imports.add(
                            f"from {module.strip()} import {class_name.strip()}"
                        )
                elif " imported from " in prereq:
                    parts = prereq.split(" imported from ")
                    if len(parts) == 2:
                        class_name, module = parts
                        imports.add(
                            f"from {module.strip()} import {class_name.strip()}"
                        )

        return sorted(imports)

    def reset(self) -> None:
        """Reset applicator state (useful for testing)."""
        self._applied_patterns.clear()
        logger.debug("PatternApplicator reset")

    def _sort_by_priority(
        self, patterns: list[ModelPatternMetadata]
    ) -> list[ModelPatternMetadata]:
        """
        Sort patterns by priority for application order.

        Priority order:
        1. Structure patterns (imports, class declaration, init)
        2. Integration patterns (Consul, Kafka, lifecycle)
        3. Resilience patterns (error handling, health checks)
        4. Observability patterns (logging, metrics, events)
        5. Configuration patterns (config loading, env fallback)
        6. Security patterns
        7. Performance patterns

        Args:
            patterns: Patterns to sort

        Returns:
            Sorted list of patterns
        """
        category_priority = {
            "structure": 1,
            "integration": 2,
            "resilience": 3,
            "observability": 4,
            "configuration": 5,
            "security": 6,
            "performance": 7,
        }

        # Within structure category, apply specific ordering
        structure_order = {
            "standard_imports": 1,
            "class_declaration": 2,
            "initialization_pattern": 3,
            "type_hints": 4,
        }

        def sort_key(pattern: ModelPatternMetadata) -> tuple[int, int, str]:
            category_rank = category_priority.get(pattern.category.value, 99)

            # Special handling for structure patterns
            if pattern.category.value == "structure":
                structure_rank = structure_order.get(pattern.name, 50)
            else:
                structure_rank = 0

            return (category_rank, structure_rank, pattern.name)

        return sorted(patterns, key=sort_key)
