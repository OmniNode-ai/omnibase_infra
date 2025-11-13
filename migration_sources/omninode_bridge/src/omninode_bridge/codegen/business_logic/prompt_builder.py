#!/usr/bin/env python3
"""
Prompt Builder for LLM Business Logic Generation.

Constructs context-rich prompts for LLM-based code generation by:
1. Gathering context from PRD requirements
2. Loading similar patterns from RAG/KB (optional)
3. Formatting contract specifications
4. Applying ONEX best practices
5. Building structured system and user prompts

ONEX v2.0 Compliance:
- Separation of concerns (prompt building isolated)
- Template-based prompt construction
- Optional intelligence integration
- Token estimation for cost control
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Optional

try:
    # Try relative import first (when imported as package)
    from .models import GenerationContext, PromptPair, StubInfo
except ImportError:
    # Fall back to absolute import (for testing and standalone use)
    from omninode_bridge.codegen.business_logic.models import (
        GenerationContext,
        PromptPair,
        StubInfo,
    )

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds context-rich prompts for LLM business logic generation.

    Context Sources:
    1. PRD requirements (what the method should do)
    2. Contract specification (input/output types, constraints)
    3. Similar code patterns (from RAG/KB)
    4. ONEX best practices (architectural patterns)
    5. Node type specifics (Effect/Compute/Reducer/Orchestrator)

    Usage:
        builder = PromptBuilder(rag_client=archon_client)
        prompts = await builder.build_prompt(context, stub_info)
        system_prompt, user_prompt = prompts.system_prompt, prompts.user_prompt
    """

    # ONEX best practices by node type
    ONEX_BEST_PRACTICES: ClassVar[dict[str, list[str]]] = {
        "effect": [
            "Use async/await for all I/O operations",
            "Implement circuit breaker pattern for external calls",
            "Add retry logic with exponential backoff",
            "Validate external responses before returning",
            "Use connection pooling for database/API clients",
            "Set appropriate timeouts for external calls",
            "Log all external interactions with correlation_id",
        ],
        "compute": [
            "Implement pure functions without side effects",
            "Avoid external I/O operations",
            "Optimize for performance and readability",
            "Use type hints for all parameters and returns",
            "Return deterministic results for same inputs",
            "Handle edge cases and null values",
            "Document complex algorithms with comments",
        ],
        "reducer": [
            "Use streaming aggregation for large datasets",
            "Implement incremental state updates",
            "Support multiple aggregation strategies",
            "Manage memory efficiently (generators, chunking)",
            "Persist state periodically or on completion",
            "Handle late-arriving data appropriately",
            "Use proper data structures for aggregation",
        ],
        "orchestrator": [
            "Implement FSM patterns for workflow states",
            "Use event-driven state transitions",
            "Handle concurrent sub-tasks properly",
            "Implement timeout and deadline handling",
            "Provide workflow visibility via events",
            "Support workflow pause/resume if needed",
            "Handle partial failures gracefully",
        ],
    }

    # Error handling patterns
    ERROR_HANDLING_PATTERNS: ClassVar[list[str]] = [
        "Use ModelOnexError for domain errors with error_code",
        "Catch specific exceptions before generic Exception",
        "Log errors with emit_log_event(EnumLogLevel.ERROR, ...)",
        "Include correlation_id in all error logs",
        "Provide actionable error messages",
        "Clean up resources in finally blocks",
        "Re-raise unknown exceptions after logging",
    ]

    def __init__(
        self,
        rag_client: Optional[Any] = None,
        kb_client: Optional[Any] = None,
        templates_dir: Optional[Path] = None,
    ):
        """
        Initialize PromptBuilder.

        Args:
            rag_client: Optional RAG client for similar pattern lookup (Archon MCP)
            kb_client: Optional knowledge base client
            templates_dir: Optional custom templates directory
        """
        self.rag_client = rag_client
        self.kb_client = kb_client

        # Load templates
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = templates_dir
        self.system_template = self._load_template("system_prompt.txt")
        self.user_template = self._load_template("user_prompt.txt")

    def _load_template(self, filename: str) -> str:
        """Load prompt template from file."""
        template_path = self.templates_dir / filename
        if not template_path.exists():
            logger.warning(f"Template not found: {template_path}, using default")
            return ""

        return template_path.read_text()

    async def build_prompt(
        self, context: GenerationContext, stub_info: StubInfo
    ) -> PromptPair:
        """
        Build system and user prompts for code generation.

        Args:
            context: Generation context (PRD, contract, etc.)
            stub_info: Information about the stub to replace

        Returns:
            PromptPair with system_prompt and user_prompt
        """
        # Gather similar patterns if RAG client available
        if self.rag_client:
            similar_patterns = await self._gather_similar_patterns(
                context.node_type, stub_info.method_name
            )
            if similar_patterns:
                context.similar_patterns.extend(similar_patterns)

        # Add ONEX best practices for node type
        node_type_lower = context.node_type.lower()
        if node_type_lower in self.ONEX_BEST_PRACTICES:
            context.best_practices.extend(self.ONEX_BEST_PRACTICES[node_type_lower])

        # Add error handling patterns
        context.error_handling_patterns.extend(self.ERROR_HANDLING_PATTERNS)

        # Build system prompt
        system_prompt = self._build_system_prompt(context)

        # Build user prompt
        user_prompt = self._build_user_prompt(context, stub_info)

        # Estimate tokens (rough: ~4 chars per token)
        estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4

        return PromptPair(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            estimated_tokens=estimated_tokens,
        )

    def _build_system_prompt(self, context: GenerationContext) -> str:
        """Build system prompt from template."""
        return self.system_template.format(node_type=context.node_type.capitalize())

    def _build_user_prompt(
        self, context: GenerationContext, stub_info: StubInfo
    ) -> str:
        """Build user prompt from template with all context."""
        # Format similar patterns
        similar_patterns_str = self._format_list(
            context.similar_patterns, "No similar patterns available"
        )

        # Format best practices
        best_practices_str = self._format_list(
            context.best_practices[:8], "Follow ONEX v2.0 patterns"
        )

        # Format error handling patterns
        error_handling_str = self._format_list(
            context.error_handling_patterns[:7], "Use standard error handling"
        )

        # Format contract spec
        contract_spec_str = self._format_contract_spec(context.contract_spec)

        # Format performance requirements
        perf_req_str = self._format_dict(
            context.performance_requirements, "No specific requirements"
        )

        # Format operations and features
        operations_str = (
            ", ".join(context.operations)
            if context.operations
            else "General operations"
        )
        features_str = (
            ", ".join(context.features) if context.features else "Standard features"
        )

        # Format docstring
        docstring_str = (
            stub_info.docstring if stub_info.docstring else "No docstring provided"
        )

        return self.user_template.format(
            service_name=context.service_name,
            node_type=context.node_type,
            business_description=context.business_description,
            operations=operations_str,
            features=features_str,
            method_signature=stub_info.signature,
            method_docstring=docstring_str,
            contract_spec=contract_spec_str,
            performance_requirements=perf_req_str,
            similar_patterns=similar_patterns_str,
            best_practices=best_practices_str,
            error_handling_patterns=error_handling_str,
            stub_code=stub_info.stub_code,
        )

    async def _gather_similar_patterns(
        self, node_type: str, method_name: str
    ) -> list[str]:
        """
        Query RAG for similar code patterns.

        This method will integrate with Archon MCP when available.
        For now, returns empty list (patterns come from context).

        Args:
            node_type: Node type (effect/compute/reducer/orchestrator)
            method_name: Method name to find patterns for

        Returns:
            List of similar code pattern descriptions
        """
        if not self.rag_client:
            return []

        try:
            # TODO: Integrate with Archon MCP
            # Example query:
            # patterns = await self.rag_client.search_code_patterns(
            #     query=f"{node_type} {method_name} implementation",
            #     limit=3
            # )
            logger.debug(f"RAG lookup for {node_type}.{method_name} (not implemented)")
            return []

        except Exception as e:
            logger.warning(f"Failed to gather similar patterns: {e}")
            return []

    def _format_contract_spec(self, contract_spec: dict[str, Any]) -> str:
        """
        Format contract specification for prompt.

        Args:
            contract_spec: Contract specification dictionary

        Returns:
            Formatted contract specification string
        """
        if not contract_spec:
            return "No contract specification provided"

        lines = ["Contract fields and types:"]

        # Format input fields
        if "input" in contract_spec:
            lines.append("\n**Input Fields**:")
            for field, info in contract_spec.get("input", {}).items():
                field_type = info.get("type", "Any")
                description = info.get("description", "")
                required = "required" if info.get("required", False) else "optional"
                lines.append(f"  - {field}: {field_type} ({required}) - {description}")

        # Format output fields
        if "output" in contract_spec:
            lines.append("\n**Output Fields**:")
            for field, info in contract_spec.get("output", {}).items():
                field_type = info.get("type", "Any")
                description = info.get("description", "")
                lines.append(f"  - {field}: {field_type} - {description}")

        # Format constraints
        if "constraints" in contract_spec:
            lines.append("\n**Constraints**:")
            for constraint in contract_spec.get("constraints", []):
                lines.append(f"  - {constraint}")

        return "\n".join(lines)

    def _format_list(self, items: list[str], empty_msg: str = "None") -> str:
        """Format list as bullet points."""
        if not items:
            return empty_msg

        return "\n".join(f"- {item}" for item in items)

    def _format_dict(self, data: dict[str, Any], empty_msg: str = "None") -> str:
        """Format dictionary as key-value pairs."""
        if not data:
            return empty_msg

        lines = []
        for key, value in data.items():
            lines.append(f"- {key}: {value}")

        return "\n".join(lines)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses rough approximation: ~4 characters per token.
        For more accurate estimation, integrate tiktoken library.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        return len(text) // 4
