#!/usr/bin/env python3
"""
Models for business logic generation.

Defines data structures for prompt building, stub information,
and generation context.

ONEX v2.0 Compliance:
- Pydantic v2 models for type safety
- Field validation and descriptions
- Structured context for LLM generation
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

# Import at runtime for model_rebuild() to work
from ..template_engine import ModelGeneratedArtifacts


class StubInfo(BaseModel):
    """Information about a code stub to replace."""

    file_path: str = Field(..., description="Path to file containing stub")
    method_name: str = Field(..., description="Method name (e.g., execute_effect)")
    stub_code: str = Field(..., description="Current stub implementation")
    line_start: int = Field(..., ge=1, description="Starting line number")
    line_end: int = Field(..., ge=1, description="Ending line number")
    signature: str = Field(..., description="Full method signature with type hints")
    docstring: Optional[str] = Field(None, description="Method docstring if present")


class GenerationContext(BaseModel):
    """Context for business logic generation."""

    # PRD context
    node_type: str = Field(
        ..., description="Node type (effect/compute/reducer/orchestrator)"
    )
    service_name: str = Field(..., description="Service name")
    business_description: str = Field(..., description="What this node does")
    operations: list[str] = Field(
        default_factory=list, description="Operations to implement"
    )
    features: list[str] = Field(default_factory=list, description="Key features")

    # Contract specifics
    contract_spec: dict[str, Any] = Field(
        default_factory=dict, description="Contract specification"
    )
    performance_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Performance requirements"
    )

    # Code patterns (from RAG/KB)
    similar_patterns: list[str] = Field(
        default_factory=list, description="Similar code patterns from intelligence"
    )
    best_practices: list[str] = Field(
        default_factory=list, description="ONEX best practices"
    )

    # Error handling
    error_handling_patterns: list[str] = Field(
        default_factory=list, description="Error handling patterns"
    )


class PromptPair(BaseModel):
    """Pair of system and user prompts for LLM."""

    system_prompt: str = Field(..., description="System prompt defining role and rules")
    user_prompt: str = Field(
        ..., description="User prompt with specific task and context"
    )
    estimated_tokens: int = Field(default=0, ge=0, description="Estimated token count")


class ModelMethodStub(BaseModel):
    """Method stub extracted from template."""

    method_name: str = Field(..., description="Method name (e.g., execute_effect)")
    signature: str = Field(..., description="Full method signature")
    docstring: Optional[str] = Field(None, description="Existing docstring")
    line_number: int = Field(..., ge=1, description="Line number in file")
    needs_implementation: bool = Field(
        default=True, description="Whether stub needs LLM generation"
    )


class ModelBusinessLogicContext(BaseModel):
    """Context for business logic generation."""

    # PRD context
    node_type: str = Field(
        ..., description="Node type (effect/compute/reducer/orchestrator)"
    )
    service_name: str = Field(..., description="Service name")
    business_description: str = Field(..., description="What this node does")
    operations: list[str] = Field(..., description="Operations to implement")
    features: list[str] = Field(..., description="Key features")

    # Method context
    method_name: str = Field(..., description="Method to generate")
    method_signature: str = Field(..., description="Method signature with types")
    method_docstring: Optional[str] = None

    # Code patterns (from RAG)
    similar_patterns: list[str] = Field(
        default_factory=list, description="Similar code patterns from intelligence"
    )
    best_practices: list[str] = Field(
        default_factory=list, description="ONEX best practices"
    )

    # Contract specifics
    performance_requirements: dict[str, Any] = Field(
        default_factory=dict, description="Performance requirements"
    )
    error_handling_patterns: list[str] = Field(
        default_factory=list, description="Error handling patterns"
    )


class ModelGeneratedMethod(BaseModel):
    """Generated method implementation."""

    method_name: str
    generated_code: str = Field(..., description="Generated implementation")

    # Quality metrics
    syntax_valid: bool = Field(..., description="AST parsing succeeded")
    onex_compliant: bool = Field(..., description="ONEX patterns followed")
    has_type_hints: bool = Field(..., description="Type hints present")
    has_docstring: bool = Field(..., description="Docstring present")
    security_issues: list[str] = Field(
        default_factory=list, description="Security issues found"
    )

    # LLM metrics
    tokens_used: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)
    latency_ms: float = Field(..., ge=0.0)
    model_used: str


class ModelEnhancedArtifacts(BaseModel):
    """Enhanced artifacts with LLM-generated business logic."""

    # Original artifacts
    original_artifacts: "ModelGeneratedArtifacts" = Field(
        ..., description="Original generated artifacts (ModelGeneratedArtifacts)"
    )

    # Enhanced node file
    enhanced_node_file: str = Field(
        ..., description="Node file with LLM implementations"
    )

    # Generation details
    methods_generated: list[ModelGeneratedMethod] = Field(default_factory=list)
    total_tokens_used: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    total_latency_ms: float = Field(default=0.0, ge=0.0)
    generation_success_rate: float = Field(default=1.0, ge=0.0, le=1.0)


class ModelLLMContext(BaseModel):
    """
    Complete LLM context for code generation (Phase 3 Enhanced).

    This model represents the full context assembled from multiple sources:
    - System context (ONEX v2.0 guidelines, node type info)
    - Operation specification (method signature, I/O models)
    - Production patterns (from PatternMatcher)
    - Reference implementations (from similar nodes)
    - Constraints (mixins, error handling, security)
    - Generation instructions (output format, quality requirements)

    Attributes:
        system_context: ONEX and node type information
        operation_spec: Method specification and business context
        production_patterns: Formatted patterns from PatternMatcher
        reference_implementations: Code examples from similar nodes
        constraints: Requirements and restrictions
        generation_instructions: Output format and quality requirements
        total_tokens: Estimated total token count
        truncation_applied: Whether sections were truncated to fit budget
        node_type: Node type being generated
        method_name: Method being implemented
        variant_selected: Template variant selected
        patterns_included: Number of patterns included
        references_included: Number of reference implementations
    """

    # Context sections
    system_context: str = Field(..., description="ONEX v2.0 and node type context")
    operation_spec: str = Field(..., description="Method specification")
    production_patterns: str = Field(..., description="Formatted production patterns")
    reference_implementations: str = Field(..., description="Similar node examples")
    constraints: str = Field(..., description="Requirements and restrictions")
    generation_instructions: str = Field(..., description="Output format instructions")

    # Metadata
    total_tokens: int = Field(..., ge=0, description="Total estimated tokens")
    truncation_applied: bool = Field(
        default=False, description="Whether truncation was applied"
    )

    # Context metadata
    node_type: str = Field(..., description="Node type (EFFECT/COMPUTE/etc.)")
    method_name: str = Field(..., description="Method being implemented")
    variant_selected: str = Field(..., description="Template variant selected")
    patterns_included: int = Field(default=0, ge=0, description="Pattern count")
    references_included: int = Field(default=0, ge=0, description="Reference count")

    def to_prompt(self) -> str:
        """
        Convert context to single prompt string.

        Concatenates all sections in order.

        Returns:
            Complete prompt string for LLM
        """
        sections = [
            self.system_context,
            self.operation_spec,
            self.production_patterns,
            self.reference_implementations,
            self.constraints,
            self.generation_instructions,
        ]

        return "\n\n".join(section.strip() for section in sections if section.strip())


class ModelContextBuildingMetrics(BaseModel):
    """Metrics for context building operation."""

    build_time_ms: float = Field(..., ge=0.0, description="Time to build context")
    total_tokens: int = Field(..., ge=0, description="Total token count")
    section_tokens: dict[str, int] = Field(
        default_factory=dict, description="Tokens per section"
    )
    patterns_included: int = Field(default=0, ge=0, description="Patterns included")
    patterns_available: int = Field(default=0, ge=0, description="Patterns available")
    references_included: int = Field(default=0, ge=0, description="References included")
    references_available: int = Field(
        default=0, ge=0, description="References available"
    )
    truncation_applied: bool = Field(
        default=False, description="Whether truncation occurred"
    )
    truncated_sections: list[str] = Field(
        default_factory=list, description="Sections that were truncated"
    )
    quality_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Context quality score"
    )


# Rebuild model to resolve forward references (Pydantic v2 requirement)
# ModelGeneratedArtifacts is imported from template_engine.py
ModelEnhancedArtifacts.model_rebuild()
