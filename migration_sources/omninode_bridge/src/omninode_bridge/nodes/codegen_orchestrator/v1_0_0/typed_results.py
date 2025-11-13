"""
Type definitions for code generation orchestrator results.

Provides TypedDict definitions for generation results to eliminate Any types
in public APIs.
"""

from typing import TypedDict


class GenerationResult(TypedDict, total=False):
    """
    Code generation workflow result structure.

    Used in NodeCodegenOrchestrator.execute_orchestration return type.

    Required Fields:
        total_duration_seconds: Total time for generation workflow
        generated_files: List of file paths generated
        node_type: Node type (effect|orchestrator|reducer|compute)
        service_name: Service name for the generated node

    Optional Fields:
        quality_score: Overall quality score (0-1)
        test_coverage: Test coverage percentage (0-1)
        complexity_score: Cyclomatic complexity score
        patterns_applied: List of RAG patterns applied
        intelligence_sources: List of intelligence sources used (Qdrant|Memgraph|...)
        primary_model: Primary AI model used
        total_tokens: Total tokens consumed
        total_cost_usd: Total cost in USD
        contract_yaml: Generated contract YAML content
        node_module: Generated node module content
        models: List of generated model file paths
        enums: List of generated enum file paths
        tests: List of generated test file paths
    """

    # Required fields
    total_duration_seconds: float
    generated_files: list[str]
    node_type: str
    service_name: str

    # Optional quality metrics
    quality_score: float
    test_coverage: float
    complexity_score: float

    # Optional intelligence metadata
    patterns_applied: list[str]
    intelligence_sources: list[str]

    # Optional model performance
    primary_model: str
    total_tokens: int
    total_cost_usd: float

    # Optional generated artifacts
    contract_yaml: str
    node_module: str
    models: list[str]
    enums: list[str]
    tests: list[str]
