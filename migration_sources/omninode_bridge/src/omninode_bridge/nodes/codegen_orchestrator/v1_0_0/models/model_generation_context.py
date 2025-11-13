#!/usr/bin/env python3
"""
Generation Workflow Context Model.

Shared state across LlamaIndex workflow execution for node code generation.

ONEX v2.0 Compliance:
- Model prefix naming: ModelGenerationContext
- Strong typing with Pydantic validation
- Integration with LlamaIndex Context
"""

from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .enum_pipeline_stage import EnumPipelineStage


class ModelGenerationContext(BaseModel):
    """
    Workflow context for node code generation pipeline.

    This model is stored in LlamaIndex Context.data and provides
    shared state across all pipeline stages.
    """

    # Workflow metadata
    workflow_id: UUID = Field(..., description="Unique workflow identifier")
    correlation_id: UUID = Field(..., description="Request correlation ID")
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # User request
    prompt: str = Field(..., description="Original user prompt")
    output_directory: str = Field(..., description="Target output directory")
    node_type_hint: Optional[str] = Field(
        None,
        description="User hint for node type (effect|orchestrator|reducer|compute)",
    )

    # User preferences
    interactive_mode: bool = Field(default=False)
    enable_intelligence: bool = Field(default=True)
    enable_quorum: bool = Field(default=False)

    # Current stage tracking
    current_stage: Optional[EnumPipelineStage] = None
    current_stage_start: Optional[datetime] = None
    completed_stages: list[str] = Field(default_factory=list)

    # Stage results (accumulated throughout pipeline)
    parsed_requirements: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage 1: Extracted requirements from prompt",
    )
    intelligence_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage 2: RAG intelligence results",
    )
    contract_yaml: str = Field(
        default="",
        description="Stage 3: Generated contract YAML",
    )
    generated_code: dict[str, str] = Field(
        default_factory=dict,
        description="Stage 4: Generated code files {filename: content}",
    )
    event_integration: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage 5: Event bus integration details",
    )
    validation_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage 6: Validation results (linting, type checking, tests)",
    )
    refinement_changes: dict[str, Any] = Field(
        default_factory=dict,
        description="Stage 7: Quality improvements applied",
    )
    written_files: list[str] = Field(
        default_factory=list,
        description="Stage 8: List of files written to disk",
    )

    # Performance tracking
    stage_durations: dict[str, float] = Field(
        default_factory=dict,
        description="Duration in seconds for each completed stage",
    )
    stage_warnings: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Warnings emitted by each stage",
    )

    # Quality metrics
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    test_coverage: Optional[float] = None
    complexity_score: Optional[float] = None

    # Model usage tracking
    primary_model: str = Field(default="gemini-2.5-flash")
    total_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)

    # Error tracking
    errors: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Errors encountered during pipeline execution",
    )

    def enter_stage(self, stage: EnumPipelineStage) -> None:
        """Mark entry into a new pipeline stage."""
        self.current_stage = stage
        self.current_stage_start = datetime.now(UTC)

    def complete_stage(
        self,
        stage: EnumPipelineStage,
        duration_seconds: float,
        warnings: Optional[list[str]] = None,
    ) -> None:
        """Mark completion of current pipeline stage."""
        self.completed_stages.append(stage.value)
        self.stage_durations[stage.value] = duration_seconds
        if warnings:
            self.stage_warnings[stage.value] = warnings
        self.current_stage = None
        self.current_stage_start = None

    def add_error(
        self, stage: str, error_message: str, error_context: dict[str, Any]
    ) -> None:
        """Add error to tracking."""
        self.errors.append(
            {
                "stage": stage,
                "message": error_message,
                "context": error_context,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    def get_total_duration(self) -> float:
        """Get total pipeline duration in seconds."""
        if not self.stage_durations:
            return 0.0
        return sum(self.stage_durations.values())

    def get_progress_percentage(self) -> float:
        """Get pipeline completion percentage (0.0-100.0)."""
        total_stages = len(EnumPipelineStage.all_stages())
        completed = len(self.completed_stages)
        return (completed / total_stages) * 100.0

    def is_complete(self) -> bool:
        """Check if all pipeline stages are complete."""
        all_stage_names = [s.value for s in EnumPipelineStage.all_stages()]
        return all(stage in self.completed_stages for stage in all_stage_names)
