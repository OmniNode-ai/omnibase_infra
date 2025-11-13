#!/usr/bin/env python3
"""
Pipeline Stage Enumeration for NodeCodegenOrchestrator.

Defines the 9-stage generation pipeline with target execution times.

ONEX v2.0 Compliance:
- Enum-based naming: EnumPipelineStage
- Clear stage definitions with performance targets
- Integration with LlamaIndex Workflows
"""

from enum import Enum


class EnumPipelineStage(str, Enum):
    """
    9-Stage generation pipeline for ONEX node code generation.

    Total Target Time: 56 seconds

    Stage Breakdown:
    1. Prompt parsing (5s) - Parse user prompt and extract requirements
    2. Intelligence gathering (3s) - Query RAG for patterns and best practices
    3. Contract building (2s) - Generate ONEX v2.0 compliant contract YAML
    4. Code generation (10-15s) - Generate node implementation
    5. Event bus integration (2s) - Wire up Kafka event publishing
    6. Validation (5s) - Run linting, type checking, basic tests
    7. Refinement (3s) - Apply quality improvements based on validation
    8. File writing (3s) - Write all generated files to disk
    9. Test generation (3s) - Generate comprehensive test files
    """

    PROMPT_PARSING = "prompt_parsing"
    """Stage 1: Parse natural language prompt and extract requirements (Target: 5s)"""

    INTELLIGENCE_GATHERING = "intelligence_gathering"
    """Stage 2: Query RAG for relevant patterns and best practices (Target: 3s)"""

    CONTRACT_BUILDING = "contract_building"
    """Stage 3: Generate ONEX v2.0 compliant contract YAML (Target: 2s)"""

    CODE_GENERATION = "code_generation"
    """Stage 4: Generate node implementation code (Target: 10-15s)"""

    EVENT_BUS_INTEGRATION = "event_bus_integration"
    """Stage 5: Wire up Kafka event publishing (Target: 2s)"""

    VALIDATION = "validation"
    """Stage 6: Run linting, type checking, basic tests (Target: 5s)"""

    REFINEMENT = "refinement"
    """Stage 7: Apply quality improvements (Target: 3s)"""

    FILE_WRITING = "file_writing"
    """Stage 8: Write all generated files to disk (Target: 3s)"""

    TEST_GENERATION = "test_generation"
    """Stage 9: Generate comprehensive test files (Target: 3s)"""

    @property
    def target_duration_seconds(self) -> float:
        """Get target execution time for this stage in seconds."""
        return self._stage_durations().get(self, 5.0)

    @property
    def stage_number(self) -> int:
        """Get stage number (1-9)."""
        stages = list(EnumPipelineStage)
        return stages.index(self) + 1

    @property
    def description(self) -> str:
        """Get human-readable stage description."""
        return self._stage_descriptions().get(self, "Unknown stage")

    @staticmethod
    def _stage_durations() -> dict["EnumPipelineStage", float]:
        """Map stages to target durations in seconds."""
        return {
            EnumPipelineStage.PROMPT_PARSING: 5.0,
            EnumPipelineStage.INTELLIGENCE_GATHERING: 3.0,
            EnumPipelineStage.CONTRACT_BUILDING: 2.0,
            EnumPipelineStage.CODE_GENERATION: 12.5,  # Average of 10-15s
            EnumPipelineStage.EVENT_BUS_INTEGRATION: 2.0,
            EnumPipelineStage.VALIDATION: 5.0,
            EnumPipelineStage.REFINEMENT: 3.0,
            EnumPipelineStage.FILE_WRITING: 3.0,
            EnumPipelineStage.TEST_GENERATION: 3.0,
        }

    @staticmethod
    def _stage_descriptions() -> dict["EnumPipelineStage", str]:
        """Map stages to human-readable descriptions."""
        return {
            EnumPipelineStage.PROMPT_PARSING: "Parsing user prompt and extracting requirements",
            EnumPipelineStage.INTELLIGENCE_GATHERING: "Querying RAG for patterns and best practices",
            EnumPipelineStage.CONTRACT_BUILDING: "Generating ONEX v2.0 contract YAML",
            EnumPipelineStage.CODE_GENERATION: "Generating node implementation code",
            EnumPipelineStage.EVENT_BUS_INTEGRATION: "Wiring up Kafka event publishing",
            EnumPipelineStage.VALIDATION: "Running linting, type checking, and tests",
            EnumPipelineStage.REFINEMENT: "Applying quality improvements",
            EnumPipelineStage.FILE_WRITING: "Writing generated files to disk",
            EnumPipelineStage.TEST_GENERATION: "Generating comprehensive test files",
        }

    @staticmethod
    def all_stages() -> list["EnumPipelineStage"]:
        """Get all pipeline stages in execution order."""
        return [
            EnumPipelineStage.PROMPT_PARSING,
            EnumPipelineStage.INTELLIGENCE_GATHERING,
            EnumPipelineStage.CONTRACT_BUILDING,
            EnumPipelineStage.CODE_GENERATION,
            EnumPipelineStage.EVENT_BUS_INTEGRATION,
            EnumPipelineStage.VALIDATION,
            EnumPipelineStage.REFINEMENT,
            EnumPipelineStage.FILE_WRITING,
            EnumPipelineStage.TEST_GENERATION,
        ]

    @staticmethod
    def total_target_duration() -> float:
        """Get total target duration for entire pipeline in seconds."""
        return sum(
            stage.target_duration_seconds for stage in EnumPipelineStage.all_stages()
        )
