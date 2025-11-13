"""
Phase 4 Workflows Integration.

This module provides:
- CodeGenerationWorkflow: Integrated orchestration of all 4 workflow components
- AI Quorum: 4-model consensus validation for code generation quality assurance
- Template Management: LRU-cached template loading and rendering with 85-95% hit rate target
- Staged Parallel Execution: 6-phase code generation pipeline with parallel contract processing
- Validation Pipeline: Multi-stage validation with completeness, quality, and ONEX compliance
- Error Recovery Orchestration: Multi-strategy error recovery with pattern matching (Pattern 7)
- Performance Optimization: Profiling and optimization achieving 2-3x speedup vs Phase 3

Phase 4 Weeks 5-6: Workflows Phase - Components 1, 2, 3, & 4 + Integration
Phase 4 Weeks 7-8: Optimization Phase - Error Recovery Orchestration (Pattern 7) + Performance Optimization
"""

from omninode_bridge.agents.workflows.ai_quorum import AIQuorum
from omninode_bridge.agents.workflows.code_generation_workflow import (
    CodeGenerationWorkflow,
)
from omninode_bridge.agents.workflows.error_recovery import ErrorRecoveryOrchestrator
from omninode_bridge.agents.workflows.llm_client import (
    CodestralClient,
    GeminiClient,
    GLMClient,
    LLMClient,
)
from omninode_bridge.agents.workflows.optimization_models import (
    IOPerformanceStats,
    MemoryUsageStats,
    OptimizationArea,
    OptimizationPriority,
    OptimizationRecommendation,
    ParallelExecutionStats,
    PerformanceReport,
    ProfileMetricType,
    ProfileResult,
)
from omninode_bridge.agents.workflows.optimization_models import (
    TemplateCacheStats as OptimizationTemplateCacheStats,
)
from omninode_bridge.agents.workflows.performance_optimizer import PerformanceOptimizer
from omninode_bridge.agents.workflows.profiling import PerformanceProfiler
from omninode_bridge.agents.workflows.quorum_models import (
    ModelConfig,
    QuorumResult,
    QuorumVote,
)
from omninode_bridge.agents.workflows.recovery_models import (
    ErrorPattern,
    ErrorType,
    RecoveryContext,
    RecoveryResult,
    RecoveryStatistics,
    RecoveryStrategy,
)
from omninode_bridge.agents.workflows.recovery_strategies import (
    AlternativePathStrategy,
    ErrorCorrectionStrategy,
    EscalationStrategy,
    GracefulDegradationStrategy,
    RetryStrategy,
)
from omninode_bridge.agents.workflows.staged_execution import StagedParallelExecutor
from omninode_bridge.agents.workflows.template_cache import TemplateLRUCache
from omninode_bridge.agents.workflows.template_manager import TemplateManager
from omninode_bridge.agents.workflows.template_models import (
    Template,
    TemplateCacheStats,
    TemplateMetadata,
    TemplateRenderContext,
    TemplateType,
)
from omninode_bridge.agents.workflows.validation_models import (
    ValidationContext,
    ValidationResult,
    ValidationSummary,
)
from omninode_bridge.agents.workflows.validation_pipeline import ValidationPipeline
from omninode_bridge.agents.workflows.workflow_models import (
    EnumStageStatus,
    EnumStepType,
    StageResult,
    StepResult,
    WorkflowConfig,
    WorkflowResult,
    WorkflowStage,
    WorkflowStep,
)

__all__ = [
    # Integrated Workflow (Main Entry Point)
    "CodeGenerationWorkflow",
    # AI Quorum components
    "AIQuorum",
    "LLMClient",
    "GeminiClient",
    "GLMClient",
    "CodestralClient",
    "ModelConfig",
    "QuorumVote",
    "QuorumResult",
    # Template Management components
    "Template",
    "TemplateType",
    "TemplateMetadata",
    "TemplateRenderContext",
    "TemplateCacheStats",
    "TemplateLRUCache",
    "TemplateManager",
    # Validation Pipeline components
    "ValidationPipeline",
    "ValidationResult",
    "ValidationContext",
    "ValidationSummary",
    # Staged Parallel Execution components
    "StagedParallelExecutor",
    "WorkflowStage",
    "WorkflowStep",
    "WorkflowConfig",
    "WorkflowResult",
    "StageResult",
    "StepResult",
    "EnumStageStatus",
    "EnumStepType",
    # Error Recovery Orchestration components (Pattern 7)
    "ErrorRecoveryOrchestrator",
    "ErrorPattern",
    "ErrorType",
    "RecoveryContext",
    "RecoveryResult",
    "RecoveryStatistics",
    "RecoveryStrategy",
    "RetryStrategy",
    "AlternativePathStrategy",
    "GracefulDegradationStrategy",
    "ErrorCorrectionStrategy",
    "EscalationStrategy",
    # Performance Optimization components (Weeks 7-8)
    "PerformanceOptimizer",
    "PerformanceProfiler",
    "ProfileResult",
    "OptimizationRecommendation",
    "PerformanceReport",
    "OptimizationArea",
    "OptimizationPriority",
    "ProfileMetricType",
    "OptimizationTemplateCacheStats",
    "ParallelExecutionStats",
    "MemoryUsageStats",
    "IOPerformanceStats",
]
