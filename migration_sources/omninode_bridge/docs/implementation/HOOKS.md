# Hook Implementation Guide

## Overview

This guide provides step-by-step implementation of the Hook Intelligence System, including the HookReceiver service, hook registration patterns, and intelligence processing pipeline. Based on patterns discovered from `omnibase_3`, this system transforms basic service coordination into intelligent, self-learning orchestration.

## HookReceiver Service Implementation

### Project Structure
```
services/hook-receiver/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── events.py
│   │   ├── intelligence.py
│   │   └── sessions.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── hook_processor.py
│   │   ├── intelligence_analyzer.py
│   │   ├── session_manager.py
│   │   └── kafka_producer.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── hooks.py
│   │   ├── health.py
│   │   └── metrics.py
│   └── config/
│       ├── __init__.py
│       └── settings.py
├── requirements.txt
├── Dockerfile
└── tests/
    ├── __init__.py
    ├── test_hooks.py
    ├── test_intelligence.py
    └── test_integration.py
```

### Core Models

#### Event Models
```python
# app/models/events.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import uuid

class EventType(str, Enum):
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    SERVICE_HEALTH_CHANGE = "service_health_change"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    USER_INTERACTION = "user_interaction"
    GIT_WORKFLOW = "git_workflow"
    TOOL_REGISTRATION = "tool_registration"

class ServiceInfo(BaseModel):
    name: str
    version: str
    address: str
    port: int
    capabilities: List[str] = []
    endpoints: List[str] = []
    dependencies: List[str] = []
    metadata: Dict[str, Any] = {}

class ServiceLifecycleEvent(BaseModel):
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    service_info: ServiceInfo
    intelligence_metadata: Dict[str, Any] = {}

class ExecutionContext(BaseModel):
    session_id: str
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    service_name: str
    operation: str
    parameters: Dict[str, Any] = {}
    user_context: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ExecutionResult(BaseModel):
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: int
    resource_usage: Dict[str, Any] = {}

class ExecutionEvent(BaseModel):
    event_type: EventType
    correlation_id: str
    pre_context: ExecutionContext
    results: ExecutionResult
    intelligence_insights: Dict[str, Any] = {}

class UserInteractionEvent(BaseModel):
    event_type: EventType = EventType.USER_INTERACTION
    session_id: str
    user_id: Optional[str] = None
    interaction_type: str  # "request", "response", "error", "feedback"
    context: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class GitWorkflowEvent(BaseModel):
    event_type: EventType = EventType.GIT_WORKFLOW
    repository: str
    branch: str
    commit_hash: Optional[str] = None
    workflow_type: str  # "pre_commit", "post_commit", "pre_push", "post_push"
    changes: List[Dict[str, Any]] = []
    metadata: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)
```

#### Intelligence Models
```python
# app/models/intelligence.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

class PatternType(str, Enum):
    PERFORMANCE_PATTERN = "performance_pattern"
    FAILURE_PATTERN = "failure_pattern"
    USER_BEHAVIOR = "user_behavior"
    SERVICE_INTERACTION = "service_interaction"
    OPTIMIZATION_OPPORTUNITY = "optimization_opportunity"
    SECURITY_PATTERN = "security_pattern"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class IntelligencePattern(BaseModel):
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: PatternType
    pattern_data: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    frequency: int = 1
    first_seen: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    success_rate: float = Field(ge=0.0, le=1.0, default=0.0)

class IntelligenceInsight(BaseModel):
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_events: List[str]  # Correlation IDs
    insight_type: str
    insight_data: Dict[str, Any]
    confidence: ConfidenceLevel
    actionable_recommendations: List[str] = []
    expected_impact: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

class OptimizationRecommendation(BaseModel):
    recommendation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    target_service: str
    optimization_type: str
    recommendation_data: Dict[str, Any]
    expected_impact: Dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)
    priority: str = "medium"  # low, medium, high, critical
    implementation_complexity: str = "medium"  # low, medium, high
    created_at: datetime = Field(default_factory=datetime.utcnow)
```

### Core Service Implementation

#### HookReceiver Service
```python
# app/services/hook_processor.py
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Any, List
import asyncio
import json
import hashlib
from datetime import datetime, timedelta

from ..models.events import (
    ServiceLifecycleEvent, ExecutionEvent, UserInteractionEvent, GitWorkflowEvent
)
from ..models.intelligence import IntelligencePattern, IntelligenceInsight
from .intelligence_analyzer import IntelligenceAnalyzer
from .session_manager import SessionManager
from .kafka_producer import KafkaProducerService

class HookProcessor:
    """
    Core hook processing service with intelligence extraction.
    """

    def __init__(self):
        self.intelligence_analyzer = IntelligenceAnalyzer()
        self.session_manager = SessionManager()
        self.kafka_producer = KafkaProducerService()
        self.processing_queue = asyncio.Queue()
        self.background_tasks = set()

    async def initialize(self):
        """
        Initialize hook processor and start background workers.
        """
        # Start background intelligence processing
        task = asyncio.create_task(self._process_intelligence_queue())
        self.background_tasks.add(task)

        # Start periodic pattern analysis
        task = asyncio.create_task(self._periodic_pattern_analysis())
        self.background_tasks.add(task)

        await self.kafka_producer.initialize()

    async def process_service_lifecycle_hook(self, event: ServiceLifecycleEvent) -> Dict[str, Any]:
        """
        Process service lifecycle events with intelligence extraction.
        """
        try:
            # Extract immediate intelligence
            intelligence_data = await self.intelligence_analyzer.analyze_service_lifecycle(event)

            # Enhance event with intelligence metadata
            enhanced_event = event.copy(deep=True)
            enhanced_event.intelligence_metadata.update({
                "intelligence_extracted": True,
                "pattern_matches": intelligence_data.get("pattern_matches", []),
                "ecosystem_impact": intelligence_data.get("ecosystem_impact", {}),
                "processing_timestamp": datetime.utcnow().isoformat()
            })

            # Publish to Kafka for downstream processing
            await self.kafka_producer.send_message(
                topic="hooks.service_lifecycle",
                key=event.service_info.name,
                value=enhanced_event.dict()
            )

            # Queue for deep intelligence analysis
            await self.processing_queue.put({
                "type": "service_lifecycle",
                "event": enhanced_event,
                "intelligence_data": intelligence_data
            })

            # Update service registry (Consul integration)
            await self._update_service_registry(event.service_info, intelligence_data)

            return {
                "status": "processed",
                "intelligence_extracted": True,
                "pattern_matches": len(intelligence_data.get("pattern_matches", [])),
                "recommendations": intelligence_data.get("immediate_recommendations", [])
            }

        except Exception as e:
            await self._handle_processing_error(e, event, "service_lifecycle")
            raise HTTPException(status_code=500, detail=f"Hook processing failed: {str(e)}")

    async def process_execution_hook(self, event: ExecutionEvent) -> Dict[str, Any]:
        """
        Process execution events with performance and pattern analysis.
        """
        try:
            # Correlate with pre-execution context
            pre_context = await self.session_manager.get_execution_context(
                event.correlation_id
            )

            if not pre_context:
                # Create missing context for orphaned post-execution event
                pre_context = await self._reconstruct_execution_context(event)

            # Analyze execution intelligence
            intelligence_data = await self.intelligence_analyzer.analyze_execution_performance(
                event, pre_context
            )

            # Extract patterns
            patterns = await self.intelligence_analyzer.extract_execution_patterns(
                event, intelligence_data
            )

            # Generate insights
            insights = await self.intelligence_analyzer.generate_execution_insights(
                patterns, intelligence_data
            )

            # Enhanced event with intelligence
            enhanced_event = event.copy(deep=True)
            enhanced_event.intelligence_insights.update({
                "performance_analysis": intelligence_data.get("performance_analysis", {}),
                "pattern_matches": [p.dict() for p in patterns],
                "insights": [i.dict() for i in insights],
                "optimization_opportunities": intelligence_data.get("optimization_opportunities", [])
            })

            # Publish enhanced event
            await self.kafka_producer.send_message(
                topic="hooks.execution_intelligence",
                key=event.correlation_id,
                value=enhanced_event.dict()
            )

            # Store patterns and insights
            await self._store_intelligence_data(patterns, insights)

            # Generate real-time recommendations
            recommendations = await self._generate_real_time_recommendations(
                enhanced_event, intelligence_data
            )

            return {
                "status": "processed",
                "execution_time_ms": event.results.execution_time_ms,
                "patterns_detected": len(patterns),
                "insights_generated": len(insights),
                "recommendations": recommendations,
                "performance_score": intelligence_data.get("performance_score", 0.0)
            }

        except Exception as e:
            await self._handle_processing_error(e, event, "execution")
            raise HTTPException(status_code=500, detail=f"Execution hook processing failed: {str(e)}")

    async def process_user_interaction_hook(self, event: UserInteractionEvent) -> Dict[str, Any]:
        """
        Process user interaction events for behavior analysis.
        """
        try:
            # Analyze user behavior patterns
            behavior_analysis = await self.intelligence_analyzer.analyze_user_behavior(event)

            # Extract interaction patterns
            interaction_patterns = await self.intelligence_analyzer.extract_interaction_patterns(
                event, behavior_analysis
            )

            # Generate user experience insights
            ux_insights = await self.intelligence_analyzer.generate_ux_insights(
                event, behavior_analysis, interaction_patterns
            )

            # Enhanced event
            enhanced_event = event.copy(deep=True)
            enhanced_event.context.update({
                "behavior_analysis": behavior_analysis,
                "interaction_patterns": [p.dict() for p in interaction_patterns],
                "ux_insights": [i.dict() for i in ux_insights],
                "intelligence_processed": True
            })

            # Publish for downstream processing
            await self.kafka_producer.send_message(
                topic="hooks.user_interaction",
                key=event.session_id,
                value=enhanced_event.dict()
            )

            # Update user session intelligence
            await self.session_manager.update_session_intelligence(
                event.session_id,
                behavior_analysis,
                interaction_patterns
            )

            return {
                "status": "processed",
                "behavior_patterns": len(interaction_patterns),
                "ux_insights": len(ux_insights),
                "session_intelligence_updated": True
            }

        except Exception as e:
            await self._handle_processing_error(e, event, "user_interaction")
            raise HTTPException(status_code=500, detail=f"User interaction hook processing failed: {str(e)}")

    async def process_git_workflow_hook(self, event: GitWorkflowEvent) -> Dict[str, Any]:
        """
        Process git workflow events for development intelligence.
        """
        try:
            # Analyze development patterns
            dev_analysis = await self.intelligence_analyzer.analyze_development_patterns(event)

            # Extract code quality insights
            quality_insights = await self.intelligence_analyzer.extract_code_quality_insights(
                event, dev_analysis
            )

            # Generate deployment intelligence
            deployment_intelligence = await self.intelligence_analyzer.generate_deployment_intelligence(
                event, dev_analysis, quality_insights
            )

            # Enhanced event
            enhanced_event = event.copy(deep=True)
            enhanced_event.metadata.update({
                "development_analysis": dev_analysis,
                "quality_insights": [i.dict() for i in quality_insights],
                "deployment_intelligence": deployment_intelligence,
                "intelligence_processed": True
            })

            # Publish for CI/CD intelligence
            await self.kafka_producer.send_message(
                topic="hooks.git_intelligence",
                key=f"{event.repository}:{event.branch}",
                value=enhanced_event.dict()
            )

            # Store development patterns
            await self._store_development_patterns(
                event, dev_analysis, quality_insights, deployment_intelligence
            )

            return {
                "status": "processed",
                "code_quality_score": dev_analysis.get("quality_score", 0.0),
                "deployment_readiness": deployment_intelligence.get("readiness_score", 0.0),
                "quality_insights": len(quality_insights),
                "risk_assessment": deployment_intelligence.get("risk_level", "unknown")
            }

        except Exception as e:
            await self._handle_processing_error(e, event, "git_workflow")
            raise HTTPException(status_code=500, detail=f"Git workflow hook processing failed: {str(e)}")

    async def _process_intelligence_queue(self):
        """
        Background worker for deep intelligence processing.
        """
        while True:
            try:
                # Get next item from queue with timeout
                item = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=60.0
                )

                # Process based on type
                if item["type"] == "service_lifecycle":
                    await self._deep_service_analysis(item)
                elif item["type"] == "execution":
                    await self._deep_execution_analysis(item)
                elif item["type"] == "user_interaction":
                    await self._deep_behavior_analysis(item)
                elif item["type"] == "git_workflow":
                    await self._deep_development_analysis(item)

                # Mark task as done
                self.processing_queue.task_done()

            except asyncio.TimeoutError:
                # No items in queue, continue
                continue
            except Exception as e:
                await self._log_error(f"Intelligence queue processing error: {str(e)}")

    async def _periodic_pattern_analysis(self):
        """
        Periodic deep pattern analysis and optimization generation.
        """
        while True:
            try:
                # Wait for 5 minutes between analyses
                await asyncio.sleep(300)

                # Perform comprehensive pattern analysis
                await self._comprehensive_pattern_analysis()

                # Generate optimization recommendations
                await self._generate_optimization_recommendations()

                # Update predictive models
                await self._update_predictive_models()

            except Exception as e:
                await self._log_error(f"Periodic pattern analysis error: {str(e)}")

    async def _comprehensive_pattern_analysis(self):
        """
        Perform comprehensive analysis of all captured patterns.
        """
        # Analyze patterns across time windows
        time_windows = [
            ("1h", timedelta(hours=1)),
            ("6h", timedelta(hours=6)),
            ("24h", timedelta(hours=24)),
            ("7d", timedelta(days=7))
        ]

        for window_name, window_duration in time_windows:
            await self._analyze_patterns_in_window(window_name, window_duration)

        # Cross-pattern correlation analysis
        await self._analyze_cross_pattern_correlations()

        # Trend analysis
        await self._analyze_pattern_trends()

    async def _update_service_registry(self, service_info, intelligence_data):
        """
        Update Consul service registry with intelligence metadata.
        """
        # This would integrate with Consul to update service metadata
        # with intelligence insights for dynamic service discovery
        pass

    async def _handle_processing_error(self, error: Exception, event: Any, hook_type: str):
        """
        Handle processing errors with intelligence capture.
        """
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "hook_type": hook_type,
            "event_data": event.dict() if hasattr(event, 'dict') else str(event),
            "timestamp": datetime.utcnow().isoformat()
        }

        await self.kafka_producer.send_message(
            topic="hooks.processing_errors",
            key=f"error_{hook_type}",
            value=error_data
        )
```

#### Intelligence Analyzer
```python
# app/services/intelligence_analyzer.py
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta
import statistics
import json

from ..models.events import ServiceLifecycleEvent, ExecutionEvent, UserInteractionEvent
from ..models.intelligence import IntelligencePattern, IntelligenceInsight, PatternType

class IntelligenceAnalyzer:
    """
    Advanced intelligence analysis for pattern detection and insight generation.
    """

    def __init__(self):
        self.pattern_cache = {}
        self.analysis_cache = {}
        self.statistical_models = {}

    async def analyze_service_lifecycle(self, event: ServiceLifecycleEvent) -> Dict[str, Any]:
        """
        Analyze service lifecycle events for ecosystem intelligence.
        """
        service_name = event.service_info.name

        # Get historical service data
        historical_data = await self._get_service_historical_data(service_name)

        # Ecosystem impact analysis
        ecosystem_impact = await self._analyze_ecosystem_impact(event.service_info, historical_data)

        # Capability analysis
        capability_analysis = await self._analyze_service_capabilities(
            event.service_info.capabilities,
            historical_data
        )

        # Integration opportunity detection
        integration_opportunities = await self._detect_integration_opportunities(
            event.service_info,
            ecosystem_impact
        )

        # Pattern matching
        pattern_matches = await self._match_service_patterns(event, historical_data)

        return {
            "ecosystem_impact": ecosystem_impact,
            "capability_analysis": capability_analysis,
            "integration_opportunities": integration_opportunities,
            "pattern_matches": pattern_matches,
            "intelligence_score": await self._calculate_service_intelligence_score(
                ecosystem_impact, capability_analysis, pattern_matches
            ),
            "immediate_recommendations": await self._generate_immediate_service_recommendations(
                event, ecosystem_impact, capability_analysis
            )
        }

    async def analyze_execution_performance(self, event: ExecutionEvent, pre_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze execution performance for optimization insights.
        """
        # Performance metrics calculation
        performance_metrics = {
            "execution_time_ms": event.results.execution_time_ms,
            "success": event.results.success,
            "resource_usage": event.results.resource_usage
        }

        # Historical performance comparison
        historical_performance = await self._get_operation_historical_performance(
            event.pre_context.service_name,
            event.pre_context.operation
        )

        # Performance analysis
        performance_analysis = {
            "current_vs_average": self._compare_performance_vs_average(
                performance_metrics, historical_performance
            ),
            "trend_analysis": await self._analyze_performance_trend(
                event.pre_context.service_name,
                event.pre_context.operation,
                performance_metrics
            ),
            "bottleneck_analysis": await self._analyze_bottlenecks(
                performance_metrics, pre_context
            ),
            "resource_efficiency": self._calculate_resource_efficiency(
                performance_metrics, pre_context
            )
        }

        # Optimization opportunities
        optimization_opportunities = await self._identify_optimization_opportunities(
            performance_analysis, historical_performance
        )

        # Performance score
        performance_score = self._calculate_performance_score(
            performance_analysis, optimization_opportunities
        )

        return {
            "performance_analysis": performance_analysis,
            "optimization_opportunities": optimization_opportunities,
            "performance_score": performance_score,
            "historical_comparison": historical_performance,
            "anomaly_detection": await self._detect_performance_anomalies(
                performance_metrics, historical_performance
            )
        }

    async def extract_execution_patterns(self, event: ExecutionEvent, intelligence_data: Dict[str, Any]) -> List[IntelligencePattern]:
        """
        Extract execution patterns for learning and optimization.
        """
        patterns = []

        # Performance patterns
        performance_pattern = await self._extract_performance_pattern(event, intelligence_data)
        if performance_pattern:
            patterns.append(performance_pattern)

        # Failure patterns (if execution failed)
        if not event.results.success:
            failure_pattern = await self._extract_failure_pattern(event, intelligence_data)
            if failure_pattern:
                patterns.append(failure_pattern)

        # Usage patterns
        usage_pattern = await self._extract_usage_pattern(event, intelligence_data)
        if usage_pattern:
            patterns.append(usage_pattern)

        # Resource utilization patterns
        resource_pattern = await self._extract_resource_pattern(event, intelligence_data)
        if resource_pattern:
            patterns.append(resource_pattern)

        return patterns

    async def generate_execution_insights(self, patterns: List[IntelligencePattern], intelligence_data: Dict[str, Any]) -> List[IntelligenceInsight]:
        """
        Generate actionable insights from execution patterns.
        """
        insights = []

        # Performance optimization insights
        perf_insight = await self._generate_performance_insight(patterns, intelligence_data)
        if perf_insight:
            insights.append(perf_insight)

        # Resource optimization insights
        resource_insight = await self._generate_resource_insight(patterns, intelligence_data)
        if resource_insight:
            insights.append(resource_insight)

        # Failure prevention insights
        failure_prevention_insight = await self._generate_failure_prevention_insight(
            patterns, intelligence_data
        )
        if failure_prevention_insight:
            insights.append(failure_prevention_insight)

        # User experience insights
        ux_insight = await self._generate_ux_insight(patterns, intelligence_data)
        if ux_insight:
            insights.append(ux_insight)

        return insights

    async def _extract_performance_pattern(self, event: ExecutionEvent, intelligence_data: Dict[str, Any]) -> Optional[IntelligencePattern]:
        """
        Extract performance patterns from execution data.
        """
        performance_analysis = intelligence_data.get("performance_analysis", {})

        # Check if this represents a significant performance pattern
        if not self._is_significant_performance_pattern(performance_analysis):
            return None

        pattern_data = {
            "operation": event.pre_context.operation,
            "service": event.pre_context.service_name,
            "execution_time_ms": event.results.execution_time_ms,
            "performance_tier": self._classify_performance_tier(event.results.execution_time_ms),
            "resource_usage": event.results.resource_usage,
            "success": event.results.success,
            "context_parameters": event.pre_context.parameters,
            "trend_indicator": performance_analysis.get("trend_analysis", {}).get("trend", "stable"),
            "optimization_potential": performance_analysis.get("optimization_opportunities", [])
        }

        confidence = self._calculate_pattern_confidence(pattern_data, intelligence_data)

        return IntelligencePattern(
            pattern_type=PatternType.PERFORMANCE_PATTERN,
            pattern_data=pattern_data,
            confidence=confidence
        )

    def _calculate_performance_score(self, performance_analysis: Dict[str, Any], optimization_opportunities: List[Dict[str, Any]]) -> float:
        """
        Calculate overall performance score (0.0 to 1.0).
        """
        # Base score from performance comparison
        current_vs_avg = performance_analysis.get("current_vs_average", {})
        base_score = max(0.0, min(1.0, 2.0 - current_vs_avg.get("relative_performance", 1.0)))

        # Penalty for resource inefficiency
        resource_efficiency = performance_analysis.get("resource_efficiency", 1.0)
        efficiency_factor = min(1.0, resource_efficiency)

        # Penalty for detected bottlenecks
        bottlenecks = performance_analysis.get("bottleneck_analysis", {}).get("bottlenecks", [])
        bottleneck_penalty = len(bottlenecks) * 0.1

        # Penalty for optimization opportunities
        optimization_penalty = len(optimization_opportunities) * 0.05

        final_score = base_score * efficiency_factor - bottleneck_penalty - optimization_penalty
        return max(0.0, min(1.0, final_score))
```

### FastAPI Application
```python
# app/main.py
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .services.hook_processor import HookProcessor
from .models.events import ServiceLifecycleEvent, ExecutionEvent, UserInteractionEvent, GitWorkflowEvent
from .api import hooks, health, metrics
from .config.settings import get_settings

# Global hook processor instance
hook_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global hook_processor

    # Startup
    hook_processor = HookProcessor()
    await hook_processor.initialize()
    logging.info("HookReceiver service initialized")

    yield

    # Shutdown
    if hook_processor and hook_processor.background_tasks:
        for task in hook_processor.background_tasks:
            task.cancel()
    logging.info("HookReceiver service shutdown complete")

app = FastAPI(
    title="HookReceiver Intelligence Service",
    description="Intelligent hook processing for OmniNode Bridge",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(hooks.router, prefix="/hooks", tags=["hooks"])
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])

def get_hook_processor() -> HookProcessor:
    """Dependency to get hook processor instance."""
    if hook_processor is None:
        raise HTTPException(status_code=503, detail="Hook processor not initialized")
    return hook_processor

# Root endpoint
@app.get("/")
async def root():
    return {
        "service": "HookReceiver Intelligence Service",
        "version": "0.1.0",
        "status": "operational",
        "capabilities": [
            "service_lifecycle_hooks",
            "execution_intelligence",
            "user_interaction_analysis",
            "git_workflow_intelligence",
            "pattern_detection",
            "optimization_recommendations"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.service_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
```

### API Endpoints
```python
# app/api/hooks.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any

from ..services.hook_processor import HookProcessor
from ..models.events import ServiceLifecycleEvent, ExecutionEvent, UserInteractionEvent, GitWorkflowEvent
from ..main import get_hook_processor

router = APIRouter()

@router.post("/service-lifecycle", response_model=Dict[str, Any])
async def service_lifecycle_hook(
    event: ServiceLifecycleEvent,
    background_tasks: BackgroundTasks,
    processor: HookProcessor = Depends(get_hook_processor)
):
    """
    Process service lifecycle events with intelligence extraction.
    """
    try:
        result = await processor.process_service_lifecycle_hook(event)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execution", response_model=Dict[str, Any])
async def execution_hook(
    event: ExecutionEvent,
    background_tasks: BackgroundTasks,
    processor: HookProcessor = Depends(get_hook_processor)
):
    """
    Process execution events with performance and pattern analysis.
    """
    try:
        result = await processor.process_execution_hook(event)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/user-interaction", response_model=Dict[str, Any])
async def user_interaction_hook(
    event: UserInteractionEvent,
    background_tasks: BackgroundTasks,
    processor: HookProcessor = Depends(get_hook_processor)
):
    """
    Process user interaction events for behavior analysis.
    """
    try:
        result = await processor.process_user_interaction_hook(event)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/git-workflow", response_model=Dict[str, Any])
async def git_workflow_hook(
    event: GitWorkflowEvent,
    background_tasks: BackgroundTasks,
    processor: HookProcessor = Depends(get_hook_processor)
):
    """
    Process git workflow events for development intelligence.
    """
    try:
        result = await processor.process_git_workflow_hook(event)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patterns", response_model=Dict[str, Any])
async def get_patterns(
    pattern_type: str = None,
    service_name: str = None,
    limit: int = 100,
    processor: HookProcessor = Depends(get_hook_processor)
):
    """
    Retrieve detected patterns with filtering options.
    """
    try:
        patterns = await processor.intelligence_analyzer.get_patterns(
            pattern_type=pattern_type,
            service_name=service_name,
            limit=limit
        )
        return {
            "patterns": [p.dict() for p in patterns],
            "count": len(patterns),
            "filters_applied": {
                "pattern_type": pattern_type,
                "service_name": service_name,
                "limit": limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/insights", response_model=Dict[str, Any])
async def get_insights(
    service_name: str = None,
    insight_type: str = None,
    limit: int = 50,
    processor: HookProcessor = Depends(get_hook_processor)
):
    """
    Retrieve generated insights with filtering options.
    """
    try:
        insights = await processor.intelligence_analyzer.get_insights(
            service_name=service_name,
            insight_type=insight_type,
            limit=limit
        )
        return {
            "insights": [i.dict() for i in insights],
            "count": len(insights),
            "filters_applied": {
                "service_name": service_name,
                "insight_type": insight_type,
                "limit": limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations", response_model=Dict[str, Any])
async def get_recommendations(
    target_service: str = None,
    optimization_type: str = None,
    priority: str = None,
    limit: int = 20,
    processor: HookProcessor = Depends(get_hook_processor)
):
    """
    Retrieve optimization recommendations with filtering options.
    """
    try:
        recommendations = await processor.intelligence_analyzer.get_recommendations(
            target_service=target_service,
            optimization_type=optimization_type,
            priority=priority,
            limit=limit
        )
        return {
            "recommendations": [r.dict() for r in recommendations],
            "count": len(recommendations),
            "filters_applied": {
                "target_service": target_service,
                "optimization_type": optimization_type,
                "priority": priority,
                "limit": limit
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Deployment Configuration

#### Dockerfile
```dockerfile
# services/hook-receiver/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Run application
CMD ["python", "-m", "app.main"]
```

#### Requirements
```txt
# services/hook-receiver/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
aiokafka==0.8.11
asyncpg==0.29.0
redis==5.0.1
consul-python==1.1.0
sqlalchemy[asyncio]==2.0.23
alembic==1.13.0
structlog==23.2.0
prometheus-client==0.19.0
numpy==1.26.2
scikit-learn==1.3.2
```

This comprehensive hook implementation guide provides the foundation for the intelligent Hook Intelligence System that transforms basic service coordination into self-learning orchestration.
