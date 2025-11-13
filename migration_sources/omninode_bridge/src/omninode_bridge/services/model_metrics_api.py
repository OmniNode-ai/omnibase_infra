"""Model metrics and performance comparison API service."""

import os
from datetime import UTC, datetime, timedelta
from typing import Any

from circuitbreaker import circuit
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ..models.model_metrics import ModelTier, TaskType, global_metrics
from ..security.audit_logger import AuditEventType, AuditSeverity, get_audit_logger
from .smart_responder_integration import smart_responder_client

# Rate limiting will be configured in create_app function

# Initialize audit logger
audit_logger = get_audit_logger("model_metrics_api", "0.1.0")


class TaskRequest(BaseModel):
    """Request to execute a task with model metrics tracking."""

    task_type: TaskType
    prompt: str
    context_size: int | None = None
    complexity: str = "moderate"
    max_latency_ms: float | None = None
    preferred_model: str | None = None
    track_metrics: bool = True


class TaskResponse(BaseModel):
    """Response from task execution with metrics."""

    success: bool
    response: str
    model_used: str
    node_location: str
    execution_metrics: dict[str, Any]
    recommendations: list[dict[str, Any]]
    escalated: bool = False


class ModelComparisonRequest(BaseModel):
    """Request to compare models on a specific task."""

    task_type: TaskType
    prompt: str
    models_to_compare: list[str] | None = (
        None  # If None, compares representative models from each tier
    )
    context_size: int | None = None
    complexity: str = "moderate"


class ModelComparisonResult(BaseModel):
    """Results from model comparison."""

    task_type: TaskType
    prompt_summary: str
    comparison_results: list[dict[str, Any]]
    best_model: dict[str, Any]
    worst_model: dict[str, Any]
    analysis: str


class ModelMetricsAPI:
    """FastAPI service for model metrics and performance comparison."""

    def __init__(self):
        self.app = FastAPI(
            title="OmniNode Bridge - Model Metrics API",
            description="AI Lab model performance tracking and intelligent routing",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        # Initialize comprehensive rate limiting service
        from omninode_bridge.services.rate_limiting_service import RateLimitingService

        self.rate_limiting_service = RateLimitingService(
            enable_metrics=True, enable_adaptive_limits=False
        )

        # Add comprehensive rate limiting middleware
        self.app.middleware("http")(
            self.rate_limiting_service.create_fastapi_middleware()
        )

        # Add CORS middleware with environment-based configuration
        # Use environment variables directly to avoid SecureConfig complexity during initialization
        cors_origins = (
            os.getenv("CORS_ORIGINS", "*").split(",")
            if os.getenv("CORS_ORIGINS")
            else ["*"]
        )

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=False,  # Set to False for security unless explicitly needed
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=[
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "X-API-Key",
            ],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        # Authentication setup
        security = HTTPBearer(auto_error=False)
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise ValueError(
                "API_KEY environment variable is required for authentication"
            )

        async def verify_api_key(
            request: Request,
            authorization: HTTPAuthorizationCredentials | None = Depends(security),
            x_api_key: str | None = Header(None),
        ) -> bool:
            """Verify API key from Authorization header or X-API-Key header."""
            provided_key = None
            auth_method = None

            if authorization and authorization.scheme.lower() == "bearer":
                provided_key = authorization.credentials
                auth_method = "bearer_token"
            elif x_api_key:
                provided_key = x_api_key
                auth_method = "api_key_header"

            if not provided_key:
                # Log missing API key
                audit_logger.log_authentication_failure(
                    reason="Missing API key",
                    auth_method="api_key",
                    request=request,
                )
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
                )
            elif provided_key != api_key:
                # Log invalid API key
                audit_logger.log_authentication_failure(
                    reason="Invalid API key",
                    auth_method=auth_method,
                    request=request,
                )
                raise HTTPException(
                    status_code=401,
                    detail="Invalid or missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
                )

            # Log successful authentication
            audit_logger.log_authentication_success(
                auth_method=auth_method,
                request=request,
            )
            return True

        @self.app.post("/execute", response_model=TaskResponse)
        async def execute_task(
            request: TaskRequest,
            req: Request,
            _: bool = Depends(verify_api_key),
        ) -> TaskResponse:
            """Execute a task with intelligent model selection and metrics tracking."""

            # Log task execution for audit
            audit_logger.log_event(
                event_type=AuditEventType.WORKFLOW_EXECUTION_START,
                severity=AuditSeverity.LOW,
                request=req,
                additional_data={
                    "task_type": request.task_type.value,
                    "complexity": request.complexity,
                    "context_size": request.context_size,
                    "preferred_model": request.preferred_model,
                    "prompt_length": len(request.prompt),
                    "track_metrics": request.track_metrics,
                },
                message=f"AI task execution started: {request.task_type.value}",
            )

            try:
                async with smart_responder_client:
                    (
                        response_data,
                        execution_metrics,
                    ) = await self._execute_task_with_circuit_breaker(
                        request.task_type,
                        request.prompt,
                        request.context_size,
                        request.complexity,
                        request.max_latency_ms,
                        request.preferred_model,
                    )

                    # Get recommendations for this task type
                    recommendations = global_metrics.get_model_recommendation(
                        task_type=request.task_type,
                        context_size=request.context_size
                        or len(request.prompt.split()) * 1.3,
                        complexity=request.complexity,
                    )

                    return TaskResponse(
                        success=response_data.get("success", False),
                        response=response_data.get("response", ""),
                        model_used=response_data.get("model", "unknown"),
                        node_location=response_data.get("node", "unknown"),
                        execution_metrics={
                            "execution_id": str(execution_metrics.execution_id),
                            "latency_ms": execution_metrics.latency_ms,
                            "tokens_per_second": execution_metrics.tokens_per_second,
                            "quality_score": execution_metrics.quality_score,
                            "success": execution_metrics.success,
                            "context_size": execution_metrics.context_size,
                            "model_tier": execution_metrics.model_tier.value,
                            "retry_count": execution_metrics.retry_count,
                        },
                        recommendations=[
                            {
                                "model": rec.model_endpoint,
                                "tier": rec.model_tier.value,
                                "confidence": rec.confidence_score,
                                "expected_latency_ms": rec.expected_latency_ms,
                                "expected_success_rate": rec.expected_success_rate,
                                "reason": rec.reason,
                            }
                            for rec in recommendations[:3]
                        ],
                        escalated=execution_metrics.escalated_to_tier is not None,
                    )

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Task execution failed: {e!s}",
                )

        @self.app.post("/compare", response_model=ModelComparisonResult)
        async def compare_models(
            request: ModelComparisonRequest,
            req: Request,
            _: bool = Depends(verify_api_key),
        ) -> ModelComparisonResult:
            """Compare performance of different models on the same task."""

            models_to_test = request.models_to_compare
            if not models_to_test:
                # Select representative models from each tier
                models_to_test = []
                for tier in [
                    ModelTier.TINY,
                    ModelTier.SMALL,
                    ModelTier.MEDIUM,
                    ModelTier.LARGE,
                ]:
                    tier_models = [
                        m for m in global_metrics.available_models if m.tier == tier
                    ]
                    if tier_models:
                        models_to_test.append(tier_models[0].model_id)

            comparison_results = []

            try:
                async with smart_responder_client:
                    # Test each model
                    for model_id in models_to_test:
                        try:
                            (
                                response_data,
                                execution_metrics,
                            ) = await smart_responder_client.execute_task_with_metrics(
                                task_type=request.task_type,
                                prompt=request.prompt,
                                context_size=request.context_size,
                                complexity=request.complexity,
                                preferred_model=model_id,
                            )

                            comparison_results.append(
                                {
                                    "model_id": model_id,
                                    "model_tier": execution_metrics.model_tier.value,
                                    "success": execution_metrics.success,
                                    "latency_ms": execution_metrics.latency_ms,
                                    "tokens_per_second": execution_metrics.tokens_per_second,
                                    "quality_score": execution_metrics.quality_score,
                                    "response_length": len(
                                        response_data.get("response", ""),
                                    ),
                                    "error": (
                                        execution_metrics.error_message
                                        if not execution_metrics.success
                                        else None
                                    ),
                                },
                            )

                        except Exception as e:
                            comparison_results.append(
                                {
                                    "model_id": model_id,
                                    "model_tier": "unknown",
                                    "success": False,
                                    "error": str(e),
                                },
                            )

                # Analyze results
                successful_results = [r for r in comparison_results if r["success"]]

                if successful_results:
                    best_model = min(
                        successful_results,
                        key=lambda x: (1 - x.get("quality_score", 0))
                        + (x.get("latency_ms", 999999) / 10000),
                    )
                    worst_model = max(
                        successful_results,
                        key=lambda x: (1 - x.get("quality_score", 0))
                        + (x.get("latency_ms", 0) / 10000),
                    )
                else:
                    best_model = {
                        "model_id": "none",
                        "error": "No successful executions",
                    }
                    worst_model = {
                        "model_id": "none",
                        "error": "No successful executions",
                    }

                # Generate analysis
                analysis = self._generate_comparison_analysis(
                    comparison_results,
                    request.task_type,
                )

                return ModelComparisonResult(
                    task_type=request.task_type,
                    prompt_summary=(
                        request.prompt[:100] + "..."
                        if len(request.prompt) > 100
                        else request.prompt
                    ),
                    comparison_results=comparison_results,
                    best_model=best_model,
                    worst_model=worst_model,
                    analysis=analysis,
                )

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Model comparison failed: {e!s}",
                )

        @self.app.get("/metrics/performance")
        async def get_performance_metrics(
            task_type: TaskType | None = None,
            model_tier: ModelTier | None = None,
            hours: int = Query(24, description="Hours of data to include"),
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """Get performance metrics with optional filtering."""

            # Filter executions by time window
            cutoff_time = datetime.now(UTC) - timedelta(hours=hours)
            recent_executions = [
                e
                for e in global_metrics.execution_history
                if e.started_at >= cutoff_time
            ]

            # Apply filters
            if task_type:
                recent_executions = [
                    e for e in recent_executions if e.task_type == task_type
                ]
            if model_tier:
                recent_executions = [
                    e for e in recent_executions if e.model_tier == model_tier
                ]

            if not recent_executions:
                return {
                    "message": "No executions found for the specified criteria",
                    "total_executions": 0,
                    "filters": {
                        "task_type": task_type.value if task_type else None,
                        "model_tier": model_tier.value if model_tier else None,
                        "hours": hours,
                    },
                }

            # Calculate metrics
            total_executions = len(recent_executions)
            successful_executions = len([e for e in recent_executions if e.success])
            success_rate = successful_executions / total_executions

            avg_latency = sum(
                e.latency_ms for e in recent_executions if e.latency_ms
            ) / len([e for e in recent_executions if e.latency_ms])
            avg_quality = sum(
                e.quality_score for e in recent_executions if e.quality_score
            ) / len([e for e in recent_executions if e.quality_score])

            # Group by model
            model_performance = {}
            for execution in recent_executions:
                if execution.model_endpoint not in model_performance:
                    model_performance[execution.model_endpoint] = {
                        "executions": 0,
                        "successes": 0,
                        "total_latency": 0,
                        "total_quality": 0,
                    }

                perf = model_performance[execution.model_endpoint]
                perf["executions"] += 1
                if execution.success:
                    perf["successes"] += 1
                if execution.latency_ms:
                    perf["total_latency"] += execution.latency_ms
                if execution.quality_score:
                    perf["total_quality"] += execution.quality_score

            # Format model performance
            formatted_performance = {}
            for model_id, perf in model_performance.items():
                if perf["executions"] > 0:
                    formatted_performance[model_id] = {
                        "executions": perf["executions"],
                        "success_rate": perf["successes"] / perf["executions"],
                        "avg_latency_ms": (
                            perf["total_latency"] / perf["executions"]
                            if perf["total_latency"] > 0
                            else 0
                        ),
                        "avg_quality_score": (
                            perf["total_quality"] / perf["executions"]
                            if perf["total_quality"] > 0
                            else 0
                        ),
                    }

            return {
                "summary": {
                    "total_executions": total_executions,
                    "success_rate": success_rate,
                    "avg_latency_ms": avg_latency,
                    "avg_quality_score": avg_quality,
                },
                "model_performance": formatted_performance,
                "filters": {
                    "task_type": task_type.value if task_type else None,
                    "model_tier": model_tier.value if model_tier else None,
                    "hours": hours,
                },
                "time_range": {
                    "from": cutoff_time.isoformat(),
                    "to": datetime.now(UTC).isoformat(),
                },
            }

        @self.app.get("/metrics/recommendations")
        async def get_model_recommendations(
            task_type: TaskType,
            context_size: int = 1000,
            complexity: str = "moderate",
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """Get model recommendations for a specific task configuration."""

            recommendations = global_metrics.get_model_recommendation(
                task_type=task_type,
                context_size=context_size,
                complexity=complexity,
            )

            return {
                "task_configuration": {
                    "task_type": task_type.value,
                    "context_size": context_size,
                    "complexity": complexity,
                },
                "recommendations": [
                    {
                        "rank": i + 1,
                        "model_endpoint": rec.model_endpoint,
                        "model_tier": rec.model_tier.value,
                        "confidence_score": rec.confidence_score,
                        "expected_latency_ms": rec.expected_latency_ms,
                        "expected_success_rate": rec.expected_success_rate,
                        "expected_quality_score": rec.expected_quality_score,
                        "reason": rec.reason,
                    }
                    for i, rec in enumerate(recommendations)
                ],
                "available_models": len(global_metrics.available_models),
                "total_historical_executions": len(global_metrics.execution_history),
            }

        @self.app.get("/lab/health")
        async def check_lab_health(_: bool = Depends(verify_api_key)) -> dict[str, Any]:
            """Check health status of AI lab nodes."""
            async with smart_responder_client:
                return await self._health_check_with_circuit_breaker()

        @self.app.get("/lab/performance-report")
        async def get_lab_performance_report(
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """Get comprehensive performance report for the AI lab."""
            async with smart_responder_client:
                return await self._get_performance_report_with_circuit_breaker()

        @self.app.get("/models/available")
        async def list_available_models(
            _: bool = Depends(verify_api_key),
        ) -> dict[str, Any]:
            """List all available models in the AI lab."""
            return {
                "total_models": len(global_metrics.available_models),
                "models": [
                    {
                        "model_id": model.model_id,
                        "model_name": model.model_name,
                        "tier": model.tier.value,
                        "node_location": model.node_location,
                        "max_context_tokens": model.max_context_tokens,
                        "specialized_for": [
                            task.value for task in model.specialized_for
                        ],
                        "endpoint_url": model.endpoint_url,
                    }
                    for model in global_metrics.available_models
                ],
                "nodes": {
                    node: [
                        m.model_name
                        for m in global_metrics.available_models
                        if m.node_location == node
                    ]
                    for node in {
                        m.node_location for m in global_metrics.available_models
                    }
                },
            }

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            """Root endpoint with API information."""
            return {
                "service": "OmniNode Bridge - Model Metrics API",
                "version": "0.1.0",
                "description": "AI Lab model performance tracking and intelligent routing",
                "available_models": len(global_metrics.available_models),
                "total_executions": len(global_metrics.execution_history),
                "docs": "/docs",
            }

    def _generate_comparison_analysis(
        self,
        results: list[dict[str, Any]],
        task_type: TaskType,
    ) -> str:
        """Generate analysis text for model comparison results."""

        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        analysis_parts = []

        # Overall success analysis
        if successful_results:
            analysis_parts.append(
                f"Successfully executed on {len(successful_results)}/{len(results)} models.",
            )

            # Performance analysis
            latencies = [
                r["latency_ms"] for r in successful_results if r.get("latency_ms")
            ]
            if latencies:
                fastest = min(latencies)
                slowest = max(latencies)
                analysis_parts.append(
                    f"Latency range: {fastest:.0f}ms to {slowest:.0f}ms.",
                )

            # Quality analysis
            qualities = [
                r["quality_score"] for r in successful_results if r.get("quality_score")
            ]
            if qualities:
                best_quality = max(qualities)
                worst_quality = min(qualities)
                analysis_parts.append(
                    f"Quality scores: {worst_quality:.2f} to {best_quality:.2f}.",
                )

            # Tier analysis
            tier_performance = {}
            for result in successful_results:
                tier = result.get("model_tier", "unknown")
                if tier not in tier_performance:
                    tier_performance[tier] = {
                        "count": 0,
                        "avg_latency": 0,
                        "avg_quality": 0,
                    }

                perf = tier_performance[tier]
                perf["count"] += 1
                if result.get("latency_ms"):
                    perf["avg_latency"] += result["latency_ms"]
                if result.get("quality_score"):
                    perf["avg_quality"] += result["quality_score"]

            # Tier recommendations
            if len(tier_performance) > 1:
                for tier, perf in tier_performance.items():
                    if perf["count"] > 0:
                        avg_lat = perf["avg_latency"] / perf["count"]
                        avg_qual = (
                            perf["avg_quality"] / perf["count"]
                            if perf["avg_quality"] > 0
                            else 0
                        )

                        if tier in ["tiny", "small"] and avg_qual > 0.7:
                            analysis_parts.append(
                                f"{tier.title()} models performed surprisingly well.",
                            )
                        elif tier in ["large", "xlarge", "huge"] and avg_qual < 0.6:
                            analysis_parts.append(
                                f"{tier.title()} models underperformed expectations.",
                            )

        else:
            analysis_parts.append(
                "All models failed to complete the task successfully.",
            )

        # Failure analysis
        if failed_results:
            error_types = {}
            for result in failed_results:
                error = result.get("error", "unknown error")
                error_type = error.split(":")[0] if ":" in error else error
                error_types[error_type] = error_types.get(error_type, 0) + 1

            analysis_parts.append(
                f"Common failure modes: {', '.join(error_types.keys())}",
            )

        # Task-specific insights
        if task_type == TaskType.CODE_GENERATION:
            analysis_parts.append(
                "For code generation, consider response length and syntax correctness.",
            )
        elif task_type == TaskType.DEBUGGING:
            analysis_parts.append(
                "Debugging tasks benefit from models with strong analytical capabilities.",
            )
        elif task_type == TaskType.ARCHITECTURE:
            analysis_parts.append(
                "Architecture discussions typically require larger models for comprehensive analysis.",
            )

        return " ".join(analysis_parts)

    @circuit(failure_threshold=5, recovery_timeout=30, expected_exception=Exception)
    async def _execute_task_with_circuit_breaker(
        self,
        task_type: TaskType,
        prompt: str,
        context_size: int | None,
        complexity: str,
        max_latency_ms: float | None,
        preferred_model: str | None,
    ):
        """Execute task with AI lab service with circuit breaker protection."""
        return await smart_responder_client.execute_task_with_metrics(
            task_type=task_type,
            prompt=prompt,
            context_size=context_size,
            complexity=complexity,
            max_latency_ms=max_latency_ms,
            preferred_model=preferred_model,
        )

    @circuit(failure_threshold=3, recovery_timeout=20, expected_exception=Exception)
    async def _health_check_with_circuit_breaker(self):
        """Health check AI lab nodes with circuit breaker protection."""
        return await smart_responder_client.health_check_lab_nodes()

    @circuit(failure_threshold=3, recovery_timeout=20, expected_exception=Exception)
    async def _get_performance_report_with_circuit_breaker(self):
        """Get performance report with circuit breaker protection."""
        return await smart_responder_client.get_performance_report()


def create_metrics_app() -> FastAPI:
    """Factory function to create the model metrics API application."""
    metrics_api = ModelMetricsAPI()
    return metrics_api.app
