"""Smart Responder Chain integration with enhanced model metrics tracking."""

import time
from datetime import UTC, datetime
from typing import Any

import aiohttp

from ..models.model_metrics import (
    ModelEndpoint,
    ModelTier,
    TaskExecution,
    TaskType,
    global_metrics,
)


class SmartResponderClient:
    """Enhanced Smart Responder Chain client with comprehensive metrics tracking."""

    def __init__(self, base_config: dict[str, Any] | None = None):
        """
        Initialize Smart Responder client with AI lab configuration.

        Args:
            base_config: Optional base configuration, will auto-detect lab setup if None
        """
        self.config = base_config or self._load_ai_lab_config()
        self.metrics = global_metrics
        self.session: aiohttp.ClientSession | None = None

        # Initialize with discovered models from AI lab
        self._initialize_lab_models()

    def _load_ai_lab_config(self) -> dict[str, Any]:
        """Load AI lab configuration based on discovered infrastructure."""
        return {
            "nodes": {
                "mac_studio": {
                    "host": "192.168.86.200",
                    "ollama_port": 11434,
                    "capabilities": ["langgraph", "heavy_compute", "large_models"],
                    "memory_gb": 192,
                    "primary_models": ["llama3.1:70b", "mixtral:8x22b", "qwen2.5:14b"],
                },
                "mac_mini": {
                    "host": "192.168.86.101",
                    "ollama_port": 11434,
                    "capabilities": ["balanced_compute", "medium_models"],
                    "memory_gb": 32,
                    "primary_models": ["llama3.2:11b", "qwen2.5:7b", "codestral:22b"],
                },
                "ai_pc": {
                    "host": "192.168.86.201",
                    "ollama_port": 11434,
                    "capabilities": ["gpu_acceleration", "inference_optimization"],
                    "gpu": "RTX 5090",
                    "primary_models": [
                        "llama3.2:8b",
                        "phi4:14b",
                        "deepseek-coder:6.7b",
                    ],
                },
                "macbook_air": {
                    "host": "192.168.86.105",
                    "ollama_port": 11434,
                    "capabilities": ["lightweight_compute", "small_models"],
                    "memory_gb": 16,
                    "primary_models": ["gpt-oss:20b", "mistral:latest", "phi3:latest"],
                },
            },
            "tiers": {
                "tiny": {
                    "max_params": "3B",
                    "target_latency_ms": 500,
                    "use_for": ["simple_tasks", "quick_responses"],
                },
                "small": {
                    "max_params": "7B",
                    "target_latency_ms": 1000,
                    "use_for": ["moderate_tasks", "code_snippets"],
                },
                "medium": {
                    "max_params": "8B",
                    "target_latency_ms": 1500,
                    "use_for": ["balanced_tasks", "documentation"],
                },
                "large": {
                    "max_params": "14B",
                    "target_latency_ms": 3000,
                    "use_for": ["complex_reasoning", "architecture"],
                },
                "xlarge": {
                    "max_params": "22B",
                    "target_latency_ms": 5000,
                    "use_for": ["advanced_tasks", "research"],
                },
                "huge": {
                    "max_params": "70B+",
                    "target_latency_ms": 10000,
                    "use_for": ["critical_tasks", "comprehensive_analysis"],
                },
            },
            "task_routing": {
                TaskType.CODE_GENERATION: ["medium", "large", "xlarge"],
                TaskType.CODE_REVIEW: ["large", "xlarge", "huge"],
                TaskType.DEBUGGING: ["medium", "large", "xlarge"],
                TaskType.DOCUMENTATION: ["small", "medium", "large"],
                TaskType.API_DESIGN: ["large", "xlarge", "huge"],
                TaskType.TESTING: ["medium", "large"],
                TaskType.ARCHITECTURE: ["xlarge", "huge"],
                TaskType.SECURITY_AUDIT: ["large", "xlarge", "huge"],
                TaskType.PERFORMANCE_OPTIMIZATION: ["large", "xlarge"],
                TaskType.WEBHOOK_PROCESSING: ["small", "medium"],
                TaskType.EVENT_TRANSFORMATION: ["small", "medium"],
                TaskType.GENERAL_REASONING: ["medium", "large"],
            },
        }

    def _initialize_lab_models(self) -> None:
        """Initialize available models from AI lab nodes."""
        models = []

        # Define model capabilities based on actual AI lab setup (verified models)
        model_definitions = [
            # Mac Studio (Heavy compute, large models) - verified available
            (
                "gpt-oss:120b",
                "mac_studio",
                ModelTier.HUGE,
                ["api_design", "architecture", "security_audit", "complex_reasoning"],
                128000,
            ),
            (
                "mixtral:8x22b-instruct-v0.1-q4_K_M",
                "mac_studio",
                ModelTier.HUGE,
                ["code_review", "architecture", "complex_reasoning"],
                65536,
            ),
            (
                "yi:34b-chat-q4_K_M",
                "mac_studio",
                ModelTier.XLARGE,
                ["code_generation", "debugging", "analysis"],
                32768,
            ),
            (
                "codestral:22b-v0.1-q4_K_M",
                "mac_studio",
                ModelTier.XLARGE,
                ["code_generation", "code_review"],
                32768,
            ),
            (
                "llama3.2:3b",
                "mac_studio",
                ModelTier.TINY,
                ["simple_tasks", "webhook_processing"],
                8192,
            ),
            # Mac Mini (Balanced compute) - common models
            (
                "llama3.1:70b-instruct-q4_K_M",
                "mac_mini",
                ModelTier.HUGE,
                ["complex_reasoning", "architecture"],
                128000,
            ),
            (
                "qwen2.5:32b-instruct-q4_K_M",
                "mac_mini",
                ModelTier.XLARGE,
                ["code_generation", "documentation"],
                32768,
            ),
            (
                "llama3.2:3b",
                "mac_mini",
                ModelTier.TINY,
                ["general_reasoning", "testing"],
                8192,
            ),
            # AI PC (GPU acceleration) - verified available
            (
                "llama3.1:8b-instruct-q6_k",
                "ai_pc",
                ModelTier.MEDIUM,
                ["code_generation", "debugging"],
                8192,
            ),
            (
                "phi3.5:latest",
                "ai_pc",
                ModelTier.SMALL,
                ["reasoning", "analysis"],
                16384,
            ),
            (
                "qwen2.5:7b-instruct",
                "ai_pc",
                ModelTier.SMALL,
                ["code_generation", "webhooks"],
                16384,
            ),
            (
                "deepseek-coder:6.7b-instruct",
                "ai_pc",
                ModelTier.SMALL,
                ["code_generation", "debugging"],
                16384,
            ),
            # MacBook Air (Lightweight) - verified available
            (
                "gpt-oss:20b",
                "macbook_air",
                ModelTier.LARGE,
                ["general_reasoning", "complex_tasks"],
                32768,
            ),
            (
                "mistral:latest",
                "macbook_air",
                ModelTier.SMALL,
                ["event_transformation", "basic_docs"],
                8192,
            ),
            (
                "phi3:latest",
                "macbook_air",
                ModelTier.TINY,
                ["webhook_processing", "simple_tasks"],
                4096,
            ),
        ]

        for model_name, node, tier, specializations, max_context in model_definitions:
            node_config = self.config["nodes"][node]

            # Convert specializations to TaskType enums
            task_types = []
            for spec in specializations:
                for task_type in TaskType:
                    if (
                        spec.lower() in task_type.value.lower()
                        or task_type.value.lower() in spec.lower()
                    ):
                        task_types.append(task_type)
                        break

            endpoint = ModelEndpoint(
                model_id=f"{node}_{model_name.replace(':', '_')}",
                endpoint_url=f"http://{node_config['host']}:{node_config['ollama_port']}",
                tier=tier,
                node_location=node,
                model_name=model_name,
                parameter_count=(
                    model_name.split(":")[1] if ":" in model_name else "unknown"
                ),
                specialized_for=task_types,
                max_context_tokens=max_context,
            )
            models.append(endpoint)

        self.metrics.available_models = models

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120, connect=15),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def execute_task_with_metrics(
        self,
        task_type: TaskType,
        prompt: str,
        context_size: int | None = None,
        complexity: str = "moderate",
        max_latency_ms: float | None = None,
        preferred_model: str | None = None,
    ) -> tuple[dict[str, Any], TaskExecution]:
        """
        Execute a task with comprehensive metrics tracking.

        Args:
            task_type: Type of task being executed
            prompt: Task prompt/description
            context_size: Size of context in tokens (estimated if None)
            complexity: Task complexity level
            max_latency_ms: Maximum acceptable latency
            preferred_model: Specific model to use (overrides recommendation)

        Returns:
            Tuple of (response_data, execution_metrics)
        """
        # Estimate context size if not provided
        if context_size is None:
            context_size = len(prompt.split()) * 1.3  # Rough token estimation

        # Get model recommendation
        if preferred_model:
            model_endpoint = next(
                (
                    m
                    for m in self.metrics.available_models
                    if m.model_id == preferred_model
                ),
                self.metrics.available_models[0],  # Fallback to first model
            )
        else:
            recommendations = self.metrics.get_model_recommendation(
                task_type=task_type,
                context_size=int(context_size),
                complexity=complexity,
                max_latency_ms=max_latency_ms,
            )

            if not recommendations:
                # Fallback to tier-based selection
                preferred_tiers = self.config["task_routing"].get(task_type, ["medium"])
                model_endpoint = self._select_model_by_tier(preferred_tiers[0])
            else:
                model_endpoint = next(
                    (
                        m
                        for m in self.metrics.available_models
                        if m.model_id == recommendations[0].model_endpoint
                    ),
                    self.metrics.available_models[0],
                )

        # Create execution record
        execution = TaskExecution(
            task_type=task_type,
            model_endpoint=model_endpoint.model_id,
            model_tier=model_endpoint.tier,
            input_tokens=int(context_size),
            output_tokens=0,  # Will be updated after response
            context_size=int(context_size),
            task_complexity=complexity,
        )

        try:
            # Execute the task
            start_time = time.time()
            response_data = await self._execute_model_request(
                model_endpoint,
                prompt,
                execution,
            )
            end_time = time.time()

            # Update execution metrics
            execution.completed_at = datetime.now(UTC)
            execution.latency_ms = (end_time - start_time) * 1000
            execution.success = response_data.get("success", False)

            # Extract response metrics if available
            if "usage" in response_data:
                usage = response_data["usage"]
                execution.output_tokens = usage.get("completion_tokens", 0)
                execution.tokens_per_second = (
                    execution.output_tokens / (execution.latency_ms / 1000)
                    if execution.latency_ms > 0
                    else 0
                )

            # Quality assessment (could be enhanced with actual quality metrics)
            execution.quality_score = self._assess_response_quality(
                response_data,
                task_type,
            )

            # Add to metrics
            self.metrics.add_execution(execution)

            return response_data, execution

        except Exception as e:
            # Handle execution failure
            execution.completed_at = datetime.now(UTC)
            execution.success = False
            execution.error_type = type(e).__name__
            execution.error_message = str(e)
            execution.latency_ms = (time.time() - start_time) * 1000

            # Try escalation if this wasn't already an escalated request
            if not execution.escalated_to_tier and self._should_escalate(
                model_endpoint.tier,
            ):
                return await self._escalate_task(
                    execution,
                    prompt,
                    task_type,
                    complexity,
                )

            self.metrics.add_execution(execution)
            raise

    def _select_model_by_tier(self, tier_name: str) -> ModelEndpoint:
        """Select best available model for a given tier."""
        tier = ModelTier(tier_name.lower())

        # Find models in the specified tier
        tier_models = [m for m in self.metrics.available_models if m.tier == tier]

        if not tier_models:
            # Fallback to any available model
            return self.metrics.available_models[0]

        # Select based on recent performance or just the first one
        return tier_models[0]

    async def _execute_model_request(
        self,
        model_endpoint: ModelEndpoint,
        prompt: str,
        execution: TaskExecution,
    ) -> dict[str, Any]:
        """Execute request to specific model endpoint."""

        if not self.session:
            raise RuntimeError("Client session not initialized")

        # Build request payload for Ollama API
        payload = {
            "model": model_endpoint.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "num_predict": 2048,  # Max output tokens
            },
        }

        url = f"{model_endpoint.endpoint_url}/api/generate"

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "model": model_endpoint.model_name,
                        "node": model_endpoint.node_location,
                        "usage": {
                            "prompt_tokens": result.get("prompt_eval_count", 0),
                            "completion_tokens": result.get("eval_count", 0),
                            "total_tokens": result.get("prompt_eval_count", 0)
                            + result.get("eval_count", 0),
                        },
                        "timing": {
                            "prompt_eval_duration": result.get(
                                "prompt_eval_duration",
                                0,
                            ),
                            "eval_duration": result.get("eval_duration", 0),
                            "total_duration": result.get("total_duration", 0),
                        },
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}",
                        "model": model_endpoint.model_name,
                    }

        except TimeoutError:
            return {
                "success": False,
                "error": "Request timeout",
                "model": model_endpoint.model_name,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model_endpoint.model_name,
            }

    def _assess_response_quality(
        self,
        response_data: dict[str, Any],
        task_type: TaskType,
    ) -> float:
        """Assess quality of model response (0.0-1.0)."""
        if not response_data.get("success", False):
            return 0.0

        response_text = response_data.get("response", "")

        # Basic quality heuristics
        base_score = 0.5  # Start with neutral score

        # Length appropriateness
        if len(response_text) > 50:
            base_score += 0.1
        if len(response_text) > 200:
            base_score += 0.1

        # Task-specific quality indicators
        if task_type == TaskType.CODE_GENERATION:
            if "```" in response_text:  # Code blocks present
                base_score += 0.2
            if any(
                keyword in response_text.lower()
                for keyword in ["def ", "class ", "function", "import"]
            ):
                base_score += 0.1

        elif task_type == TaskType.DEBUGGING:
            if any(
                keyword in response_text.lower()
                for keyword in ["error", "issue", "problem", "fix", "solution"]
            ):
                base_score += 0.2

        elif task_type == TaskType.DOCUMENTATION:
            if any(
                keyword in response_text.lower()
                for keyword in ["##", "###", "example", "usage"]
            ):
                base_score += 0.2

        # Coherence indicators
        if not any(
            indicator in response_text.lower()
            for indicator in ["error", "cannot", "unable", "sorry"]
        ):
            base_score += 0.1

        return min(1.0, base_score)

    def _should_escalate(self, current_tier: ModelTier) -> bool:
        """Determine if task should be escalated to higher tier."""
        tier_order = [
            ModelTier.TINY,
            ModelTier.SMALL,
            ModelTier.MEDIUM,
            ModelTier.LARGE,
            ModelTier.XLARGE,
            ModelTier.HUGE,
        ]

        current_index = tier_order.index(current_tier)
        return (
            current_index < len(tier_order) - 1
        )  # Can escalate if not at highest tier

    async def _escalate_task(
        self,
        failed_execution: TaskExecution,
        prompt: str,
        task_type: TaskType,
        complexity: str,
    ) -> tuple[dict[str, Any], TaskExecution]:
        """Escalate failed task to higher tier model."""

        tier_order = [
            ModelTier.TINY,
            ModelTier.SMALL,
            ModelTier.MEDIUM,
            ModelTier.LARGE,
            ModelTier.XLARGE,
            ModelTier.HUGE,
        ]

        current_index = tier_order.index(failed_execution.model_tier)
        next_tier = tier_order[current_index + 1]

        # Select model from next tier
        next_model = self._select_model_by_tier(next_tier.value)

        # Update failed execution with escalation info
        failed_execution.escalated_to_tier = next_tier
        self.metrics.add_execution(failed_execution)

        # Execute with higher tier model
        return await self.execute_task_with_metrics(
            task_type=task_type,
            prompt=prompt,
            context_size=failed_execution.context_size,
            complexity=complexity,
            preferred_model=next_model.model_id,
        )

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_executions": len(self.metrics.execution_history),
            "available_models": len(self.metrics.available_models),
            "lab_nodes": list(self.config["nodes"].keys()),
            "performance_by_tier": {},
            "performance_by_task_type": {},
            "recent_escalations": len(self.metrics.recent_escalation_patterns),
            "top_performing_models": [],
            "recommendations": [],
        }

        # Analyze performance by tier
        for tier in ModelTier:
            tier_executions = [
                e for e in self.metrics.execution_history if e.model_tier == tier
            ]
            if tier_executions:
                report["performance_by_tier"][tier.value] = {
                    "total_executions": len(tier_executions),
                    "success_rate": sum(1 for e in tier_executions if e.success)
                    / len(tier_executions),
                    "avg_latency_ms": sum(e.latency_ms or 0 for e in tier_executions)
                    / len(tier_executions),
                    "avg_quality_score": sum(
                        e.quality_score or 0 for e in tier_executions
                    )
                    / len(tier_executions),
                }

        # Analyze performance by task type
        for task_type in TaskType:
            task_executions = [
                e for e in self.metrics.execution_history if e.task_type == task_type
            ]
            if task_executions:
                report["performance_by_task_type"][task_type.value] = {
                    "total_executions": len(task_executions),
                    "success_rate": sum(1 for e in task_executions if e.success)
                    / len(task_executions),
                    "avg_latency_ms": sum(e.latency_ms or 0 for e in task_executions)
                    / len(task_executions),
                    "most_used_tier": max(
                        {e.model_tier.value for e in task_executions},
                        key=[e.model_tier.value for e in task_executions].count,
                    ),
                }

        # Find top performing models
        model_performance = {}
        for execution in self.metrics.execution_history:
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

        # Calculate model rankings
        for model_id, perf in model_performance.items():
            if perf["executions"] >= 5:  # Only consider models with sufficient data
                success_rate = perf["successes"] / perf["executions"]
                avg_latency = perf["total_latency"] / perf["executions"]
                avg_quality = perf["total_quality"] / perf["executions"]

                composite_score = (
                    (success_rate * 0.5)
                    + ((1 - avg_latency / 10000) * 0.3)
                    + (avg_quality * 0.2)
                )

                report["top_performing_models"].append(
                    {
                        "model_id": model_id,
                        "success_rate": success_rate,
                        "avg_latency_ms": avg_latency,
                        "avg_quality_score": avg_quality,
                        "composite_score": composite_score,
                        "total_executions": perf["executions"],
                    },
                )

        # Sort by composite score
        report["top_performing_models"].sort(
            key=lambda x: x["composite_score"],
            reverse=True,
        )
        report["top_performing_models"] = report["top_performing_models"][:10]

        return report

    async def health_check_lab_nodes(self) -> dict[str, Any]:
        """Check health status of all AI lab nodes."""
        health_status = {
            "timestamp": datetime.now(UTC).isoformat(),
            "nodes": {},
            "overall_status": "unknown",
        }

        healthy_nodes = 0
        total_nodes = len(self.config["nodes"])

        for node_name, node_config in self.config["nodes"].items():
            try:
                url = f"http://{node_config['host']}:{node_config['ollama_port']}/api/tags"

                if not self.session:
                    self.session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=15),
                    )

                async with self.session.get(url) as response:
                    if response.status == 200:
                        models_data = await response.json()
                        health_status["nodes"][node_name] = {
                            "status": "healthy",
                            "available_models": len(models_data.get("models", [])),
                            "host": node_config["host"],
                        }
                        healthy_nodes += 1
                    else:
                        health_status["nodes"][node_name] = {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}",
                            "host": node_config["host"],
                        }

            except Exception as e:
                health_status["nodes"][node_name] = {
                    "status": "unreachable",
                    "error": str(e),
                    "host": node_config["host"],
                }

        # Determine overall status
        if healthy_nodes == total_nodes:
            health_status["overall_status"] = "healthy"
        elif healthy_nodes > 0:
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "unhealthy"

        health_status["healthy_nodes"] = healthy_nodes
        health_status["total_nodes"] = total_nodes

        return health_status


# Global client instance
smart_responder_client = SmartResponderClient()
