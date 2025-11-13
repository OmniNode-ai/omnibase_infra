#!/usr/bin/env python3
"""
Business Logic Generator for ONEX nodes.

Orchestrates LLM-based generation of business logic implementations to replace
template stubs with intelligent, context-aware code.

Pipeline:
1. Extract method stubs from generated templates
2. Build context-rich prompts for each method
3. Call NodeLLMEffect to generate implementation
4. Validate generated code (AST, ONEX compliance, security)
5. Inject validated code back into template
6. Collect metrics for learning

ONEX v2.0 Compliance:
- Async/await throughout
- ModelOnexError for error handling
- Structured logging with emit_log_event
- Comprehensive metrics collection
"""

import ast
import logging
import os
import re
from typing import Any, Optional, cast

from omnibase_core import EnumCoreErrorCode, ModelOnexError
from omnibase_core.models.contracts.model_contract_effect import ModelContractEffect
from omnibase_core.models.contracts.model_io_operation_config import (
    ModelIOOperationConfig,
)
from omnibase_core.models.core import ModelContainer
from omnibase_core.primitives.model_semver import ModelSemVer

from omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts
from omninode_bridge.intelligence.llm_metrics_store import LLMMetricsStore
from omninode_bridge.intelligence.models import LLMGenerationMetric
from omninode_bridge.nodes.llm_effect.v1_0_0.models.enum_llm_tier import EnumLLMTier
from omninode_bridge.nodes.llm_effect.v1_0_0.models.model_request import ModelLLMRequest
from omninode_bridge.nodes.llm_effect.v1_0_0.node import NodeLLMEffect

from .config import BusinessLogicConfig
from .models import (
    ModelBusinessLogicContext,
    ModelEnhancedArtifacts,
    ModelGeneratedMethod,
    ModelMethodStub,
)

logger = logging.getLogger(__name__)


class BusinessLogicGenerator:
    """
    Generate intelligent business logic for ONEX nodes using LLMs.

    Replaces template stubs with context-aware implementations.

    Example:
        >>> import os
        >>> os.environ["ZAI_API_KEY"] = "your_api_key"  # pragma: allowlist secret
        >>> generator = BusinessLogicGenerator(enable_llm=True)
        >>> enhanced = await generator.enhance_artifacts(
        ...     artifacts=artifacts,
        ...     requirements=requirements,
        ...     context_data={"patterns": ["..."]}
        ... )
        >>> print(f"Generated {len(enhanced.methods_generated)} methods")
    """

    def __init__(
        self,
        enable_llm: bool = True,
        llm_tier: EnumLLMTier = EnumLLMTier.CLOUD_FAST,
        metrics_store: Optional[LLMMetricsStore] = None,
    ):
        """
        Initialize business logic generator.

        Args:
            enable_llm: Enable LLM generation (if False, returns original artifacts)
            llm_tier: LLM tier to use (CLOUD_FAST by default)
            metrics_store: Optional metrics store for tracking generation metrics

        Raises:
            ModelOnexError: If ZAI_API_KEY not set when enable_llm=True
        """
        self.enable_llm = enable_llm
        self.llm_tier = llm_tier
        self.metrics_store = metrics_store

        # Initialize NodeLLMEffect if LLM enabled
        self.llm_node: Optional[NodeLLMEffect]
        if self.enable_llm:
            # Credentials are ALWAYS read from environment (security best practice)
            zai_api_key = os.getenv("ZAI_API_KEY")
            if not zai_api_key:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_INPUT,
                    message="ZAI_API_KEY environment variable required when enable_llm=True",
                    details={"enable_llm": enable_llm},
                )

            # Container doesn't contain secrets, only config
            container = ModelContainer(value={}, container_type="config")
            self.llm_node = NodeLLMEffect(container)
        else:
            self.llm_node = None

        logger.info(
            f"BusinessLogicGenerator initialized (enable_llm={enable_llm}, tier={llm_tier.value})"
        )

    async def enhance_artifacts(
        self,
        artifacts: ModelGeneratedArtifacts,
        requirements: ModelPRDRequirements,
        context_data: Optional[dict[str, Any]] = None,
    ) -> ModelEnhancedArtifacts:
        """
        Enhance generated artifacts with LLM business logic.

        Args:
            artifacts: Generated artifacts from TemplateEngine
            requirements: PRD requirements
            context_data: Additional context (patterns, best practices)

        Returns:
            ModelEnhancedArtifacts with enhanced node file and metrics

        Raises:
            ModelOnexError: On generation or validation failures
        """
        if not self.enable_llm:
            logger.info("LLM disabled - returning original artifacts")
            return ModelEnhancedArtifacts(
                original_artifacts=artifacts,
                enhanced_node_file=artifacts.node_file,
            )

        logger.info(
            f"Enhancing artifacts for {artifacts.node_name} with LLM ({self.llm_tier.value})"
        )

        try:
            # Step 1: Extract method stubs from node file
            stubs = self._extract_method_stubs(artifacts.node_file, artifacts.node_type)
            logger.info(f"Found {len(stubs)} method stubs to implement")

            # Step 2: Generate implementations for each stub
            generated_methods = []
            enhanced_node_file = artifacts.node_file

            for stub in stubs:
                if not stub.needs_implementation:
                    logger.debug(f"Skipping {stub.method_name} (already implemented)")
                    continue

                # Build context for this method
                method_context = self._build_method_context(
                    stub=stub,
                    requirements=requirements,
                    node_type=artifacts.node_type,
                    context_data=context_data or {},
                )

                # Generate implementation
                generated_method = await self._generate_method_implementation(
                    context=method_context
                )
                generated_methods.append(generated_method)

                # Inject into node file
                enhanced_node_file = self._inject_implementation(
                    node_file=enhanced_node_file,
                    method_name=stub.method_name,
                    implementation=generated_method.generated_code,
                )

            # Calculate aggregate metrics
            total_tokens = sum(m.tokens_used for m in generated_methods)
            total_cost = sum(m.cost_usd for m in generated_methods)
            total_latency = sum(m.latency_ms for m in generated_methods)
            success_count = sum(1 for m in generated_methods if m.syntax_valid)
            success_rate = (
                success_count / len(generated_methods) if generated_methods else 1.0
            )

            logger.info(
                f"Enhanced {len(generated_methods)} methods "
                f"(tokens={total_tokens}, cost=${total_cost:.4f}, "
                f"latency={total_latency:.1f}ms, success_rate={success_rate:.1%})"
            )

            return ModelEnhancedArtifacts(
                original_artifacts=artifacts,
                enhanced_node_file=enhanced_node_file,
                methods_generated=generated_methods,
                total_tokens_used=total_tokens,
                total_cost_usd=total_cost,
                total_latency_ms=total_latency,
                generation_success_rate=success_rate,
            )

        except Exception as e:
            logger.error(f"Failed to enhance artifacts: {e}")
            if isinstance(e, ModelOnexError):
                raise
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Failed to enhance artifacts: {e}",
                details={"error_type": type(e).__name__},
            )

    def _extract_method_stubs(
        self, node_file: str, node_type: str
    ) -> list[ModelMethodStub]:
        """
        Extract method stubs that need implementation.

        Looks for methods with stub comments like:
        - # IMPLEMENTATION REQUIRED
        - # TODO: Implement
        - pass  # Stub
        - raise NotImplementedError
        """
        stubs = []

        try:
            tree = ast.parse(node_file)

            # Find class definition
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Find methods
                    for item in node.body:
                        if isinstance(item, ast.AsyncFunctionDef):
                            # Check if method needs implementation
                            method_source = ast.get_source_segment(node_file, item)
                            if not method_source:
                                continue

                            # Check for stub indicators
                            needs_impl = any(
                                indicator in method_source
                                for indicator in BusinessLogicConfig.STUB_INDICATORS
                            )

                            if needs_impl:
                                # Extract docstring
                                docstring = ast.get_docstring(item)

                                stubs.append(
                                    ModelMethodStub(
                                        method_name=item.name,
                                        signature=method_source.split("\n")[0],
                                        docstring=docstring,
                                        line_number=item.lineno,
                                        needs_implementation=True,
                                    )
                                )

        except SyntaxError as e:
            logger.error(f"Failed to parse node file: {e}")
            return []

        return stubs

    def _build_method_context(
        self,
        stub: ModelMethodStub,
        requirements: ModelPRDRequirements,
        node_type: str,
        context_data: dict[str, Any],
    ) -> ModelBusinessLogicContext:
        """Build context for method generation."""
        return ModelBusinessLogicContext(
            node_type=node_type,
            service_name=requirements.service_name,
            business_description=requirements.business_description,
            operations=requirements.operations,
            features=requirements.features,
            method_name=stub.method_name,
            method_signature=stub.signature,
            method_docstring=stub.docstring,
            similar_patterns=context_data.get("patterns", []),
            best_practices=context_data.get("best_practices", []),
            performance_requirements=requirements.performance_requirements,
        )

    async def _generate_method_implementation(
        self, context: ModelBusinessLogicContext
    ) -> ModelGeneratedMethod:
        """
        Generate method implementation using LLM.

        Builds structured prompt with context and calls NodeLLMEffect.
        """
        # Build prompt
        prompt = self._build_generation_prompt(context)

        # Build LLM request
        llm_request = ModelLLMRequest(
            prompt=prompt,
            tier=self.llm_tier,
            max_tokens=BusinessLogicConfig.DEFAULT_MAX_TOKENS,
            temperature=BusinessLogicConfig.DEFAULT_TEMPERATURE,
            top_p=BusinessLogicConfig.DEFAULT_TOP_P,
            system_prompt=BusinessLogicConfig.SYSTEM_PROMPT,
            operation_type="method_implementation",
        )

        try:
            # Ensure LLM node is available
            if self.llm_node is None:
                raise ModelOnexError(
                    error_code=EnumCoreErrorCode.INVALID_STATE,
                    message="LLM node not initialized",
                    details={"enable_llm": self.enable_llm},
                )

            # Initialize LLM node if not already done
            if self.llm_node.http_client is None:
                await self.llm_node.initialize()

            # Create contract for LLM call
            # Import EnumNodeType for proper validation
            from omnibase_core.enums.enum_node_type import EnumNodeType

            # Convert string to EnumNodeType enum member
            node_type_enum = EnumNodeType[context.node_type.upper()]

            contract = ModelContractEffect(
                name="llm_generation",
                version=ModelSemVer(major=1, minor=0, patch=0),
                description="Generate Python method implementation",
                node_type=node_type_enum,  # Pass enum member, not string
                input_model="ModelLLMRequest",
                output_model="ModelLLMResponse",
                input_state=llm_request.model_dump(),  # Use input_state, not input_data
                io_operations=[
                    ModelIOOperationConfig(
                        operation_type="llm_request",
                        atomic=True,
                        timeout_seconds=300,  # 5 minutes for LLM generation
                    )
                ],
            )

            # Call LLM
            llm_response = await self.llm_node.execute_effect(contract)

            generated_code = llm_response.generated_text.strip()

            # Strip markdown code fences if present
            if generated_code.startswith("```python"):
                generated_code = generated_code[len("```python") :].strip()
            if generated_code.startswith("```"):
                generated_code = generated_code[3:].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-3].strip()

            # Validate generated code
            validation = self._validate_generated_code(generated_code)

            # Store metrics if store available
            if self.metrics_store and BusinessLogicConfig.ENABLE_METRICS:
                try:
                    metric = LLMGenerationMetric(
                        session_id="codegen",  # TODO: Use actual session ID
                        correlation_id=None,
                        node_type=context.node_type,
                        model_tier=self.llm_tier.value,
                        model_name=llm_response.model_used,
                        prompt_tokens=llm_response.tokens_input,
                        completion_tokens=llm_response.tokens_output,
                        total_tokens=llm_response.tokens_total,
                        latency_ms=llm_response.latency_ms,
                        cost_usd=llm_response.cost_usd,
                        success=validation["syntax_valid"],
                        error_message=None,
                        metadata={
                            "method_name": context.method_name,
                            "onex_compliant": validation["onex_compliant"],
                        },
                    )
                    await self.metrics_store.store_generation_metric(metric)
                except Exception as e:
                    logger.warning(f"Failed to store metrics: {e}")

            return ModelGeneratedMethod(
                method_name=context.method_name,
                generated_code=generated_code,
                syntax_valid=validation["syntax_valid"],
                onex_compliant=validation["onex_compliant"],
                has_type_hints=validation["has_type_hints"],
                has_docstring=validation["has_docstring"],
                security_issues=validation["security_issues"],
                tokens_used=llm_response.tokens_total,
                cost_usd=llm_response.cost_usd,
                latency_ms=llm_response.latency_ms,
                model_used=llm_response.model_used,
            )

        except Exception as e:
            logger.error(f"Failed to generate {context.method_name}: {e}")
            # Return stub as fallback
            return ModelGeneratedMethod(
                method_name=context.method_name,
                generated_code="        pass  # Generation failed",
                syntax_valid=True,
                onex_compliant=False,
                has_type_hints=False,
                has_docstring=False,
                security_issues=[f"Generation failed: {e}"],
                tokens_used=0,
                cost_usd=0.0,
                latency_ms=0.0,
                model_used="none",
            )

    def _build_generation_prompt(self, context: ModelBusinessLogicContext) -> str:
        """Build structured prompt for method generation."""
        prompt_parts = [
            f"# Task: Implement {context.method_name} for {context.service_name}",
            "",
            "## Context",
            f"Node Type: {context.node_type}",
            f"Purpose: {context.business_description}",
            f"Operations: {', '.join(context.operations)}",
            f"Features: {', '.join(context.features)}",
            "",
            "## Method Signature",
            "```python",
            context.method_signature,
            "```",
        ]

        if context.method_docstring:
            prompt_parts.extend(
                [
                    "",
                    "## Docstring",
                    context.method_docstring,
                ]
            )

        if context.similar_patterns:
            prompt_parts.extend(
                [
                    "",
                    "## Similar Patterns (for reference)",
                    *[f"- {pattern}" for pattern in context.similar_patterns[:3]],
                ]
            )

        if context.best_practices:
            prompt_parts.extend(
                [
                    "",
                    "## ONEX Best Practices",
                    *[f"- {practice}" for practice in context.best_practices[:5]],
                ]
            )

        prompt_parts.extend(
            [
                "",
                "## Requirements",
                "1. Return ONLY the method body (indented, starting with try/except)",
                "2. Include proper error handling with ModelOnexError",
                "3. Add emit_log_event calls for INFO and ERROR",
                "4. Use type hints for all variables (already imported: os, Dict, List, Optional, Any)",
                "5. Follow ONEX patterns (correlation tracking, structured logging)",
                "6. No hardcoded secrets or sensitive data",
                "",
                "## IMPORTANT: Imports Already Available",
                "The following imports are ALREADY available in the node file:",
                "- Standard library: os",
                "- Typing: Any, Dict, List, Optional",
                "- External dependencies: Check if hvac, requests, etc. are needed",
                "",
                "DO NOT add import statements in your implementation - assume all",
                "dependencies are already imported at the top of the file.",
                "",
                "Generate the implementation:",
            ]
        )

        return "\n".join(prompt_parts)

    def _validate_generated_code(self, code: str) -> dict[str, Any]:
        """
        Validate generated code for syntax, ONEX compliance, security.

        Returns dict with validation results.
        """
        validation = {
            "syntax_valid": False,
            "onex_compliant": False,
            "has_type_hints": False,
            "has_docstring": False,
            "security_issues": [],
        }

        # AST parsing for syntax
        try:
            ast.parse(code)
            validation["syntax_valid"] = True
        except SyntaxError:
            return validation

        # Check for ONEX patterns
        validation["onex_compliant"] = all(
            pattern in code for pattern in BusinessLogicConfig.ONEX_PATTERNS[:2]
        )  # Check at least ModelOnexError and emit_log_event

        # Check for type hints (basic check)
        validation["has_type_hints"] = "->" in code or ": " in code

        # Security checks
        for pattern, message in BusinessLogicConfig.SECURITY_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                cast(list, validation["security_issues"]).append(message)

        return validation

    def _inject_implementation(
        self, node_file: str, method_name: str, implementation: str
    ) -> str:
        """
        Inject generated implementation into node file.

        Replaces stub marker with actual implementation.
        """
        # Find method stub and replace
        # Pattern captures: signature + optional docstring, then replaces stub body
        # Group 1: signature and docstring (to keep)
        # Stub body: everything after docstring (to replace)
        # Note: Class methods have 4-space indentation, bodies have 8-space indentation
        stub_pattern = (
            rf"(    async def {method_name}\([^)]+\)[^:]*:.*?\n"  # Signature (4 spaces)
            r"(?:        \"\"\"[\s\S]*?\"\"\"\n)?)"  # Optional multiline docstring (8 spaces)
            r"        # IMPLEMENTATION REQUIRED\n"  # Stub marker (8 spaces)
            r"        pass"  # Stub body (8 spaces)
        )

        # Indent all lines of implementation by 8 spaces for proper method body indentation
        # This ensures multi-line code (including try/except blocks) is properly indented
        indented_implementation = "\n".join(
            "        " + line if line.strip() else line
            for line in implementation.splitlines()
        )

        # Replace with signature/docstring + indented implementation
        replacement = rf"\1{indented_implementation}"

        enhanced = re.sub(stub_pattern, replacement, node_file, flags=re.DOTALL)

        return enhanced

    async def cleanup(self) -> None:
        """Cleanup LLM node resources."""
        if self.llm_node:
            await self.llm_node.cleanup()


__all__ = ["BusinessLogicGenerator"]
