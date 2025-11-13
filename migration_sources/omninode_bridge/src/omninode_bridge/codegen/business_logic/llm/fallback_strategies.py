#!/usr/bin/env python3
"""
Fallback Strategies for LLM Code Generation.

Provides multiple fallback strategies when LLM generation fails:
- Strategy 1: Retry with adjusted prompt
- Strategy 2: Template fallback (use stub with TODO comments)
- Strategy 3: Partial generation (simpler version)

Performance Target: Each strategy <100ms
"""

import logging
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FallbackMetrics(BaseModel):
    """Metrics for fallback operations."""

    strategy_used: str = Field(..., description="Fallback strategy name")
    retry_attempts: int = Field(default=0, description="Number of retry attempts")
    fallback_reason: str = Field(..., description="Why fallback was triggered")
    fallback_successful: bool = Field(..., description="Whether fallback succeeded")
    fallback_time_ms: float = Field(..., description="Time spent in fallback")


class FallbackStrategy:
    """
    Base class for fallback strategies.

    Fallback strategies are invoked when LLM generation fails or produces
    invalid code. Each strategy provides a different approach to recover
    from failures.

    Strategy 1: Retry with Adjusted Prompt
    - Simplify context
    - Focus on core requirements
    - Use different LLM tier
    - Max 3 attempts

    Strategy 2: Template Fallback
    - Use template stub with TODO comments
    - Log fallback event
    - Continue generation
    - Mark as incomplete

    Strategy 3: Partial Generation
    - Generate simpler version
    - Reduce complexity
    - Mark methods as incomplete
    - Provide manual completion guide
    """

    def __init__(self):
        """Initialize fallback strategy."""
        self.max_retries = 3
        self.retry_delay_ms = 100

    def should_retry(
        self,
        validation_result: Any,  # ModelValidationResult
        attempt: int,
    ) -> bool:
        """
        Determine if retry is appropriate.

        Args:
            validation_result: Validation result from previous attempt
            attempt: Current attempt number (1-indexed)

        Returns:
            True if retry should be attempted
        """
        # Don't retry if max attempts reached
        if attempt >= self.max_retries:
            return False

        # Retry if validation says it's recoverable
        if hasattr(validation_result, "can_retry"):
            return validation_result.can_retry

        # Retry if syntax errors (might be due to prompt misunderstanding)
        if hasattr(validation_result, "syntax_valid"):
            return not validation_result.syntax_valid

        return False

    def adjust_prompt_for_retry(
        self,
        original_prompt: str,
        validation_result: Any,  # ModelValidationResult
        attempt: int,
    ) -> str:
        """
        Adjust prompt for retry attempt.

        Strategies for adjustment:
        - Simplify context (remove references on 2nd attempt)
        - Emphasize failing requirements
        - Add explicit examples of what went wrong
        - Use more direct instructions

        Args:
            original_prompt: Original LLM prompt
            validation_result: Validation result showing what failed
            attempt: Current attempt number

        Returns:
            Adjusted prompt string
        """
        adjustments = []

        # Add retry header
        adjustments.append(f"\n\n# RETRY ATTEMPT {attempt}/{self.max_retries}\n")
        adjustments.append(
            "The previous attempt had issues. Please focus on these requirements:\n"
        )

        # Emphasize what failed
        if hasattr(validation_result, "issues") and validation_result.issues:
            adjustments.append("\n**Issues to Fix:**")
            for issue in validation_result.issues[:3]:
                adjustments.append(f"- {issue}")

        # Simplify on 2nd+ attempts
        if attempt >= 2:
            adjustments.append(
                "\n**Simplified Requirements:**\n"
                "Focus on the core functionality. Ensure:\n"
                "1. Syntax is valid Python\n"
                "2. Uses async/await\n"
                "3. Includes ModelOnexError for errors\n"
                "4. Includes emit_log_event for logging\n"
                "5. Returns ModelContractResponse\n"
            )

        # Combine original prompt with adjustments
        adjusted = original_prompt + "\n" + "\n".join(adjustments)

        logger.debug(f"Adjusted prompt for retry attempt {attempt}")

        return adjusted

    def generate_template_fallback(
        self,
        method_name: str,
        method_signature: str,
        business_description: str,
    ) -> str:
        """
        Generate template stub fallback (Strategy 2).

        Creates a stub implementation with TODO comments that can be
        manually completed later.

        Args:
            method_name: Method name
            method_signature: Method signature
            business_description: What the method should do

        Returns:
            Stub implementation code
        """
        stub = f"""        # TODO: Implement {method_name}
        # Business logic: {business_description[:100]}
        #
        # This is a fallback stub because LLM generation failed.
        # Please implement the following:
        # 1. Validate inputs from contract.input_state
        # 2. Implement business logic
        # 3. Include error handling with ModelOnexError
        # 4. Add emit_log_event calls for INFO and ERROR
        # 5. Return ModelContractResponse with results
        #
        # Required imports are already available at the top of the file.

        try:
            emit_log_event(
                level=EnumLogLevel.INFO,
                message="Starting {method_name} (STUB IMPLEMENTATION)",
                details={{"correlation_id": contract.correlation_id}},
            )

            # TODO: Implement your logic here

            return ModelContractResponse(
                success=False,
                output_state={{"error": "Method not implemented"}},
                metadata={{"stub": True}},
            )

        except Exception as e:
            emit_log_event(
                level=EnumLogLevel.ERROR,
                message="{method_name} failed (STUB IMPLEMENTATION)",
                details={{
                    "correlation_id": contract.correlation_id,
                    "error": str(e),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"{method_name} not implemented: {{e}}",
            )
"""

        logger.info(f"Generated template fallback stub for {method_name}")

        return stub

    def generate_partial_implementation(
        self,
        method_name: str,
        method_signature: str,
        business_description: str,
        node_type: str,
    ) -> str:
        """
        Generate partial implementation (Strategy 3).

        Creates a simplified, working implementation that handles
        basic cases but may not include full functionality.

        Args:
            method_name: Method name
            method_signature: Method signature
            business_description: What the method should do
            node_type: Node type (EFFECT/COMPUTE/etc.)

        Returns:
            Partial implementation code
        """
        # Generate based on node type
        if node_type.upper() == "EFFECT":
            impl = self._generate_effect_partial(method_name, business_description)
        elif node_type.upper() == "COMPUTE":
            impl = self._generate_compute_partial(method_name, business_description)
        elif node_type.upper() == "REDUCER":
            impl = self._generate_reducer_partial(method_name, business_description)
        elif node_type.upper() == "ORCHESTRATOR":
            impl = self._generate_orchestrator_partial(
                method_name, business_description
            )
        else:
            # Generic partial implementation
            impl = self._generate_generic_partial(method_name, business_description)

        logger.info(f"Generated partial implementation for {method_name} ({node_type})")

        return impl

    def _generate_effect_partial(
        self,
        method_name: str,
        description: str,
    ) -> str:
        """Generate partial Effect node implementation."""
        return f"""        # PARTIAL IMPLEMENTATION: {method_name}
        # Note: This is a simplified version. Full implementation needed.

        # Validate inputs
        if not contract.input_state:
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.INVALID_INPUT,
                message="Contract input_state is required",
            )

        try:
            emit_log_event(
                level=EnumLogLevel.INFO,
                message="Starting {method_name}",
                details={{"correlation_id": contract.correlation_id}},
            )

            # TODO: Implement {description[:50]}
            # Add your I/O logic here (database, API, file system, etc.)

            result = {{"status": "partial_implementation"}}

            emit_log_event(
                level=EnumLogLevel.INFO,
                message="{method_name} completed (partial)",
                details={{"correlation_id": contract.correlation_id}},
            )

            return ModelContractResponse(
                success=True,
                output_state=result,
                metadata={{"partial": True}},
            )

        except ModelOnexError:
            raise

        except Exception as e:
            emit_log_event(
                level=EnumLogLevel.ERROR,
                message="{method_name} failed",
                details={{
                    "correlation_id": contract.correlation_id,
                    "error": str(e),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Operation failed: {{e}}",
            )
"""

    def _generate_compute_partial(
        self,
        method_name: str,
        description: str,
    ) -> str:
        """Generate partial Compute node implementation."""
        return f"""        # PARTIAL IMPLEMENTATION: {method_name}
        # Note: This is a simplified version. Full implementation needed.

        try:
            emit_log_event(
                level=EnumLogLevel.INFO,
                message="Starting {method_name}",
                details={{"correlation_id": contract.correlation_id}},
            )

            # TODO: Implement {description[:50]}
            # Add your computation logic here (pure function, no I/O)

            input_data = contract.input_state.get("data", {{}})
            result = input_data  # Passthrough for now

            emit_log_event(
                level=EnumLogLevel.INFO,
                message="{method_name} completed (partial)",
                details={{"correlation_id": contract.correlation_id}},
            )

            return ModelContractResponse(
                success=True,
                output_state={{"result": result}},
                metadata={{"partial": True}},
            )

        except Exception as e:
            emit_log_event(
                level=EnumLogLevel.ERROR,
                message="{method_name} failed",
                details={{
                    "correlation_id": contract.correlation_id,
                    "error": str(e),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Computation failed: {{e}}",
            )
"""

    def _generate_reducer_partial(
        self,
        method_name: str,
        description: str,
    ) -> str:
        """Generate partial Reducer node implementation."""
        return f"""        # PARTIAL IMPLEMENTATION: {method_name}
        # Note: This is a simplified version. Full implementation needed.

        try:
            emit_log_event(
                level=EnumLogLevel.INFO,
                message="Starting {method_name}",
                details={{"correlation_id": contract.correlation_id}},
            )

            # TODO: Implement {description[:50]}
            # Add your aggregation logic here

            items = contract.input_state.get("items", [])
            result = {{"count": len(items), "items_processed": len(items)}}

            emit_log_event(
                level=EnumLogLevel.INFO,
                message="{method_name} completed (partial)",
                details={{"correlation_id": contract.correlation_id}},
            )

            return ModelContractResponse(
                success=True,
                output_state=result,
                metadata={{"partial": True}},
            )

        except Exception as e:
            emit_log_event(
                level=EnumLogLevel.ERROR,
                message="{method_name} failed",
                details={{
                    "correlation_id": contract.correlation_id,
                    "error": str(e),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Aggregation failed: {{e}}",
            )
"""

    def _generate_orchestrator_partial(
        self,
        method_name: str,
        description: str,
    ) -> str:
        """Generate partial Orchestrator node implementation."""
        return f"""        # PARTIAL IMPLEMENTATION: {method_name}
        # Note: This is a simplified version. Full implementation needed.

        try:
            emit_log_event(
                level=EnumLogLevel.INFO,
                message="Starting {method_name}",
                details={{"correlation_id": contract.correlation_id}},
            )

            # TODO: Implement {description[:50]}
            # Add your workflow orchestration logic here

            steps_completed = []
            result = {{"steps": steps_completed, "status": "partial"}}

            emit_log_event(
                level=EnumLogLevel.INFO,
                message="{method_name} completed (partial)",
                details={{"correlation_id": contract.correlation_id}},
            )

            return ModelContractResponse(
                success=True,
                output_state=result,
                metadata={{"partial": True}},
            )

        except Exception as e:
            emit_log_event(
                level=EnumLogLevel.ERROR,
                message="{method_name} failed",
                details={{
                    "correlation_id": contract.correlation_id,
                    "error": str(e),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Orchestration failed: {{e}}",
            )
"""

    def _generate_generic_partial(
        self,
        method_name: str,
        description: str,
    ) -> str:
        """Generate generic partial implementation."""
        return f"""        # PARTIAL IMPLEMENTATION: {method_name}
        # Note: This is a simplified version. Full implementation needed.

        try:
            emit_log_event(
                level=EnumLogLevel.INFO,
                message="Starting {method_name}",
                details={{"correlation_id": contract.correlation_id}},
            )

            # TODO: Implement {description[:50]}

            result = {{"status": "partial_implementation"}}

            emit_log_event(
                level=EnumLogLevel.INFO,
                message="{method_name} completed (partial)",
                details={{"correlation_id": contract.correlation_id}},
            )

            return ModelContractResponse(
                success=True,
                output_state=result,
                metadata={{"partial": True}},
            )

        except Exception as e:
            emit_log_event(
                level=EnumLogLevel.ERROR,
                message="{method_name} failed",
                details={{
                    "correlation_id": contract.correlation_id,
                    "error": str(e),
                }},
            )
            raise ModelOnexError(
                error_code=EnumCoreErrorCode.EXECUTION_ERROR,
                message=f"Operation failed: {{e}}",
            )
"""


__all__ = ["FallbackStrategy", "FallbackMetrics"]
