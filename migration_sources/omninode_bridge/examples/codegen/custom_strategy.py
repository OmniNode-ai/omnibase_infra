#!/usr/bin/env python3
"""
Custom Strategy Implementation Example.

Demonstrates how to create and register a custom code generation strategy.

Usage:
    python examples/codegen/custom_strategy.py
"""

import asyncio
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from omninode_bridge.codegen import CodeGenerationService, ModelPRDRequirements
from omninode_bridge.codegen.node_classifier import EnumNodeType
from omninode_bridge.codegen.strategies.base import (
    BaseGenerationStrategy,
    EnumStrategyType,
    ModelGenerationRequest,
    ModelGenerationResult,
)
from omninode_bridge.codegen.template_engine import ModelGeneratedArtifacts


class SimpleCustomStrategy(BaseGenerationStrategy):
    """
    Simple custom strategy example.

    Generates minimal node code with custom template.
    Demonstrates the minimum required to implement a strategy.
    """

    def __init__(self):
        """Initialize custom strategy."""
        super().__init__(
            strategy_name="Simple Custom Strategy",
            strategy_type=EnumStrategyType.JINJA2,  # Reuse existing enum for demo
            enable_validation=True,
        )

    async def generate(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        """
        Generate code using custom logic.

        Args:
            request: Generation request with requirements

        Returns:
            ModelGenerationResult with generated artifacts
        """
        start_time = time.perf_counter()

        self.log_generation_start(request)

        # Step 1: Validate requirements
        is_valid, errors = self.validate_requirements(
            request.requirements, request.validation_level
        )
        if not is_valid:
            raise ValueError(f"Requirements validation failed: {errors}")

        # Step 2: Generate artifacts using custom logic
        artifacts = await self._generate_artifacts(request)

        # Step 3: Calculate generation time
        end_time = time.perf_counter()
        generation_time_ms = (end_time - start_time) * 1000

        # Step 4: Create result
        result = ModelGenerationResult(
            artifacts=artifacts,
            strategy_used=self.strategy_type,
            generation_time_ms=generation_time_ms,
            validation_passed=True,  # Simple validation
            validation_errors=[],
            llm_used=False,
            intelligence_sources=[],
            correlation_id=request.correlation_id,
        )

        self.log_generation_complete(result, generation_time_ms)

        return result

    async def _generate_artifacts(
        self, request: ModelGenerationRequest
    ) -> ModelGeneratedArtifacts:
        """
        Generate artifacts using custom template.

        This is where your custom generation logic goes.
        """
        req = request.requirements
        cls = request.classification

        # Generate node name
        node_name = (
            f"Node{self._to_pascal_case(req.service_name)}{cls.node_type.value.title()}"
        )

        # Generate simple node implementation
        node_code = self._generate_node_code(node_name, req)

        # Generate simple contract
        contract_code = self._generate_contract(req)

        # Generate simple test
        test_code = self._generate_test(node_name, req)

        # Create artifacts
        artifacts = ModelGeneratedArtifacts(
            node_name=node_name,
            service_name=req.service_name,
            node_type=cls.node_type.value,
            node_file=node_code,
            contract_file=contract_code,
            models_file="# Models placeholder\n",
            test_files={"test_node.py": test_code},
            metadata={
                "strategy": self.strategy_name,
                "custom": True,
            },
        )

        return artifacts

    def _generate_node_code(self, node_name: str, requirements) -> str:
        """Generate minimal node implementation."""
        operations = "\n    ".join(
            [
                f'async def {op}(self):\n        """TODO: Implement {op}."""\n        pass'
                for op in requirements.operations[:3]  # First 3 operations
            ]
        )

        return f'''"""
{node_name} - Custom Generated Node.

{requirements.business_description}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class {node_name}:
    """
    {requirements.business_description}

    Operations: {", ".join(requirements.operations)}
    """

    def __init__(self):
        """Initialize node."""
        self.logger = logger

    {operations}

    async def execute(self, **kwargs: Any) -> dict:
        """Execute node logic."""
        self.logger.info("Executing {node_name}")
        return {{"status": "success", "data": {{}}}}
'''

    def _generate_contract(self, requirements) -> str:
        """Generate minimal contract."""
        return f"""# {requirements.service_name} Contract

version: "1.0.0"
node_type: "{requirements.node_type}"
service_name: "{requirements.service_name}"
domain: "{requirements.domain}"

operations:
{chr(10).join(f'  - {op}' for op in requirements.operations)}

features:
{chr(10).join(f'  - {feat}' for feat in requirements.features[:3])}
"""

    def _generate_test(self, node_name: str, requirements) -> str:
        """Generate minimal test."""
        return f'''"""
Tests for {node_name}.
"""

import pytest


@pytest.mark.asyncio
async def test_{requirements.service_name}_execute():
    """Test basic execution."""
    from node import {node_name}

    node = {node_name}()
    result = await node.execute()

    assert result["status"] == "success"
'''

    def _to_pascal_case(self, snake_str: str) -> str:
        """Convert snake_case to PascalCase."""
        return "".join(word.capitalize() for word in snake_str.split("_"))

    def supports_node_type(self, node_type: EnumNodeType) -> bool:
        """Check if strategy supports node type."""
        # This simple strategy supports all node types
        return True

    def get_strategy_info(self) -> dict[str, Any]:
        """Get strategy information."""
        return {
            "name": self.strategy_name,
            "type": self.strategy_type.value,
            "supported_node_types": ["effect", "compute", "reducer", "orchestrator"],
            "requires_llm": False,
            "performance_profile": "very_fast",
            "description": "Simple custom strategy for demonstration",
        }


async def use_custom_strategy():
    """
    Example: Use custom strategy.

    Demonstrates:
    1. Create custom strategy
    2. Register with service
    3. Use in generation
    4. Compare with built-in strategies
    """
    print("🎨 Custom Strategy Example")
    print("=" * 60)

    # Step 1: Initialize service
    service = CodeGenerationService()

    # Step 2: Create and register custom strategy
    print("\n📝 Registering custom strategy...")
    custom_strategy = SimpleCustomStrategy()
    service.strategy_registry.register(
        strategy=custom_strategy,
        is_default=False,  # Don't make it default
    )
    print("✅ Custom strategy registered")

    # Step 3: List all strategies (including custom)
    print("\n📋 Available strategies:")
    strategies = service.list_strategies()
    for strategy in strategies:
        default_marker = " [DEFAULT]" if strategy["is_default"] else ""
        print(f"   - {strategy['name']} ({strategy['type']}){default_marker}")

    # Step 4: Use custom strategy
    print("\n⏳ Generating code with custom strategy...")

    requirements = ModelPRDRequirements(
        node_type="effect",
        service_name="custom_test",
        domain="testing",
        business_description="Testing custom strategy",
        operations=["test_op1", "test_op2", "test_op3"],
        features=["feature1", "feature2"],
    )

    output_dir = Path(f"./generated/custom_test_{uuid4().hex[:8]}")

    result = await service.generate_node(
        requirements=requirements,
        output_directory=output_dir,
        strategy="jinja2",  # Use custom strategy (mapped to jinja2 for demo)
        validation_level="basic",
    )

    print("\n✅ Generation complete!")
    print(f"   Node: {result.artifacts.node_name}")
    print(f"   Time: {result.generation_time_ms:.0f}ms")
    print(f"   Strategy: {result.strategy_used.value}")

    # Step 5: Write files
    for filename, content in result.artifacts.get_all_files().items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    print(f"\n💾 Files written to: {output_dir}")

    # Step 6: Show custom artifacts
    print("\n📄 Custom generated node code (first 20 lines):")
    print("-" * 60)
    node_lines = result.artifacts.node_file.split("\n")[:20]
    for i, line in enumerate(node_lines, 1):
        print(f"{i:3}: {line}")
    print("    ...")

    return result


async def advanced_custom_strategy():
    """
    Advanced example: Custom strategy with more features.

    Demonstrates:
    - Custom validation rules
    - Intelligence integration
    - Performance tracking
    - Error handling
    """
    print("\n" + "=" * 60)
    print("🚀 Advanced Custom Strategy Concepts")
    print("=" * 60)

    print("\n💡 Advanced Features You Can Add:")
    print("\n1. Custom Validation Rules:")
    print("   - Add domain-specific checks")
    print("   - Enforce coding standards")
    print("   - Validate against security policies")

    print("\n2. Intelligence Integration:")
    print("   - Query RAG for similar implementations")
    print("   - Use Archon MCP for best practices")
    print("   - Learn from existing codebase")

    print("\n3. Performance Tracking:")
    print("   - Measure generation time by component")
    print("   - Track cache hit rates")
    print("   - Monitor memory usage")

    print("\n4. Error Recovery:")
    print("   - Automatic retry on failure")
    print("   - Fallback to simpler generation")
    print("   - Partial artifact recovery")

    print("\n5. Multi-Stage Generation:")
    print("   - Generate base code first")
    print("   - Apply enhancements incrementally")
    print("   - Allow user feedback between stages")

    print("\n📖 Implementation Pattern:")
    print(
        """
    class AdvancedCustomStrategy(BaseGenerationStrategy):
        async def generate(self, request):
            # Stage 1: Generate base
            base = await self._generate_base(request)

            # Stage 2: Apply intelligence
            enhanced = await self._apply_intelligence(base)

            # Stage 3: Optimize
            optimized = await self._optimize(enhanced)

            # Stage 4: Validate
            validated = await self._comprehensive_validation(optimized)

            return validated
    """
    )


async def main():
    """Run all examples."""
    try:
        # Basic custom strategy
        result = await use_custom_strategy()

        # Advanced concepts
        await advanced_custom_strategy()

        print("\n" + "=" * 60)
        print("✨ Custom strategy examples completed!")
        print("=" * 60)

        print("\n🔍 Key Takeaways:")
        print("   1. Custom strategies extend BaseGenerationStrategy")
        print("   2. Register with service.strategy_registry.register()")
        print("   3. Implement generate(), supports_node_type(), get_strategy_info()")
        print("   4. Add custom validation, intelligence, and features as needed")
        print("   5. Can coexist with built-in strategies")

        print("\n📚 Next Steps:")
        print("   1. Implement your custom generation logic")
        print("   2. Add domain-specific features")
        print("   3. Test thoroughly with various requirements")
        print("   4. Share your strategy with the team!")

        return 0

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
