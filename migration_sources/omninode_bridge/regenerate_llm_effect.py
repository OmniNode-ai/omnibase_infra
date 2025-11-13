#!/usr/bin/env python3
"""
Regenerate llm_effect node with mixin-enhanced code generation.

Wave 4 Phase 1: First high-priority node regeneration.
"""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from src.omninode_bridge.codegen import CodeGenerationService
from src.omninode_bridge.codegen.prd_analyzer import ModelPRDRequirements
from src.omninode_bridge.codegen.yaml_contract_parser import YAMLContractParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def regenerate_llm_effect_node():
    """
    Regenerate llm_effect node with mixin enhancements.

    Steps:
    1. Parse enhanced contract (contract_v2.yaml)
    2. Create requirements from contract
    3. Generate new node with mixins enabled
    4. Compare and validate
    """
    print("=" * 80)
    print("🚀 Regenerating llm_effect Node (Wave 4 Phase 1)")
    print("=" * 80)

    # Paths
    node_dir = Path("src/omninode_bridge/nodes/llm_effect/v1_0_0")
    contract_v2_path = node_dir / "contract_v2.yaml"
    backup_dir = Path(str(node_dir) + ".backup")
    output_dir = node_dir / "generated"

    # Verify contract exists
    if not contract_v2_path.exists():
        raise FileNotFoundError(f"Contract not found: {contract_v2_path}")

    print(f"\n📄 Contract: {contract_v2_path}")
    print(f"📦 Output: {output_dir}")
    print(f"💾 Backup: {backup_dir}")

    # Step 1: Parse contract
    print("\n" + "=" * 80)
    print("Step 1: Parse Enhanced Contract")
    print("=" * 80)

    parser = YAMLContractParser()
    contract = parser.parse_contract_file(str(contract_v2_path))

    if not contract.is_valid:
        print(f"❌ Contract validation failed:")
        for error in contract.validation_errors:
            print(f"   - {error}")
        return 1

    print(f"✅ Contract parsed: {contract.name} v{contract.version}")
    print(f"✅ Mixins: {', '.join(contract.get_mixin_names())}")
    print(f"✅ Node Type: {contract.node_type}")

    # Step 2: Create requirements from contract
    print("\n" + "=" * 80)
    print("Step 2: Create Requirements from Contract")
    print("=" * 80)

    # Extract operations from contract
    operations = []
    if contract.io_operations:
        operations = [op.get("operation_type", "unknown") for op in contract.io_operations]

    requirements = ModelPRDRequirements(
        node_type=contract.node_type,
        service_name=contract.name,
        domain="ai-services",
        business_description=contract.description or "LLM Effect Node for multi-tier LLM generation",
        operations=operations or ["generate_text"],
        features=[
            "Multi-tier LLM support (LOCAL, CLOUD_FAST, CLOUD_PREMIUM)",
            "Circuit breaker pattern",
            "Retry logic with exponential backoff",
            "Token usage tracking",
            "Cost tracking",
            "Comprehensive metrics (MixinMetrics)",
            "Health checks (MixinHealthCheck)",
            "Z.ai API integration",
        ],
    )

    print(f"✅ Requirements created:")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Service: {requirements.service_name}")
    print(f"   Domain: {requirements.domain}")
    print(f"   Operations: {', '.join(requirements.operations)}")
    print(f"   Features: {len(requirements.features)}")

    # Step 3: Initialize service and generate
    print("\n" + "=" * 80)
    print("Step 3: Generate Node with Mixin Enhancement")
    print("=" * 80)

    service = CodeGenerationService()
    correlation_id = uuid4()

    print(f"⏳ Generating code...")
    print(f"   Strategy: auto")
    print(f"   Mixins: enabled")
    print(f"   Validation: strict")
    print(f"   Correlation ID: {correlation_id}")

    result = await service.generate_node(
        requirements=requirements,
        output_directory=output_dir,
        strategy="auto",
        enable_llm=True,
        enable_mixins=True,
        validation_level="strict",
        run_tests=False,  # Run tests separately
        correlation_id=correlation_id,
        contract_path=contract_v2_path,
    )

    # Step 4: Display results
    print("\n" + "=" * 80)
    print("Step 4: Generation Results")
    print("=" * 80)

    print(f"\n✅ Generation complete!")
    print(f"   Node Name: {result.artifacts.node_name}")
    print(f"   Strategy: {result.strategy_used.value}")
    print(f"   Time: {result.generation_time_ms:.0f}ms")
    print(f"   Validation: {'✅ PASSED' if result.validation_passed else '❌ FAILED'}")
    print(f"   LLM Used: {'Yes' if result.llm_used else 'No'}")

    if result.mixins_applied:
        print(f"\n✅ Mixins Applied ({len(result.mixins_applied)}):")
        for mixin in result.mixins_applied:
            print(f"   - {mixin}")

    # List generated files
    all_files = result.artifacts.get_all_files()
    print(f"\n📦 Generated {len(all_files)} files:")
    for filename in sorted(all_files.keys()):
        file_size = len(all_files[filename])
        print(f"   - {filename} ({file_size:,} bytes)")

    # Write files
    print(f"\n💾 Writing files to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in all_files.items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"   ✅ {filename}")

    print(f"\n✅ Files written to: {output_dir.absolute()}")

    # Step 5: Validation details
    if result.validation_errors:
        print(f"\n⚠️  Validation Errors ({len(result.validation_errors)}):")
        for error in result.validation_errors[:10]:
            print(f"   - {error}")

    if result.validation_warnings:
        print(f"\n⚠️  Validation Warnings ({len(result.validation_warnings)}):")
        for warning in result.validation_warnings[:10]:
            print(f"   - {warning}")

    # Step 6: LOC comparison
    print("\n" + "=" * 80)
    print("Step 5: LOC Comparison")
    print("=" * 80)

    original_node = backup_dir / "node.py"
    generated_node = output_dir / "node.py"

    if original_node.exists() and generated_node.exists():
        original_loc = count_code_loc(original_node)
        generated_loc = count_code_loc(generated_node)
        reduction = original_loc - generated_loc
        percentage = (reduction / original_loc) * 100 if original_loc > 0 else 0

        print(f"\n📊 LOC Metrics:")
        print(f"   Original: {original_loc} lines")
        print(f"   Generated: {generated_loc} lines")
        print(f"   Reduction: {reduction} lines ({percentage:.1f}%)")
        print(f"   Target: 26-31% reduction (92-110 lines)")

        if 26 <= percentage <= 35:
            print(f"   Status: ✅ SUCCESS - Within target range")
        elif percentage > 0:
            print(f"   Status: ⚠️  REVIEW - Outside target range")
        else:
            print(f"   Status: ❌ FAILURE - No reduction achieved")
    else:
        print("⚠️  Could not perform LOC comparison (files not found)")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(f"\n✅ Regeneration Complete!")
    print(f"\n📁 Generated files: {output_dir}")
    print(f"📁 Original backup: {backup_dir}")
    print(f"\n🔍 Next Steps:")
    print(f"   1. Review generated code: {output_dir / 'node.py'}")
    print(f"   2. Compare with original: {backup_dir / 'node.py'}")
    print(f"   3. Run tests: pytest tests/unit/nodes/llm_effect/")
    print(f"   4. If tests pass, replace original with generated code")

    return 0


def count_code_loc(file_path: Path) -> int:
    """
    Count LOC excluding blank lines, comments, and docstrings.

    Args:
        file_path: Path to Python file

    Returns:
        Number of code lines
    """
    with open(file_path) as f:
        lines = f.readlines()

    code_lines = 0
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Skip blank lines
        if not stripped:
            continue

        # Handle docstrings
        if '"""' in stripped or "'''" in stripped:
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                # Single-line docstring
                continue
            in_docstring = not in_docstring
            continue

        if in_docstring:
            continue

        # Skip comment-only lines
        if stripped.startswith('#'):
            continue

        code_lines += 1

    return code_lines


async def main():
    """Main entry point."""
    try:
        exit_code = await regenerate_llm_effect_node()
        return exit_code
    except Exception as e:
        logger.error(f"Regeneration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
