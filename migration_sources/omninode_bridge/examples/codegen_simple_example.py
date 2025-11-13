#!/usr/bin/env python3
"""
Simple Code Generation Example.

Demonstrates the complete code generation workflow:
1. PRD Analysis
2. Node Classification
3. Code Generation
4. Quality Validation

Usage:
    python examples/codegen_simple_example.py
"""

import asyncio
from pathlib import Path
from uuid import uuid4

from omninode_bridge.codegen import (
    NodeClassifier,
    PRDAnalyzer,
    QualityValidator,
    TemplateEngine,
)


async def generate_postgres_crud_node():
    """Generate a PostgreSQL CRUD Effect node."""
    print("🚀 OmniNode Code Generation Example")
    print("=" * 60)

    # Configuration
    prompt = """
    Create a PostgreSQL CRUD Effect node with the following features:
    - Connection pooling (10-50 connections)
    - Automatic retry logic with exponential backoff
    - Circuit breaker pattern for resilience
    - Full CRUD operations: Create, Read, Update, Delete, List
    - Structured logging and metrics collection
    - Async/await support for all operations
    """

    output_dir = Path("./generated_nodes") / "postgres_crud_effect" / str(uuid4())[:8]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📝 Prompt:")
    print(prompt.strip())

    # Step 1: PRD Analysis
    print("\n" + "=" * 60)
    print("STEP 1: PRD Analysis & Requirement Extraction")
    print("=" * 60)

    analyzer = PRDAnalyzer(enable_intelligence=False)  # Disable for example
    requirements = await analyzer.analyze_prompt(
        prompt=prompt,
        correlation_id=uuid4(),
    )

    print("\n✅ Requirements extracted:")
    print(f"   Node Type: {requirements.node_type}")
    print(f"   Service Name: {requirements.service_name}")
    print(f"   Domain: {requirements.domain}")
    print(f"   Operations: {', '.join(requirements.operations)}")
    print(f"   Features: {', '.join(requirements.features[:5])}")
    print(f"   Confidence: {requirements.extraction_confidence:.1%}")

    # Step 2: Node Classification
    print("\n" + "=" * 60)
    print("STEP 2: Node Type Classification & Template Selection")
    print("=" * 60)

    classifier = NodeClassifier()
    classification = classifier.classify(requirements)

    print("\n✅ Classification complete:")
    print(f"   Node Type: {classification.node_type.value}")
    print(f"   Confidence: {classification.confidence:.1%}")
    print(f"   Template: {classification.template_name}")
    if classification.template_variant:
        print(f"   Variant: {classification.template_variant}")

    print("\n   Primary Indicators:")
    for indicator in classification.primary_indicators:
        print(f"      • {indicator}")

    if classification.alternatives:
        print("\n   Alternative Classifications:")
        for alt in classification.alternatives:
            print(f"      • {alt['node_type']}: {alt['confidence']:.1%} confidence")

    # Step 3: Code Generation
    print("\n" + "=" * 60)
    print("STEP 3: Code Generation from Templates")
    print("=" * 60)

    engine = TemplateEngine(enable_inline_templates=True)
    artifacts = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=output_dir,
    )

    print("\n✅ Code generated:")
    print(f"   Node Name: {artifacts.node_name}")
    print(f"   Service Name: {artifacts.service_name}")
    print(f"   Node Type: {artifacts.node_type}")

    all_files = artifacts.get_all_files()
    print(f"\n   Generated Files ({len(all_files)}):")
    for filename in sorted(all_files.keys()):
        file_size = len(all_files[filename])
        print(f"      • {filename} ({file_size} bytes)")

    # Step 4: Quality Validation
    print("\n" + "=" * 60)
    print("STEP 4: Quality Validation & ONEX Compliance")
    print("=" * 60)

    validator = QualityValidator(
        enable_mypy=False,
        enable_ruff=False,
        min_quality_threshold=0.7,
    )
    validation = await validator.validate(artifacts)

    print("\n✅ Validation complete:")
    print(f"   Overall Quality: {validation.quality_score:.1%}")
    print(f"   Status: {'✅ PASSED' if validation.passed else '❌ FAILED'}")

    print("\n   Component Scores:")
    print(f"      • ONEX Compliance: {validation.onex_compliance_score:.1%}")
    print(f"      • Type Safety: {validation.type_safety_score:.1%}")
    print(f"      • Code Quality: {validation.code_quality_score:.1%}")
    print(f"      • Documentation: {validation.documentation_score:.1%}")
    print(f"      • Test Coverage: {validation.test_coverage_score:.1%}")

    if validation.errors:
        print(f"\n   ❌ Errors ({len(validation.errors)}):")
        for error in validation.errors:
            print(f"      • {error}")

    if validation.warnings:
        print(f"\n   ⚠️  Warnings ({len(validation.warnings)}):")
        for warning in validation.warnings[:3]:  # Show first 3
            print(f"      • {warning}")

    if validation.suggestions:
        print(f"\n   💡 Suggestions ({len(validation.suggestions)}):")
        for suggestion in validation.suggestions[:3]:  # Show first 3
            print(f"      • {suggestion}")

    # Step 5: Write Files to Disk
    print("\n" + "=" * 60)
    print("STEP 5: Writing Generated Files to Disk")
    print("=" * 60)

    files_written = []
    for filename, content in all_files.items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        files_written.append(file_path)

    print(f"\n✅ Files written to: {output_dir}")
    print(f"   Total files: {len(files_written)}")

    # Summary
    print("\n" + "=" * 60)
    print("✨ Code Generation Complete!")
    print("=" * 60)

    print("\n📦 Summary:")
    print(f"   Node Class: {artifacts.node_name}")
    print(f"   Output Directory: {output_dir}")
    print(f"   Files Generated: {len(all_files)}")
    print(f"   Quality Score: {validation.quality_score:.1%}")
    print(f"   Validation: {'✅ PASSED' if validation.passed else '❌ FAILED'}")

    print("\n🔍 Next Steps:")
    print(f"   1. Review generated code in: {output_dir}")
    print("   2. Implement TODO items marked in node.py")
    print(f"   3. Run tests: pytest {output_dir}/tests/")
    print("   4. Integrate with your project")

    return artifacts, validation


async def main():
    """Run the example."""
    try:
        artifacts, validation = await generate_postgres_crud_node()
        print("\n✅ Example completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Example failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
