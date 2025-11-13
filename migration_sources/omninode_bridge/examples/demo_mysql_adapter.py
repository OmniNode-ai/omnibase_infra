#!/usr/bin/env python3
"""
MySQL Database Adapter Code Generation Demonstration.

Demonstrates the complete ONEX-compliant code generation workflow:
1. PRD Analysis
2. Node Classification
3. Code Generation
4. Quality Validation
5. Contract Validation
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


async def generate_mysql_adapter_node():
    """Generate a MySQL database adapter Effect node."""
    print("🚀 Code Generation System Demonstration")
    print("=" * 80)
    print("Objective: Generate ONEX-compliant MySQL Database Adapter Effect Node")
    print("=" * 80)

    # Configuration
    prompt = """
    Create a MySQL database adapter Effect node with the following features:
    - Connection pooling (10-100 connections)
    - Automatic retry logic with exponential backoff (max 3 retries)
    - Circuit breaker pattern for resilience
    - Full CRUD operations: Create, Read, Update, Delete, List, BulkInsert
    - Transaction support with rollback capability
    - Query builder for common operations
    - Prepared statements for SQL injection prevention
    - Connection health monitoring
    - Structured logging with query metrics
    - Async/await support for all operations
    - Configurable timeouts per operation
    """

    output_dir = Path("./generated_nodes") / "mysql_adapter_effect" / str(uuid4())[:8]
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n📝 Input Prompt:")
    print("-" * 80)
    print(prompt.strip())
    print("-" * 80)

    # Step 1: PRD Analysis
    print("\n" + "=" * 80)
    print("STAGE 1: PRD Analysis & Requirement Extraction")
    print("=" * 80)

    analyzer = PRDAnalyzer(enable_intelligence=False)
    requirements = await analyzer.analyze_prompt(
        prompt=prompt,
        correlation_id=uuid4(),
    )

    print("\n✅ Requirements Extracted:")
    print(f"   • Node Type Detected: {requirements.node_type.upper()}")
    print(f"   • Service Name: {requirements.service_name}")
    print(f"   • Domain: {requirements.domain}")
    print(f"   • Operations: {', '.join(requirements.operations)}")
    print(f"\n   • Key Features ({len(requirements.features)}):")
    for i, feature in enumerate(requirements.features, 1):
        print(f"      {i}. {feature}")
    print(f"\n   • Confidence: {requirements.extraction_confidence:.1%}")

    # Step 2: Node Classification
    print("\n" + "=" * 80)
    print("STAGE 2: Node Type Classification & Template Selection")
    print("=" * 80)

    classifier = NodeClassifier()
    classification = classifier.classify(requirements)

    print("\n✅ Classification Complete:")
    print(f"   • Classified Type: {classification.node_type.value.upper()}")
    print(f"   • Confidence: {classification.confidence:.1%}")
    print(f"   • Template: {classification.template_name}")
    if classification.template_variant:
        print(f"   • Variant: {classification.template_variant}")

    print("\n   • Primary Indicators:")
    for indicator in classification.primary_indicators:
        print(f"      • {indicator}")

    if classification.alternatives:
        print("\n   • Alternative Classifications:")
        for alt in classification.alternatives:
            print(f"      • {alt['node_type']}: {alt['confidence']:.1%} confidence")

    # Step 3: Code Generation
    print("\n" + "=" * 80)
    print("STAGE 3: Code Generation from Templates")
    print("=" * 80)

    engine = TemplateEngine(enable_inline_templates=True)
    artifacts = await engine.generate(
        requirements=requirements,
        classification=classification,
        output_directory=output_dir,
    )

    print("\n✅ Code Generated:")
    print(f"   • Node Name: {artifacts.node_name}")
    print(f"   • Service Name: {artifacts.service_name}")
    print(f"   • Node Type: {artifacts.node_type.upper()}")

    all_files = artifacts.get_all_files()
    print(f"\n   • Generated Files ({len(all_files)}):")
    for filename in sorted(all_files.keys()):
        file_size = len(all_files[filename])
        print(f"      • {filename} ({file_size:,} bytes)")

    # Step 4: Quality Validation
    print("\n" + "=" * 80)
    print("STAGE 4: Quality Validation & ONEX Compliance")
    print("=" * 80)

    validator = QualityValidator(
        enable_mypy=False,
        enable_ruff=False,
        min_quality_threshold=0.7,
    )
    validation = await validator.validate(artifacts)

    print("\n✅ Validation Complete:")
    print(f"   • Overall Quality: {validation.quality_score:.1%}")
    print(f"   • Status: {'✅ PASSED' if validation.passed else '❌ FAILED'}")

    print("\n   • Component Scores:")
    print(f"      • ONEX Compliance: {validation.onex_compliance_score:.1%}")
    print(f"      • Type Safety: {validation.type_safety_score:.1%}")
    print(f"      • Code Quality: {validation.code_quality_score:.1%}")
    print(f"      • Documentation: {validation.documentation_score:.1%}")
    print(f"      • Test Coverage: {validation.test_coverage_score:.1%}")

    if validation.errors:
        print(f"\n   ❌ Errors ({len(validation.errors)}):")
        for error in validation.errors[:5]:
            print(f"      • {error}")

    if validation.warnings:
        print(f"\n   ⚠️  Warnings ({len(validation.warnings)}):")
        for warning in validation.warnings[:5]:
            print(f"      • {warning}")

    # Step 5: Contract Validation
    print("\n" + "=" * 80)
    print("STAGE 5: Contract YAML Validation")
    print("=" * 80)

    contract_content = all_files.get("contract.yaml", "")
    if contract_content:
        print("\n✅ Contract Generated:")
        print(f"   • Size: {len(contract_content):,} bytes")

        # Extract key contract fields
        import yaml

        try:
            contract_data = yaml.safe_load(contract_content)

            print("\n   • Required Fields Verification:")
            required_fields = [
                "schema_version",
                "name",
                "version",
                "description",
                "node_type",
                "tool_specification",
                "io_operations",
                "performance_characteristics",
                "dependencies",
                "observability",
                "compliance",
            ]

            for field in required_fields:
                present = field in contract_data
                symbol = "✅" if present else "❌"
                print(f"      {symbol} {field}")

            # Display contract preview
            print("\n   • Contract Preview (First 20 lines):")
            print("   " + "-" * 76)
            for i, line in enumerate(contract_content.split("\n")[:20], 1):
                print(f"   {i:3d} | {line}")
            print("   " + "-" * 76)

        except Exception as e:
            print(f"   ⚠️  Could not parse contract YAML: {e}")
    else:
        print("   ❌ No contract.yaml found in generated files")

    # Step 6: Write Files to Disk
    print("\n" + "=" * 80)
    print("STAGE 6: Writing Generated Files to Disk")
    print("=" * 80)

    files_written = []
    for filename, content in all_files.items():
        file_path = output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        files_written.append(file_path)

    print(f"\n✅ Files written to: {output_dir}")
    print(f"   • Total files: {len(files_written)}")

    # Summary
    print("\n" + "=" * 80)
    print("✨ CODE GENERATION DEMONSTRATION COMPLETE")
    print("=" * 80)

    print("\n📦 Summary:")
    print(f"   • Node Class: {artifacts.node_name}")
    print(f"   • Output Directory: {output_dir}")
    print(f"   • Files Generated: {len(all_files)}")
    print(f"   • Quality Score: {validation.quality_score:.1%}")
    print(f"   • Validation: {'✅ PASSED' if validation.passed else '❌ FAILED'}")

    print("\n🎯 Success Criteria:")
    print(
        f"   {'✅' if not validation.errors else '❌'} Code generation completed without errors"
    )
    print(
        f"   {'✅' if len(required_fields) == sum(1 for f in required_fields if f in contract_data) else '❌'} Contract has all 11+ required fields"
    )
    print(
        f"   {'✅' if validation.onex_compliance_score >= 0.85 else '❌'} ONEX compliance ≥ 85%"
    )
    print(
        f"   {'✅' if validation.quality_score >= 0.85 else '❌'} Quality score ≥ 85%"
    )
    print(f"   {'✅' if len(all_files) >= 10 else '❌'} All files generated (≥10)")
    print(
        f"   {'✅' if validation.passed else '❌'} Generated code follows ONEX patterns"
    )

    print("\n🔍 Next Steps:")
    print(f"   1. Review generated code: {output_dir}")
    print(f"   2. Inspect contract YAML: {output_dir}/contract.yaml")
    print(f"   3. Review node implementation: {output_dir}/node.py")
    print(f"   4. Run tests: pytest {output_dir}/tests/")

    return artifacts, validation, output_dir


async def main():
    """Run the demonstration."""
    try:
        artifacts, validation, output_dir = await generate_mysql_adapter_node()
        print("\n✅ Demonstration completed successfully!")
        print(f"\n📂 Generated files location: {output_dir}")
        return 0
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
