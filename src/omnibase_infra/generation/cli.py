"""CLI interface for ONEX node generation."""

import argparse
import sys
from pathlib import Path

from omnibase_core.core.errors.onex_error import OnexError

from .node_generator import NodeGenerator


def main():
    """Main CLI entry point for node generation."""
    parser = argparse.ArgumentParser(
        description="ONEX Node Generator - Generate node scaffolding from templates"
    )

    parser.add_argument(
        "node_type",
        choices=["effect", "compute", "reducer", "orchestrator"],
        help="Type of node to generate",
    )

    parser.add_argument(
        "--domain",
        required=True,
        help="Domain name (e.g., infrastructure, ai, rsd)",
    )

    parser.add_argument(
        "--microservice",
        required=True,
        help="Microservice name (e.g., postgres_adapter, kafka_wrapper)",
    )

    parser.add_argument(
        "--repository",
        default="omnibase_infra",
        help="Repository name (default: omnibase_infra)",
    )

    parser.add_argument(
        "--description",
        default="",
        help="Business description of the node",
    )

    parser.add_argument(
        "--external-system",
        default="",
        help="External system being integrated (for EFFECT nodes)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Output directory for generated files (default: current directory)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without writing files",
    )

    parser.add_argument(
        "--list-templates",
        action="store_true",
        help="List available templates and exit",
    )

    parser.add_argument(
        "--from-contract",
        type=Path,
        help="Generate node from existing contract.yaml file",
    )

    args = parser.parse_args()

    try:
        generator = NodeGenerator(output_dir=args.output_dir)

        # List templates mode
        if args.list_templates:
            print("Available node templates:")
            for template in generator.list_available_templates():
                print(f"  - {template}")
            return 0

        # Generate from contract mode
        if args.from_contract:
            print(f"Generating node from contract: {args.from_contract}")
            result = generator.generate_from_contract(
                contract_path=args.from_contract,
                dry_run=args.dry_run,
            )
        # Generate from parameters mode
        else:
            print(f"Generating {args.node_type.upper()} node:")
            print(f"  Domain: {args.domain}")
            print(f"  Microservice: {args.microservice}")
            print(f"  Repository: {args.repository}")

            result = generator.generate_node(
                node_type=args.node_type,
                repository_name=args.repository,
                domain=args.domain,
                microservice_name=args.microservice,
                business_description=args.description,
                external_system=args.external_system,
                dry_run=args.dry_run,
            )

        # Print results
        print(f"\nNode directory: {result['node_directory']}")
        print(f"\nGenerated {len(result['generated_files'])} files:")
        for path in result['generated_files']:
            print(f"  - {path}")

        print(f"\nCreated {len(result['init_files'])} __init__.py files:")
        for path in result['init_files']:
            print(f"  - {path}")

        if args.dry_run:
            print("\n[DRY RUN] No files were actually written.")
        else:
            print("\nGeneration complete!")

        return 0

    except OnexError as e:
        print(f"ERROR: {e.message}", file=sys.stderr)
        if e.details:
            print(f"Details: {e.details}", file=sys.stderr)
        return 1

    except Exception as e:
        print(f"UNEXPECTED ERROR: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
